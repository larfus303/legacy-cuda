# Tesla K80 / Kepler: `CUBLAS_STATUS_ARCH_MISMATCH` in whisper.cpp und llama.cpp

**Betrifft:** Kepler-Karten (Compute Capability 3.x, z. B. Tesla K80 = sm_37, K40 = sm_35) mit ggml-basierten Programmen (whisper.cpp, llama.cpp und alles, was ggml einbettet). Fehler C (Abschnitt 2.4) betrifft vermutlich alle Karten unter CC 6.1, siehe dort.

**Status (2026-10-20):** Ein Patch, Fassung 2, Stand des Pull Requests [ggml-org/llama.cpp#30338](https://github.com/ggml-org/llama.cpp/pull/30338) nach der ersten Review-Runde. Damit bestehen auf einer Tesla K80 **alle 2444 `MUL_MAT`-Tests** und **alle 18 477 Tests** von `test-backend-ops`, genau wie auf einer RTX 3090. Die Prefill-Geschwindigkeit von K-Quant-Modellen auf der K80 ist dieselbe wie mit Fassung 1. Der Pull Request wartet auf die Antwort des Reviewers.

| Patch | Behebt | Datei |
|---|---|---|
| **v2** (aktuell) | Fehler A, B und C | [`patches/ggml/cuda-kepler-mul-mat-v2.patch`](../patches/ggml/cuda-kepler-mul-mat-v2.patch) |
| v1, 0001 (überholt) | Fehler A und B | [`patches/ggml/v1/0001-cuda-kepler-avoid-cublas-ex.patch`](../patches/ggml/v1/0001-cuda-kepler-avoid-cublas-ex.patch) |
| v1, 0002 (überholt) | Fehler C, durch Abgabe an die CPU | [`patches/ggml/v1/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch`](../patches/ggml/v1/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch) |

v2 ist ein einzelner Patch auf llama.cpp `23b0202` und entspricht dem Stand des Pull Requests mit Commit `6b3b848`. Die v1-Patches bleiben zur Nachvollziehbarkeit im Ordner `v1/`. Was sich geändert hat und warum, steht in Abschnitt 4.

---

## 1. Symptom

Das Programm bricht während der Inferenz ab:

```
CUDA error: CUBLAS_STATUS_ARCH_MISMATCH
  current device: 0, in function ggml_cuda_mul_mat_cublas_impl at .../ggml-cuda.cu:<Zeile>
  cublasGemmStridedBatchedEx(...)        <- oder cublasGemmBatchedEx / cublasGemmEx
```

oder, wenn nur Fehler A und B behoben sind (z. B. mit v1-Patch 0001 allein):

```
ggml-cuda.cu:<Zeile>: GGML_ASSERT(convert_func != nullptr) failed
```

Typisches Muster:

- Kleine Modelle (whisper tiny, oft auch medium) und viele quantisierte Modelle laufen.
- Größere Modelle oder F16/BF16-Modelle (whisper large-v3, F16-GGUFs in llama.cpp) stürzen ab.

Das sieht nach einem Größenproblem aus, ist aber keins. Entscheidend ist, welcher interne Rechenpfad für eine bestimmte Matrix-Multiplikation gewählt wird (Abschnitt 2.5).

## 2. Ursache: drei getrennte Fehler

Die Gegenmessung (Abschnitt 6) hat gezeigt, dass es nicht ein Fehler ist, sondern drei. Sie verdecken sich gegenseitig: Erst wenn A und B behoben sind, wird C sichtbar.

### 2.1 Hintergrund: cuBLAS kennt zwei Funktionsfamilien

| Familie | Beispiele | Kepler |
|---|---|---|
| typisiert (alt) | `cublasSgemm`, `cublasSgemmBatched`, `cublasSgemmStridedBatched` (S = Single = FP32) | ✅ läuft |
| `…Ex` (Mixed Precision) | `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx` | ❌ `CUBLAS_STATUS_ARCH_MISMATCH` |

In den Release Notes zu CUDA 9.0 schreibt NVIDIA, dass die GemmEx-Routinen nur ab Maxwell unterstützt werden und **nicht auf Kepler, ausdrücklich genannt sind Tesla K40 und K80**. Das gilt **auch, wenn alle Datentypen FP32 sind**. Die Sperre hängt an der Funktion, nicht am Datentyp.

### 2.2 Fehler A: einzelne Matrix, F16/BF16 → `cublasGemmEx`

In `ggml/src/ggml-cuda/ggml-cuda.cu` wählt `ggml_cuda_mul_mat_cublas` den Rechentyp (`compute_type`):

- quantisierte Gewichte → F16, wenn schnelles FP16 vorhanden ist, sonst F32. Auf Kepler also **F32**.
- F16-Gewichte → F32, wenn kein schnelles FP16 vorhanden ist. Auf Kepler also **F32**.
- **BF16-Gewichte** → bleiben auf NVIDIA in vielen Fällen **BF16**, auch auf Kepler.

`ggml_cuda_mul_mat_cublas_impl` ruft für eine einzelne Matrix bei F32 `cublasSgemm` auf (läuft), bei F16/BF16 aber `cublasGemmEx` (Absturz). In `test-backend-ops` ist das der erste Abbruch, schon nach 30 Tests bei `type_a=bf16`.

### 2.3 Fehler B: mehrere Matrizen auf einmal (batched) → `cublas…BatchedEx`

Für batched Matrix-Multiplikationen (z. B. Attention über mehrere Köpfe) ruft ggml **unabhängig vom Rechentyp** immer `cublasGemmStridedBatchedEx` oder `cublasGemmBatchedEx` auf. Der F32-Rechentyp ändert dort nur den Datentyp der Puffer, nicht die aufgerufene Funktion. Auf Kepler stürzt das deshalb **für jeden Gewichtstyp** ab, auch für F32 und für quantisierte Modelle.

Das ist der Fehler, den whisper large-v3 trifft.

### 2.4 Fehler C: nicht-zusammenhängende quantisierte Gewichte ohne Umrechner

Liegt die Gewichtsmatrix nicht zusammenhängend im Speicher (z. B. eine permutierte Ansicht), rechnet ggml sie vor dem cuBLAS-Aufruf mit einer `convert_nc`-Funktion nach F32 (bzw. F16) um. Solche Umrechner gibt es nur für:

> F16, BF16, Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

**Nicht** für Q2_K … Q6_K, IQ-Formate, MXFP4 usw. Dann gilt `convert_func == nullptr`, und das Programm bricht am `GGML_ASSERT` ab.

**Warum das auf neueren Karten nicht auffällt:** Quantisierte Matrix-Multiplikationen laufen dort über ggml-eigene Kernel (MMQ) und erreichen den cuBLAS-Pfad gar nicht. Ungepatcht nutzt ggml MMQ aber erst ab **Compute Capability 6.1**, wo es den Befehl `__dp4a` gibt: `ggml_cuda_should_use_mmq()` gibt darunter sofort `false` zurück, auch mit `GGML_CUDA_FORCE_MMQ`. Unter CC 6.1 landen quantisierte Matrizen mit größerem Batch deshalb immer bei cuBLAS.

Die MMQ-Kernel selbst gibt es trotzdem auch für ältere Karten: ggml hat eine eigene Konfiguration dafür (`mmq-config-pascal-older.cuh`, deckt alle MMQ-Typen ab) und ersetzt `__dp4a` dort durch eine Software-Variante. Darauf baut Fassung 2 des Patches auf (Abschnitt 4.2).

**Vermutlich ebenfalls betroffen (ungetestet):** Maxwell (M40, CC 5.2) und **P100 (CC 6.0)**. Die P40 (CC 6.1) hat MMQ und sollte nicht betroffen sein. Fehler A und B betreffen diese Karten nicht, weil die `…Ex`-Funktionen dort laufen.

### 2.5 Warum es nach „Modellgröße“ aussieht

Welcher Pfad gewählt wird (ggml-eigener Kernel oder cuBLAS, einzeln oder batched), hängt von Gewichtstyp, Tensorform und Batchgröße ab. Kleine oder quantisierte Modelle erreichen die kaputten Pfade zufällig nicht, große F16-Modelle schon.

**Diagnose:** In der Fehlermeldung steht unter `current device: …` die auslösende Anweisung:

| Steht dort … | Fehler | Behoben durch |
|---|---|---|
| `cublasGemmEx(` | A | v2 (v1: 0001) |
| `cublasGemmStridedBatchedEx(` oder `cublasGemmBatchedEx(` | B | v2 (v1: 0001) |
| `GGML_ASSERT(convert_func != nullptr)` | C | v2 (v1: 0002) |

## 3. Was nicht hilft (geprüft)

| Ansatz | Warum nicht |
|---|---|
| `-DGGML_CUDA_FORCE_MMQ=ON` | Ohne Patch gibt `ggml_cuda_should_use_mmq()` unter CC 6.1 `false` zurück, bevor das Flag geprüft wird. |
| `-DGGML_CUDA_FORCE_CUBLAS=ON` | Erzwingt cuBLAS erst recht. |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | Behebt höchstens Fehler A. Batched läuft trotzdem über die `…Ex`-Funktionen (Fehler B). |
| cuBLAS aus CUDA 10.2 nachladen | Die Kepler-Sperre für `…Ex` gilt seit CUDA 9.0. Dazu kommt der ABI-Bruch zwischen `libcublas.so.10` und `.so.11`. |
| Neueres CUDA | Ab Treiber 495 / CUDA 11.5+ gibt es keine Unterstützung mehr für sm_37. Treiber 470 + CUDA 11.4 ist das Ende. |
| Modell quantisieren | Umgeht A teilweise. B betrifft auch quantisierte Modelle, C sogar gerade diese. |

**Notlösung ohne Patch:** Vulkan-Backend (`-DGGML_VULKAN=1`). Treiber 470 bringt Vulkan für Kepler mit. Laut Community-Messungen (llama.cpp Issue #12140) ist es auf der K80 aber nur etwa halb so schnell wie CUDA (ca. 3 T/s gegenüber 5,5–6 T/s).

## 4. Die Fixes

Es gibt zwei Fassungen. Fassung 1 (zwei Patches, 0001 und 0002) war der erste Pull Request vom 2026-10-11. Fassung 2 ist der Stand nach der ersten Review-Runde durch den llama.cpp-Maintainer Johannes Gäßler. Beide beheben dieselben drei Fehler, Fassung 2 aber einfacher und ohne Umweg über die CPU.

### 4.1 Fassung 2 (aktuell): `cuda-kepler-mul-mat-v2.patch`

Vier Änderungen in drei Dateien (`ggml-cuda.cu`, `mmq.cu`, `mmq.cuh`), insgesamt +32/−6 Zeilen:

1. **Rechentyp auf Kepler immer F32** (behebt A). In `ggml_cuda_mul_mat_cublas` wird bei NVIDIA mit `cc < 500` `compute_type = GGML_TYPE_F32` gesetzt. Das übersteuert BF16-Gewichte und die Umgebungsvariable `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, weil jeder andere Rechentyp eine `…Ex`-Funktion braucht. BF16/F16 werden vorher nach F32 umgerechnet, die einzelne Matrix läuft dann über `cublasSgemm`. Unverändert aus Fassung 1.
2. **Batched F32 immer über die typisierten Funktionen** (behebt B). In `ggml_cuda_mul_mat_cublas_impl` gilt jetzt `if constexpr (compute_type == GGML_TYPE_F32)`: `cublasSgemmStridedBatched` statt `cublasGemmStridedBatchedEx` und `cublasSgemmBatched` statt `cublasGemmBatchedEx`. Das gilt **auf allen Karten**, nicht mehr nur auf Kepler, und ohne `#if` für HIP/MUSA (deren Kompatibilitäts-Header bilden beide Funktionen schon ab). So hat es der Reviewer verlangt. Die Rechnung ist mathematisch identisch (gleiche Matrizen, gleiche Strides, F32-Akkumulation).
3. **MMQ statt cuBLAS, wenn der cuBLAS-Weg nicht funktionieren kann** (behebt C). In `ggml_cuda_should_use_mmq` (`mmq.cu`) gibt die Funktion auf Karten unter CC 6.1 jetzt `true` zurück, wenn **beide** Bedingungen gelten:
   - die Gewichtsmatrix liegt nicht zusammenhängend im Speicher,
   - es gibt für den Typ keinen `convert_nc`-Umrechner (`ggml_get_to_fp32_nc_cuda(type) == nullptr`).

   Diese Fälle laufen dann über die MMQ-Kernel für ältere Karten (Abschnitt 2.4) auf der GPU, statt am `GGML_ASSERT` abzustürzen. Alles andere läuft wie vorher über cuBLAS. Die bestehende Regel für Mixture-of-Experts-Modelle ab Pascal bleibt erhalten.
4. **Neuer Parameter `src0_contiguous`** für `ggml_cuda_should_use_mmq` (`mmq.cu`, `mmq.cuh`). Die Funktion bekommt sonst nur Typ, Compute Capability, Batchgröße und Expertenzahl, aber nicht den Tensor selbst. Die drei Aufrufstellen in `ggml-cuda.cu` übergeben `ggml_is_contiguously_allocated(src0)`. Das ist dieselbe Prüfung, mit der der cuBLAS-Pfad entscheidet, ob er überhaupt einen `convert_nc`-Umrechner braucht.

**Was der Reviewer verlangt hat (Review vom 2026-10-11):**

| Anmerkung | Umgesetzt in |
|---|---|
| `#if` und die Architektur-Prüfung entfernen, für FP32 immer `Sgemm` nehmen (strided batched) | Punkt 2 |
| dasselbe für den Pointer-batched-Aufruf | Punkt 2 |
| statt der Sperre in `supports_op` lieber `ggml_cuda_should_use_mmq` ändern, sodass MMQ genommen wird, wenn der cuBLAS-Weg nicht verfügbar ist | Punkte 3 und 4 |

**Zwischenschritt mit Regression (Commit `e0641ca`, 2026-10-19):** Die erste Umsetzung von Punkt 3 prüfte nur, ob ein `convert_nc`-Umrechner existiert, nicht, ob die Matrix überhaupt nicht zusammenhängend ist. Weil es für K-Quants keinen Umrechner gibt, liefen damit auf der K80 **alle** K-Quant-Matrizen über MMQ, auch die zusammenhängenden, die vorher problemlos über cuBLAS liefen. Die Ergebnisse blieben korrekt, aber der Prefill eines Q4_K_M-Modells fiel von rund 140 auf 42,8 t/s, etwa 3,3× langsamer (Messwerte in Abschnitt 6.3). Die Generation war nicht betroffen, sie läuft über einen eigenen Kernel (MMVQ). Commit `6b3b848` (Punkt 4) behebt das: Der Prefill liegt danach wieder auf dem alten Niveau.

**Die Commits im Pull Request:**

| Commit | Inhalt |
|---|---|
| `9081d40` | Fassung 1, Patch 0001 (`…Sgemm*` für F32 nur bei `cc < 500`) |
| `dee4e6f` | Fassung 1, Patch 0002 (Sperre in `supports_op`) |
| `e0641ca` | Review umgesetzt: `Sgemm` für FP32 immer, Sperre entfernt, MMQ ohne Umrechner (mit Regression) |
| `6b3b848` | MMQ nur für nicht zusammenhängende Matrizen ohne Umrechner (Regression behoben) |

`cuda-kepler-mul-mat-v2.patch` ist die Summe dieser vier Commits gegenüber `23b0202`.

**Was das auf anderen Karten ändert:** Punkte 1, 3 und 4 greifen nur unter CC 5.0 bzw. unter CC 6.1, neuere Karten nehmen denselben Pfad wie vorher. Punkt 2 betrifft alle Karten, aber nur den Fall „batched, Rechentyp F32“. Auf einer RTX 3090 bestehen weiterhin alle Tests (Abschnitt 6.3). Ein Geschwindigkeitsvergleich für genau diesen Fall auf der 3090 wurde nicht gemacht. Übliche Modelle rechnen dort meist mit F16 und nehmen den F32-Pfad selten.

### 4.2 Fassung 1 (überholt): Patches 0001 und 0002

Zur Nachvollziehbarkeit, weil die Messungen in Abschnitt 6.1 und 6.2 damit entstanden sind.

**0001: keine `…Ex`-Aufrufe auf Kepler (Fehler A und B).** Drei Änderungen in `ggml-cuda.cu`:

1. Rechentyp auf Kepler immer F32 (wie Fassung 2, Punkt 1).
2. Strided batched: bei F32 **und** `cc < 500` `cublasSgemmStridedBatched` statt `cublasGemmStridedBatchedEx`.
3. Pointer-batched: bei F32 **und** `cc < 500` `cublasSgemmBatched` statt `cublasGemmBatchedEx`.

Punkte 2 und 3 standen in `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`, alle drei griffen nur bei CC < 5.0. (Eine allererste Fassung hatte die Umschaltung für F32 auf allen Karten aktiv und wurde vor dem Pull Request auf Kepler beschränkt. Der Reviewer hat genau die allgemeine Variante bevorzugt, daher Fassung 2, Punkt 2.)

**0002: Spezialfälle an die CPU abgeben (Fehler C).** +11 Zeilen in `ggml_backend_cuda_device_supports_op`, Abschnitt `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`. Die GPU meldete die Operation als „nicht unterstützt“, wenn **alle** Bedingungen galten:

- die Gewichte sind quantisiert,
- sie liegen nicht zusammenhängend im Speicher (`!ggml_is_contiguously_allocated`),
- die Karte hat kein MMQ (`ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A`),
- es gibt keinen `convert_nc`-Umrechner für den Typ (`ggml_get_to_fp32_nc_cuda(type) == nullptr`).

ggml rechnete dann genau diese Operation auf der CPU. In `test-backend-ops` zählten diese 21 Fälle deshalb als „not supported“ (2423 statt 2444 Tests). Fassung 2 rechnet sie stattdessen mit MMQ auf der GPU.

**Weiterhin denkbar, aber nicht nötig:** `convert_nc`-Umrechner für K-Quants und IQ-Formate ergänzen. Dann ginge auch dieser Fall über cuBLAS. Das wäre deutlich mehr Code.

## 5. Anleitung

### 5.1 Voraussetzungen

| Komponente | Version |
|---|---|
| Treiber | NVIDIA 470-Serie (getestet: 470.256.02), die letzte mit Kepler-Unterstützung |
| CUDA Toolkit | 11.4 (Toolkit only, Treiber nicht überschreiben) |
| Compiler | gcc/g++ 10 (CUDA 11.4 akzeptiert keinen neueren Host-Compiler) |

**Wichtig:** `CC=gcc-10 CXX=g++-10` allein reicht nicht. nvcc muss zusätzlich ausdrücklich `-DCMAKE_CUDA_HOST_COMPILER=g++-10` bekommen. Sonst nimmt nvcc den System-Compiler, und auf neueren Systemen (z. B. glibc 2.39, Ubuntu 24.04) bricht der Build ab.
| CMake | ≥ 3.18 |

### 5.2 llama.cpp bauen

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 23b0202                      # getesteter Stand
git apply /pfad/zu/legacy-cuda/patches/ggml/cuda-kepler-mul-mat-v2.patch

CC=gcc-10 CXX=g++-10 cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-10 \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**K80 und neuere Karte im selben Rechner (Mischbetrieb):** Soll ein einziges Programm beide Karten nutzen, braucht der Build beide Architekturen, z. B. für K80 und RTX 3090:

```bash
  -DCMAKE_CUDA_ARCHITECTURES="37;86"
```

Ein Build nur für `86` bricht ab, sobald er auf der K80 rechnen soll (`no kernel image is available for execution on the device`). Ein Build nur für `37` läuft nicht auf der 3090. Details in Abschnitt 6.6.

**Welche GPU ist welche?** CUDA nummeriert die Karten ohne weitere Angabe nach Geschwindigkeit, `nvidia-smi` nach PCI-Steckplatz. Damit `CUDA_VISIBLE_DEVICES` dieselben Nummern wie `nvidia-smi` meint, immer zusätzlich setzen:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=index,name,pci.bus_id,temperature.gpu,power.limit --format=csv
```

Eine K80 erscheint dabei als **zwei** Geräte (zwei GK210-Chips auf einer Karte).

### 5.3 whisper.cpp bauen

Gleich wie llama.cpp, nur im whisper.cpp-Repo. Der ggml-Code ist derselbe. Getestet ist bisher nur v1-Patch 0001 auf whisper.cpp `d1be6fd` mit large-v3. Ob v2 auf einen whisper.cpp-Stand passt, hängt davon ab, wie aktuell dessen ggml-Kopie ist. Das ist nicht geprüft.

```bash
./models/download-ggml-model.sh large-v3
./build/bin/whisper-cli -m models/ggml-large-v3.bin -f samples/jfk.wav
```

### 5.4 Neuere Versionen

Falls `git apply` auf einem neueren Stand fehlschlägt:

```bash
git apply -3 /pfad/zu/cuda-kepler-mul-mat-v2.patch
```

Sonst die Stellen von Hand anpassen:

```bash
grep -n "cublasGemmEx\|cublasGemmBatchedEx\|cublasGemmStridedBatchedEx" ggml/src/ggml-cuda/ggml-cuda.cu
```

Jeder Treffer muss auf Kepler unerreichbar sein bzw. für F32 durch die `cublasSgemm*`-Entsprechung ersetzt werden. Für Fehler C zusätzlich `ggml_cuda_should_use_mmq` in `mmq.cu` wie in Abschnitt 4.1, Punkte 3 und 4, anpassen.

Wenn der Pull Request in llama.cpp aufgenommen wird, ist auf neueren Ständen kein Patch mehr nötig.

### 5.5 Selbst nachmessen

Das Skript [`scripts/gegenmessung-k80.sh`](../scripts/gegenmessung-k80.sh) misst in zwei Teilen:

- **Teil 1, K80:** `test-backend-ops -o MUL_MAT` ohne Patch und mit v2, Build für sm_37.
- **Teil 2, neuere Karte** (z. B. RTX 3090): dasselbe mit einem eigenen Build für die neue Architektur.

Wird ein Modell angegeben, läuft in beiden Teilen zusätzlich `llama-bench` mit und ohne Patch. Für die K80 ein K-Quant-Modell nehmen (z. B. Q4_K_M), weil sich dort der Prefill-Pfad am ehesten unterscheidet. Ohne Patch kann `llama-bench` auf der K80 abbrechen, das ist dann Fehler A, B oder C.

(Die Messungen von Fassung 1 in Abschnitt 6.1 stammen von einer älteren Version des Skripts mit drei K80-Läufen: ohne Patch, nur 0001, 0001+0002.)

Am Ende setzt es den Quellcode wieder zurück. Das Ergebnis steht in `zusammenfassung.txt`.

```bash
./scripts/gegenmessung-k80.sh /pfad/zu/llama.cpp patches/ggml/cuda-kepler-mul-mat-v2.patch [/pfad/zu/modell.gguf]
```

Einstellungen über Umgebungsvariablen (Nummern wie in `nvidia-smi`, das Skript setzt `CUDA_DEVICE_ORDER=PCI_BUS_ID`):

| Variable | Standard | Bedeutung |
|---|---|---|
| `K80_GPU` | 2 | Nummer der K80-Hälfte. Die kühlere, nicht gedrosselte nehmen (Abschnitt 6.5). |
| `NEW_GPU` | 0 | Nummer der neueren Karte |
| `NEW_ARCH` | 86 | Architektur der neueren Karte (86 = RTX 30xx) |
| `SKIP_K80` / `SKIP_NEW` | – | auf `1` setzen, um einen Teil zu überspringen |

**Wichtig bei mehreren Karten im System:** Pro Lauf ist nur eine Karte sichtbar, und jeder Build enthält nur eine Architektur. Für Mischbetrieb siehe Abschnitt 5.2 und 6.6.

## 6. Testprotokoll

Alle Messungen auf dem Server „bodo“: Tesla K80 und RTX 3090 (Gigabyte Turbo) im selben Rechner, Xeon E5-2699 v4, Treiber 470.256.02, CUDA 11.4, gcc 10.

**Bezeichnung der GPUs** (mit `CUDA_DEVICE_ORDER=PCI_BUS_ID`, gemessen 2026-10-19):

| Nummer | Karte | Leistungsgrenze | Bemerkung |
|---|---|---|---|
| 0 | RTX 3090 | 280 W | |
| 1 | Tesla K80, Hälfte „K80-A“ | 110 W (bewusst gedrosselt, Maximum wäre 175 W) | die heißere Hälfte, 57 °C |
| 2 | Tesla K80, Hälfte „K80-B“ | 175 W (Maximum) | die kühlere Hälfte, 39 °C |

Die Temperaturen sind eine Momentaufnahme beim Ablesen, nicht unter Last.

### 6.1 Fassung 1: Gegenmessung `test-backend-ops -o MUL_MAT`

Stand llama.cpp `23b0202`, Skript `scripts/gegenmessung-k80.sh` (ältere Version mit drei K80-Läufen). Gemessen am 2026-10-11 (Endfassung von Fassung 1, 0001 mit Beschränkung auf `cc < 500`). Die K80-Läufe liefen über die Nummer 1, also vermutlich auf K80-A. Für die Korrektheit ist das unerheblich.

**Tesla K80** (eine GPU-Hälfte, sm_37):

| Lauf | Bestandene Tests | Abbruch | Exit |
|---|---|---|---|
| ohne Patch | 30 | `CUBLAS_STATUS_ARCH_MISMATCH` (Fehler A, bei `bf16`) | 134 |
| nur 0001 | 1103 | `GGML_ASSERT(convert_func != nullptr)` (Fehler C) | 134 |
| 0001 + 0002 | **2423 / 2423** | – | 0 |

Zusatzlauf ohne Patch, nur `-p "type_a=q5_K"`: 63 bestanden, dann `CUBLAS_STATUS_ARCH_MISMATCH` aus `cublasGemmBatchedEx`. Das belegt, dass Fehler B auch quantisierte Typen betrifft.

**RTX 3090** (sm_86), Gegenprobe auf neuerer Karte:

| Lauf | Bestandene Tests | Exit |
|---|---|---|
| ohne Patch | 2444 / 2444 | 0 |
| 0001 + 0002 | 2444 / 2444 | 0 |

Die Differenz von 21 Tests (2444 gegenüber 2423) ist nachgezählt: Mit 0001+0002 meldet die K80 441 Fälle als „not supported“, die 3090 420. Die 21 zusätzlichen Fälle auf der K80 sind genau **q4_K, q5_K und q6_K** in einer nicht zusammenhängenden Ansicht (`m=1000, k=1024, k_v=1280`) mit `n` ∈ {2, 3, 5, 8, 9, 13, 16}, also 3 Typen × 7 Werte. Auf der 3090 rechnet MMQ diese Fälle, auf der K80 gibt 0002 sie an die CPU ab. IQ-Formate, TQ2_0 und MXFP4 tauchen in der Differenz nicht auf, weil sie in dieser Ansicht auf beiden Karten nicht unterstützt sind.

Auf der 3090 greift keiner der Patches. Die Bedingungen `cc < 500` (0001) und „kein MMQ“ (0002) sind dort nicht erfüllt, der Code-Pfad ist derselbe.

**llama-bench auf der RTX 3090** (Mistral-7B Q4_K_M, `-ngl 99 -r 5`), zweimal in umgekehrter Reihenfolge gemessen:

tg128 (t/s):

| Messreihe | 1. Lauf | 2. Lauf |
|---|---|---|
| A: ohne → mit | ohne 131,88 ± 0,65 | mit 129,93 ± 0,66 |
| B: mit → ohne | mit 134,81 ± 0,34 | ohne 133,36 ± 0,60 |

pp512 (t/s):

| Messreihe | 1. Lauf | 2. Lauf |
|---|---|---|
| A: ohne → mit | ohne 4557,33 ± 132,82 | mit 4501,30 ± 112,31 |
| B: mit → ohne | mit 4549,26 ± 138,59 | ohne 4582,33 ± 120,20 |

Mittelwert über beide Reihenfolgen:

| | ohne Patch | mit 0001 + 0002 | Differenz |
|---|---|---|---|
| tg128 | 132,62 | 132,37 | −0,2 % |
| pp512 | 4570 | 4525 | −1,0 % (Streuung ±120–140) |

In beiden Messreihen ist der **zweite** Lauf rund 1 % langsamer, egal ob mit oder ohne Patch. Der Unterschied hängt also an der Reihenfolge (vermutlich leichte thermische Drosselung über die Laufzeit), nicht am Patch. Über beide Reihenfolgen gemittelt bleibt kein messbarer Effekt. Das passt zum Code: Auf der 3090 läuft derselbe Pfad wie ohne Patch.

### 6.2 Anwendung

| Datum | Karte | Software | Modell | Ergebnis |
|---|---|---|---|---|
| 2026-10-10 | Tesla K80 | whisper.cpp `d1be6fd` + 0001 | large-v3 | ✅ läuft |

Alle whisper-Messungen in diesem Abschnitt liefen mit v1-Patch 0001. Mit Fassung 2 ist whisper noch nicht gemessen.

#### Faire Vergleichsmessung (2026-10-12) – maßgeblich

whisper large-v3 (mit 0001), Audiodatei 30,8 s, je **drei Läufe gemittelt**, CPU-Takt-Governor `performance`:

| System | Gesamt | Encoder | Rest (Gesamt − Encoder) | RTF |
|---|---|---|---|---|
| Tesla K80, **eine** der beiden GPU-Hälften | **10,42 s** | 3,46 s | 6,96 s | 0,34 |
| Xeon E5-2699 v4, **22 Threads** (alle physischen Kerne) | 17,55 s | 13,23 s | 4,32 s | 0,57 |
| Xeon E5-2699 v4, 44 Threads (mit Hyperthreading) | 16,63 s | 12,68 s | 3,95 s | 0,54 |

**Ergebnis:**

- **Gesamt ist die K80-Hälfte etwa 1,7× schneller** als dieser Xeon mit allen Kernen (17,55 s / 10,42 s = 1,68×; gegen 44 Threads 1,60×).
- **Der Encoder läuft auf der K80 etwa 3,8× schneller** (13,23 s / 3,46 s). Hier liegt der eigentliche Gewinn der Karte.
- **Der Rest (Decoder, Audio-Verarbeitung, Datentransfer) ist auf der K80-Variante langsamer** als auf der reinen CPU (6,96 s gegen 4,32 s). Das verwässert den Encoder-Gewinn auf 1,7× gesamt. Warum, ist nicht untersucht. Plausibel sind die tokenweise Decodierung mit vielen kleinen Kernel-Aufrufen und die PCIe-Transfers auf der alten Karte.
- **Hyperthreading bringt kaum etwas:** 44 statt 22 Threads sind nur rund 5 % schneller.

#### Korrektur der ersten Messung (2026-10-10) – überholt

Die erste Messung wird hier nur zur Nachvollziehbarkeit aufgeführt. **Die dort genannten 3,1× sind kein gültiger Vergleich K80 gegen CPU.**

| System | Laufzeit | RTF | Encoder |
|---|---|---|---|
| Tesla K80, eine GPU-Hälfte | 17,8 s | 0,58 | 5,07 s |
| Xeon E5-2699 v4, **nur 12 Threads**, Governor `schedutil` | 55,3 s | 1,79 | 34,99 s |

Die Messung war in zwei Richtungen verzerrt:

1. **Die CPU lief nur mit 12 von 22 Kernen** und zusätzlich unter dem drosselnden Governor `schedutil`. Ihr Wert war dadurch viel zu schlecht (55,3 s statt 17,55 s).
2. **Auch der K80-Wert war zu schlecht** (17,8 s statt 10,42 s). Die CPU-seitigen Anteile des K80-Laufs (Decoder, Audio, Transfers) wurden ebenfalls gedrosselt. Dass der Unterschied überwiegend vom Governor kommt, ist wahrscheinlich, aber nicht isoliert gemessen. Andere Bedingungen am 10.10. (z. B. weitere laufende Prozesse) sind nicht ausgeschlossen.

**Einordnung, damit die Zahlen nicht falsch verstanden werden:**

- Verglichen wurde eine bestimmte, ältere Server-CPU (Broadwell, AVX2, ohne AVX-512), keine CPU allgemein. Eine aktuelle Desktop-CPU (z. B. mit AVX-512 und DDR5) dürfte whisper deutlich schneller rechnen und kann die K80 überholen. Das ist nicht gemessen.
- Die Audiodatei ist kurz (30,8 s). Feste Kosten wie Laden und Initialisierung haben deshalb einen großen Anteil. Mit einer längeren Aufnahme kann sich das Verhältnis verschieben.
- Ob der CPU-Build mit BLAS (z. B. OpenBLAS) gebaut war, ist weiterhin nicht festgehalten.
- Auf der K80 lief nur eine der beiden GPU-Hälften. Für einen einzelnen whisper-Auftrag spielt das kaum eine Rolle, weil er nicht auf zwei GPUs verteilt wird.
- **Für eigene Vergleiche:** Governor auf `performance` stellen (und danach zurück), alle physischen Kerne nutzen, mehrere Läufe mitteln. Sonst wird nicht nur die CPU, sondern auch eine GPU-Messung verfälscht, weil auch sie CPU-Anteile hat.

### 6.3 Fassung 2: Korrektheit und Prefill-Geschwindigkeit

Stand: Pull-Request-Branch auf `23b0202`. Gemessen am 2026-10-18/19.

**`test-backend-ops -o MUL_MAT`:**

| Stand | Tesla K80 (sm_37) | RTX 3090 (sm_86) |
|---|---|---|
| `e0641ca` (Review umgesetzt, mit Regression) | 2444 / 2444 (auf K80-A) | 2444 / 2444 |
| `6b3b848` (Fassung 2, Endstand) | 2444 / 2444 (welche Hälfte, ist nicht festgehalten) | 2444 / 2444 |

Die K80 besteht damit dieselben 2444 Tests wie die 3090. Die 21 Fälle, die Fassung 1 an die CPU abgab (q4_K, q5_K und q6_K in der nicht zusammenhängenden Ansicht `m=1000, k=1024, k_v=1280`), laufen jetzt über MMQ auf der GPU und bestehen.

**`llama-bench` auf K80-B** (Nummer 2), `-ngl 99 -r 5`, Modelle in Q4_K_M:

| Modell | Stand | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| Mistral-7B | `dee4e6f` (Fassung 1, cuBLAS) | 140,8 | 12,1 |
| Mistral-7B | `e0641ca` (alle K-Quants über MMQ) | 42,8 | 12,0 |
| Mistral-7B | `6b3b848` (Fassung 2) | 140,16 ± 0,38 | 12,04 ± 0,05 |
| Llama-3.1-8B | `dee4e6f` (Fassung 1, cuBLAS) | 138,1 | 11,3 |
| Llama-3.1-8B | `e0641ca` (alle K-Quants über MMQ) | 42,8 | 11,3 |
| Llama-3.1-8B | `6b3b848` (Fassung 2) | 140,06 ± 0,41 | 11,28 ± 0,03 |

Für `dee4e6f` und `e0641ca` sind nur die Mittelwerte festgehalten, ohne Streuung. Die Werte für `6b3b848` wurden in einer späteren Sitzung gemessen. Abweichungen von 1 bis 2 t/s zum alten Wert sind normale Schwankung zwischen den Sitzungen (Temperatur, Takt). Die Llama-Werte liegen mit Fassung 2 sogar etwas über der alten Messung.

**Ergebnis:**

- Fassung 2 ist beim Prefill so schnell wie Fassung 1. Die 3,3-fache Verlangsamung des Zwischenstands `e0641ca` ist vollständig behoben.
- MMQ ist auf der K80 für zusammenhängende K-Quant-Matrizen rund 3,3× langsamer als dequantisieren plus cuBLAS. Deshalb nutzt Fassung 2 MMQ dort nur, wo cuBLAS nicht funktioniert.
- Die Generation (tg128) ändert sich in keinem Stand. Sie läuft über einen eigenen Kernel (MMVQ), nicht über MMQ oder cuBLAS.

### 6.4 Alle Operationen: `test-backend-ops` ohne Filter

Gemessen am 2026-10-19, Code-Stand `e0641ca` (der Build enthielt diesen Stand bereits vor dem Commit), ohne `-o`, also alle Operationen:

| Karte | Build | Ergebnis |
|---|---|---|
| Tesla K80, K80-B (Nummer 2) | nur sm_37 | **18 477 / 18 477** bestanden, 0 FAIL |
| RTX 3090 (Nummer 0) | nur sm_86 | **18 477 / 18 477** bestanden, 0 FAIL |

Die Änderung von `e0641ca` zu `6b3b848` betrifft nur, ob eine Matrix-Multiplikation über MMQ oder cuBLAS läuft. Beide Wege sind in 6.3 einzeln geprüft. Ein voller Lauf mit `6b3b848` steht noch aus.

**„Bestanden“ zählt nur, was das CUDA-Backend für sich beansprucht.** Was es als „not supported“ meldet, rechnet ggml im Betrieb auf der CPU. Diese Fälle, nach Operation gruppiert:

| Operation | „not supported“ auf der K80 | auf der 3090 |
|---|---|---|
| OUT_PROD | 1344 | 1344 |
| FLASH_ATTN_EXT | 1339 | 1339 |
| CONV_2D | 1138 | 1138 |
| IM2COL_3D | 1024 | 1024 |
| SET_ROWS | 568 | 568 |
| MUL_MAT | 420 | 420 |
| CPY | 349 | 349 |
| GET_ROWS_BACK | 56 | 56 |
| ARGSORT | 28 | 28 |
| SOFT_MAX_BACK | 24 | 24 |
| GET_ROWS | 24 | 24 |
| ROPE | 20 | 20 |
| LIGHTNING_INDEXER | 18 | 18 |
| CONCAT | 15 | 15 |
| MUL_MAT_ID | 14 | 14 |
| NORM | 10 | 10 |
| L2_NORM | 10 | 10 |
| REPEAT | 6 | 6 |
| XIELU | 2 | 2 |
| ACC | 2 | 2 |
| SUM | 1 | 1 |

Gruppiert wurde mit:

```bash
grep "not supported" alle-ops-k80.log | cut -d'(' -f1 | sort | uniq -c | sort -rn
```

**Die Listen sind auf beiden Karten identisch.** Es gibt also keine Operation, die nur auf der K80 fehlt. Was hier steht, fehlt im aktuellen CUDA-Backend von llama.cpp allgemein (z. B. bestimmte Varianten, sehr große Arrays bei ARGSORT, die ternären Typen tq1_0/tq2_0 bei MUL_MAT, Operationen nur fürs Training wie GET_ROWS_BACK und SOFT_MAX_BACK). Bei FLASH_ATTN_EXT betrifft das nur einzelne Testvarianten. Flash Attention lässt sich auf der K80 einschalten und bringt messbar etwas (Abschnitt 6.5).

**Ergebnis:** Nach diesem Test gibt es auf der K80 nichts Kepler-Spezifisches mehr zu patchen. Sie ist bei der Op-Unterstützung auf demselben Stand wie die 3090, nur langsamer.

### 6.5 Betrieb: Einstellungen, Takt und Leistung

**llama.cpp-Optionen auf K80-B** (Mistral-7B Q4_K_M, `-ngl 99 -r 3`, gemessen 2026-10-19; Code-Stand vermutlich `6b3b848`, derselbe Build wie die Fassung-2-Werte in 6.3, aber nicht ausdrücklich festgehalten):

| `-ub` | `-fa` | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| 256 | 0 | 115,73 ± 0,20 | 11,57 ± 0,02 |
| 256 | 1 | 115,44 ± 0,07 | 12,02 ± 0,01 |
| 512 | 0 | 136,59 ± 0,10 | 11,52 ± 0,00 |
| 512 | 1 | 138,57 ± 0,36 | 12,01 ± 0,01 |
| 1024 | 0 | 136,43 ± 0,34 | 11,52 ± 0,00 |
| 1024 | 1 | 138,29 ± 0,10 | 12,00 ± 0,00 |

- **`-fa 1` (Flash Attention) einschalten.** Die Generation wird bei jedem `-ub`-Wert rund 4 % schneller (11,5 → 12,0 t/s), der Prefill mit `-ub 512` um 1,4 %. Kostet weder Leistung noch Kühlung. Für Dienste auf der K80 ist das die einzige Einstellung, die gegenüber der Voreinstellung messbar etwas bringt.
- **`-ub` (Micro-Batch) nicht unter 512 setzen.** 512 ist die Voreinstellung von llama-bench und llama-server. 256 kostet rund 15 % Prefill, 1024 bringt nichts zusätzlich. Auf die Generation hat `-ub` keinen Einfluss.
- Die pp512-Werte mit `-fa 0 -ub 512` liegen hier bei 136,6 statt 140 t/s wie in 6.3. Der Unterschied liegt an anderen Wiederholungen (`-r 3` statt `-r 5`) und einer anderen Sitzung, nicht am Patch.

**Takt und Leistungsgrenze** (gemessen 2026-10-19):

- K80-B läuft unter Last an ihrer Leistungsgrenze von 175 W, die zugleich das Maximum ist. Sie taktet deshalb mit 810 MHz statt der höchstmöglichen 875 MHz. Ein fester Maximaltakt (`nvidia-smi -ac 2505,875`) bringt hier nichts, weil die Leistungsgrenze bremst, nicht die Taktregelung.
- K80-A ist auf 110 W gedrosselt, weil sie heißer läuft. Hier wären bis zu 175 W möglich, aber nur mit besserer Kühlung und Temperaturüberwachung. Bei der K80 strömt die Luft durch einen gemeinsamen Kühlkörper vermutlich erst über den einen Chip und dann über den anderen. Das würde erklären, warum eine Hälfte deutlich heißer ist. Nicht überprüft.
- **ECC:** Die K80 hat die Speicher-Fehlerkorrektur ab Werk eingeschaltet. Darum zeigt jede Hälfte 11 441 MiB statt 12 GB. Ausschalten (`nvidia-smi -i <n> -e 0`, wirkt nach Neustart) gibt Speicher frei und könnte die Generation etwas beschleunigen, weil diese von der Speicherbandbreite abhängt. **Nicht gemessen.** Nachteil: Ein seltener Speicherfehler wird nicht mehr erkannt.

### 6.6 Mischbetrieb K80 + RTX 3090

Gemessen am 2026-10-19 und 2026-10-20 von Max auf „bodo“. Ausführlicher Messbericht mit allen Erklärungen: [`messberichte/messbericht-k80-mischbetrieb-2026-10-20.md`](messberichte/messbericht-k80-mischbetrieb-2026-10-20.md).

**Build:** ein einziges Binary für beide Architekturen, `-DCMAKE_CUDA_ARCHITECTURES="37;86"`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, Code-Stand `6b3b848`. Der Mischbuild braucht dieselben drei Angaben wie jeder K80-Build: `CMAKE_CUDA_COMPILER` mit Pfad zu nvcc 11.4, `CMAKE_CUDA_HOST_COMPILER` gcc/g++ 10 und beide Architekturen. CUDA 11.4 akzeptiert die Kombination `37;86`.

**Gegenprobe:** Ein Build nur für sm_86, gestartet auf der K80, bricht ab mit `CUDA error: no kernel image is available for execution on the device`. Das ist das Muster des ursprünglichen Fehlers (Abschnitt 7): Für die K80 fehlt im Binary der passende Code.

**Modell und Optionen:** Mistral-7B-Instruct-v0.3 Q4_K_M (4,07 GiB), `-ngl 99 -fa 1 -ub 512 -r 3`.

**1. Das Mischbinary auf den einzelnen K80-Hälften:**

| Stand | Gerät | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| `e0641ca` (vor dem Fix) | K80-A (110 W) | 38,95 ± 0,20 | 8,86 ± 0,02 |
| `6b3b848` | K80-A (110 W) | 101,93 ± 0,77 | 8,96 ± 0,01 |
| `6b3b848` | K80-B (175 W) | 140,96 ± 0,14 | 12,17 ± 0,08 |

Der Fix wirkt auch im Mischbinary. K80-B liegt auf dem Niveau der Messung aus Abschnitt 6.3 (140,16 t/s, Differenz 0,6 %). Der Wert 38,95 t/s vor dem Fix wurde auf K80-A gemessen, die Regression in 6.3 (42,8 t/s) auf K80-B. Beides passt zusammen, die Hälften sind wegen der unterschiedlichen Leistungsgrenze aber nicht direkt vergleichbar. Der Unterschied zwischen K80-A und K80-B kommt von der Leistungsgrenze, nicht vom Patch.

**2. 3090 und K80-B gemeinsam** (`CUDA_VISIBLE_DEVICES=0,2`, Layer-Split `-sm layer`):

```
ggml_cuda_init: found 2 CUDA devices (Total VRAM: 35707 MiB):
  Device 0: RTX 3090, CC 8.6, VRAM 24266 MiB
  Device 1: K80, CC 3.7, VRAM 11441 MiB
```

| pp512 (t/s) | tg128 (t/s) |
|---|---|
| 490,00 ± 2,48 | 35,17 ± 0,05 |

Beide Architekturen rechnen gleichzeitig in einem Programm, ohne Fehler. Schneller als die 3090 allein (135 t/s Generation) ist das nicht: Beim Layer-Split rechnet jede Karte ihre Schichten nacheinander, die langsame K80 bestimmt die Gesamtzeit mit.

**3. Rechnet der Mischbetrieb richtig?** `llama-perplexity`, WikiText-2 Test-Set, `-c 512`, `-sm layer`:

| Lauf | Perplexity |
|---|---|
| nur 3090 | 6,2250 ± 0,03760 |
| 3090 + K80-B | 6,2249 ± 0,03760 |

Die Differenz von 0,0001 liegt weit innerhalb der Messunsicherheit von ±0,038. Der Mischbetrieb rechnet praktisch identisch zum Lauf auf einer Karte. Der Lauf dauerte im Mischbetrieb etwa 37 Minuten, nur auf der 3090 etwa 1,5 Minuten.

**4. Ein Modell, das auf keine einzelne Karte passt:** Dolphin-2.9.1-Llama-3-70B Q4_K_M (39,59 GiB) über alle drei GPUs (`CUDA_VISIBLE_DEVICES=0,1,2`, `-sm layer`, `-r 1`, ohne Streuungsangabe):

| Konfiguration | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 3090 + K80-A + K80-B | 19,22 | 2,45 |
| nur CPU (22 Threads, ohne sichtbare GPU, Messmethode wie in 6.7) | 8,27 | 1,58 |

Das Modell ist zu groß für die 24 GiB der 3090, passt aber in die rund 46 GiB aller drei GPUs. Der Split ist beim Prefill 2,3× und bei der Generation 1,55× schneller als die CPU. K80-A lief dabei gedrosselt mit 110 W.

**Ergebnis:** Der Mischbetrieb funktioniert und rechnet korrekt. Für Modelle, die auf die 3090 allein passen, bremst er sie aber nur aus. Sinnvoll ist er für Modelle, die nur mit dem Speicher aller Karten zusammen auf die GPUs passen. Sonst ist es besser, die Karten getrennt zu nutzen: das Hauptmodell auf der 3090, Nebenaufgaben auf der K80.

### 6.7 LLM-Inferenz: K80 gegen CPU und RTX 3090

Mistral-7B-Instruct-v0.3 Q4_K_M, Mischbuild `6b3b848`, gemessen am 2026-10-20. GPU-Läufe mit `-ngl 99 -fa 1 -ub 512 -r 3`, CPU-Lauf mit `CUDA_VISIBLE_DEVICES=` (leer) und `-ngl 0 -t 22 -r 3`:

| Konfiguration | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| RTX 3090 allein | 4466,67 ± 312,75 | 135,01 ± 0,34 |
| 3090 + K80-B (Layer-Split) | 490,00 ± 2,48 | 35,17 ± 0,05 |
| K80-B (175 W) allein | 140,96 ± 0,14 | 12,17 ± 0,08 |
| K80-A (110 W) allein | 101,93 ± 0,77 | 8,96 ± 0,01 |
| CPU, Xeon E5-2699 v4, 22 Threads | 80,05 ± 0,42 | 14,40 ± 0,06 |

- **Prefill:** Die K80 ist schneller als die CPU. K80-B schafft 1,76× so viel (140,96 gegen 80,05 t/s), selbst die gedrosselte K80-A liegt vorn.
- **Generation:** Die CPU ist knapp schneller als K80-B (14,40 gegen 12,17 t/s) und deutlich schneller als K80-A. Das liegt nicht an der Drosselung, denn auch die volle 175-W-Hälfte liegt dahinter.
- **Die 3090** ist in beiden Werten weit vorn, bei der Generation etwa zehnmal so schnell wie CPU und K80.
- Das passt zum whisper-Vergleich in 6.2, wo die K80 rund 1,7× schneller war als dieselbe CPU, vor allem im Encoder. Große Matrix-Multiplikationen (Prefill, whisper-Encoder) nutzen die K80 gut, die tokenweise Generation nicht.

**Achtung bei eigenen CPU-Messungen:** Ein llama.cpp-Build mit CUDA-Backend rechnet bei `-ngl 0` große Batches (ab 32 Tokens) trotzdem auf einer sichtbaren GPU. Die Gewichte bleiben im RAM, der Prefill läuft aber auf der GPU. Ein erster Lauf ohne diese Vorsicht ergab so 836 t/s (7B) und 96,6 t/s (70B) Prefill „auf der CPU“, tatsächlich gerechnet auf der 3090. Für echte CPU-Werte `CUDA_VISIBLE_DEVICES=` leer setzen (oder einen Build ohne CUDA nehmen). Im Log steht dann `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`.

## 7. Offene Punkte

- **Upstream:** Pull Request [ggml-org/llama.cpp#30338](https://github.com/ggml-org/llama.cpp/pull/30338), eingereicht 2026-10-11. Erste Review-Runde umgesetzt (Abschnitt 4.1), Endstand `6b3b848` und Messwerte am 2026-10-20 im Pull Request gemeldet. Offen ist, ob der Reviewer den neuen Parameter `src0_contiguous` so annimmt oder die Prüfung lieber an den Aufrufstellen hätte.
- **Voller `test-backend-ops`-Lauf mit `6b3b848`:** steht noch aus (Abschnitt 6.4 lief mit `e0641ca`).
- **Mehrere GPUs gemischter Generation: erledigt** (Abschnitt 6.6). Ursprünglicher Fehler mit RTX 3090 und K80 gleichzeitig sichtbar: `CUDA kernel mul_mat_vec_f has no device code compatible with CUDA arch 370. ggml-cuda.cu was compiled for: __CUDA_ARCH_LIST__`. Mit einem Build für `37;86` laufen 3090 und K80 gemeinsam und rechnen korrekt. Dass in der alten Meldung wörtlich `__CUDA_ARCH_LIST__` stand statt einer Architekturliste, deutet darauf hin, dass CUDA 11.4 dieses Makro noch nicht kennt (es kam mit CUDA 11.5). Mit genau dem damaligen Binary nachgestellt wurde der Fehler nicht.
- **whisper mit Fassung 2:** noch nicht gemessen.
- **ECC aus:** Effekt auf die Generation nicht gemessen (Abschnitt 6.5).
- **Fehler C auf M40 und P100:** vermutlich ebenfalls betroffen, ungetestet. Mit Fassung 2 nutzen auch diese Karten MMQ für die betroffenen Fälle, das ist ebenfalls ungetestet.
- **whisper K80 gegen CPU, Restpunkte:** faire Messung erledigt (Abschnitt 6.2). Offen sind noch: BLAS-Angabe des CPU-Builds, eine längere Audiodatei, der isolierte Governor-Effekt auf den K80-Lauf und die Ursache, warum der Nicht-Encoder-Anteil auf der K80-Variante langsamer ist als auf der CPU.

## 8. Quellen

- NVIDIA CUDA 9.0 Release Notes, Abschnitt cuBLAS: GemmEx nicht auf Kepler – https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes
- cuBLAS 11.4.4 Dokumentation, `CUBLAS_STATUS_ARCH_MISMATCH` – https://docs.nvidia.com/cuda/archive/11.4.4/cublas/index.html
- ggml CUDA-Backend (llama.cpp master): `ggml-cuda.cu`, `mmq.cu`, `convert.cu`, `common.cuh` – https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- Pull Request mit Review-Diskussion: ggml-org/llama.cpp#30338 – https://github.com/ggml-org/llama.cpp/pull/30338
- llama.cpp Issue #12140 (K80 mit CUDA 11.4, Vulkan- vs. CUDA-Messung) – https://github.com/ggml-org/llama.cpp/issues/12140
