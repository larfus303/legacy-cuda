# Tesla K80 / Kepler: `CUBLAS_STATUS_ARCH_MISMATCH` in whisper.cpp und llama.cpp

**Betrifft:** Kepler-Karten (Compute Capability 3.x, z. B. Tesla K80 = sm_37, K40 = sm_35) mit ggml-basierten Programmen (whisper.cpp, llama.cpp und alles, was ggml einbettet). Fehler C (Abschnitt 2.4) betrifft vermutlich alle Karten unter CC 6.1, siehe dort.

**Status:** Zwei Patches. Zusammen bestehen sie auf einer Tesla K80 alle 2423 `MUL_MAT`-Tests von `test-backend-ops`. Auf einer RTX 3090 ändern sie nichts (2444/2444 mit und ohne Patch). whisper.cpp large-v3 läuft.

| Patch | Behebt | Datei |
|---|---|---|
| 0001 | Fehler A und B (`cublas…Ex` auf Kepler) | [`patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch`](../patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch) |
| 0002 | Fehler C (nicht-zusammenhängende quantisierte Gewichte ohne MMQ) | [`patches/ggml/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch`](../patches/ggml/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch) |

0002 setzt 0001 voraus. Beide nacheinander anwenden.

---

## 1. Symptom

Das Programm bricht während der Inferenz ab:

```
CUDA error: CUBLAS_STATUS_ARCH_MISMATCH
  current device: 0, in function ggml_cuda_mul_mat_cublas_impl at .../ggml-cuda.cu:<Zeile>
  cublasGemmStridedBatchedEx(...)        <- oder cublasGemmBatchedEx / cublasGemmEx
```

oder, nachdem nur Patch 0001 eingespielt ist:

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

**Warum das auf neueren Karten nicht auffällt:** Quantisierte Matrix-Multiplikationen laufen dort über ggml-eigene Kernel (MMQ) und erreichen den cuBLAS-Pfad gar nicht. MMQ braucht aber den Befehl `__dp4a`, den es erst ab **Compute Capability 6.1** gibt (`ggml_cuda_should_use_mmq()` gibt darunter sofort `false` zurück, auch mit `GGML_CUDA_FORCE_MMQ`). Unter CC 6.1 landen quantisierte Matrizen mit größerem Batch deshalb immer bei cuBLAS.

**Vermutlich ebenfalls betroffen (ungetestet):** Maxwell (M40, CC 5.2) und **P100 (CC 6.0)**. Die P40 (CC 6.1) hat MMQ und sollte nicht betroffen sein. Fehler A und B betreffen diese Karten nicht, weil die `…Ex`-Funktionen dort laufen.

### 2.5 Warum es nach „Modellgröße“ aussieht

Welcher Pfad gewählt wird (ggml-eigener Kernel oder cuBLAS, einzeln oder batched), hängt von Gewichtstyp, Tensorform und Batchgröße ab. Kleine oder quantisierte Modelle erreichen die kaputten Pfade zufällig nicht, große F16-Modelle schon.

**Diagnose:** In der Fehlermeldung steht unter `current device: …` die auslösende Anweisung:

| Steht dort … | Fehler | Behoben durch |
|---|---|---|
| `cublasGemmEx(` | A | 0001 |
| `cublasGemmStridedBatchedEx(` oder `cublasGemmBatchedEx(` | B | 0001 |
| `GGML_ASSERT(convert_func != nullptr)` | C | 0002 |

## 3. Was nicht hilft (geprüft)

| Ansatz | Warum nicht |
|---|---|
| `-DGGML_CUDA_FORCE_MMQ=ON` | `ggml_cuda_should_use_mmq()` gibt unter CC 6.1 `false` zurück, bevor das Flag geprüft wird. |
| `-DGGML_CUDA_FORCE_CUBLAS=ON` | Erzwingt cuBLAS erst recht. |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | Behebt höchstens Fehler A. Batched läuft trotzdem über die `…Ex`-Funktionen (Fehler B). |
| cuBLAS aus CUDA 10.2 nachladen | Die Kepler-Sperre für `…Ex` gilt seit CUDA 9.0. Dazu kommt der ABI-Bruch zwischen `libcublas.so.10` und `.so.11`. |
| Neueres CUDA | Ab Treiber 495 / CUDA 11.5+ gibt es keine Unterstützung mehr für sm_37. Treiber 470 + CUDA 11.4 ist das Ende. |
| Modell quantisieren | Umgeht A teilweise. B betrifft auch quantisierte Modelle, C sogar gerade diese. |

**Notlösung ohne Patch:** Vulkan-Backend (`-DGGML_VULKAN=1`). Treiber 470 bringt Vulkan für Kepler mit. Laut Community-Messungen (llama.cpp Issue #12140) ist es auf der K80 aber nur etwa halb so schnell wie CUDA (ca. 3 T/s gegenüber 5,5–6 T/s).

## 4. Die Fixes

### 4.1 Patch 0001: keine `…Ex`-Aufrufe auf Kepler (Fehler A und B)

Drei Änderungen in `ggml/src/ggml-cuda/ggml-cuda.cu`:

1. **Rechentyp auf Kepler immer F32.** In `ggml_cuda_mul_mat_cublas` wird bei NVIDIA mit `cc < 500` `compute_type = GGML_TYPE_F32` gesetzt. Das übersteuert BF16-Gewichte und die Umgebungsvariable `GGML_CUDA_CUBLAS_COMPUTE_TYPE`. BF16/F16 werden vorher nach F32 umgerechnet, die einzelne Matrix läuft dann über `cublasSgemm` (behebt A).
2. **Strided batched:** Bei F32 auf Kepler (`cc < 500`) wird `cublasSgemmStridedBatched` statt `cublasGemmStridedBatchedEx` aufgerufen.
3. **Pointer-batched:** Bei F32 auf Kepler (`cc < 500`) wird `cublasSgemmBatched` statt `cublasGemmBatchedEx` aufgerufen.

Alle drei Änderungen greifen **nur bei CC < 5.0**. Auf neueren Karten bleibt der Code-Pfad exakt wie vorher. (Eine erste Fassung hatte die Umschaltung in 2 und 3 für F32 auf allen Karten aktiv. Sie wurde auf Kepler beschränkt, damit neuere Karten nicht betroffen sind.)

Punkte 2 und 3 beheben B. Die Rechnung ist mathematisch identisch (gleiche Matrizen, gleiche Strides, F32-Akkumulation). Die Punkte 2 und 3 stehen in `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`, AMD- und Moore-Threads-Builds bleiben unberührt.

Für quantisierte Typen ändert Punkt 1 auf Kepler nichts. Dort war der Rechentyp schon vorher F32.

### 4.2 Patch 0002: Spezialfälle an die CPU abgeben (Fehler C)

+11 Zeilen in `ggml_backend_cuda_device_supports_op`, Abschnitt `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`. Die GPU meldet die Operation als „nicht unterstützt“, wenn **alle** Bedingungen gelten:

- die Gewichte sind quantisiert,
- sie liegen nicht zusammenhängend im Speicher (`!ggml_is_contiguously_allocated`),
- die Karte hat kein MMQ (`ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A`),
- es gibt keinen `convert_nc`-Umrechner für den Typ (`ggml_get_to_fp32_nc_cuda(type) == nullptr`).

ggml rechnet dann genau diese Operation auf der CPU, alles andere bleibt auf der GPU. Ebenfalls nur für CUDA (nicht HIP/MUSA).

**Alternative für später:** `convert_nc`-Umrechner für die K-Quants und IQ-Formate ergänzen. Dann bliebe auch dieser Fall auf der GPU, das ist aber deutlich mehr Code.

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
git apply /pfad/zu/legacy-cuda/patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch
git apply /pfad/zu/legacy-cuda/patches/ggml/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch

CC=gcc-10 CXX=g++-10 cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=g++-10 \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 5.3 whisper.cpp bauen

Gleich wie llama.cpp, nur im whisper.cpp-Repo. Der ggml-Code ist derselbe. Patch 0001 wurde auf whisper.cpp `d1be6fd` erstellt und mit large-v3 getestet.

```bash
./models/download-ggml-model.sh large-v3
./build/bin/whisper-cli -m models/ggml-large-v3.bin -f samples/jfk.wav
```

### 5.4 Neuere Versionen

Falls `git apply` auf einem neueren Stand fehlschlägt:

```bash
git apply -3 /pfad/zu/0001-...patch
```

Sonst die Stellen von Hand anpassen:

```bash
grep -n "cublasGemmEx\|cublasGemmBatchedEx\|cublasGemmStridedBatchedEx" ggml/src/ggml-cuda/ggml-cuda.cu
```

Jeder Treffer muss auf Kepler unerreichbar sein bzw. für F32 durch die `cublasSgemm*`-Entsprechung ersetzt werden.

### 5.5 Selbst nachmessen

Das Skript [`scripts/gegenmessung-k80.sh`](../scripts/gegenmessung-k80.sh) misst in zwei Teilen:

- **Teil 1, K80:** `test-backend-ops -o MUL_MAT` dreimal auf demselben Stand (ohne Patch, nur 0001, 0001+0002), Build für sm_37.
- **Teil 2, neuere Karte** (z. B. RTX 3090): `test-backend-ops -o MUL_MAT` ohne Patch und mit 0001+0002, eigener Build für die neue Architektur. Wird ein Modell angegeben, läuft zusätzlich `llama-bench` mit und ohne Patch, um zu zeigen, dass sich die Geschwindigkeit nicht ändert.

Am Ende setzt es den Quellcode wieder zurück. Das Ergebnis steht in `zusammenfassung.txt`.

```bash
./scripts/gegenmessung-k80.sh /pfad/zu/llama.cpp patches/ggml [/pfad/zu/modell.gguf]
```

Einstellungen über Umgebungsvariablen (Nummern wie in `nvidia-smi`, das Skript setzt `CUDA_DEVICE_ORDER=PCI_BUS_ID`):

| Variable | Standard | Bedeutung |
|---|---|---|
| `K80_GPU` | 1 | Nummer der K80 |
| `NEW_GPU` | 0 | Nummer der neueren Karte |
| `NEW_ARCH` | 86 | Architektur der neueren Karte (86 = RTX 30xx) |
| `SKIP_K80` / `SKIP_NEW` | – | auf `1` setzen, um einen Teil zu überspringen |

**Wichtig bei mehreren Karten im System:** Pro Lauf ist nur eine Karte sichtbar. Sind gleichzeitig neuere Karten sichtbar (z. B. eine RTX 3090), bricht der Test mit `CUDA kernel … has no device code compatible with CUDA arch 370` ab. Das ist ein eigenes Problem, siehe Abschnitt 7.

## 6. Testprotokoll

### 6.1 Gegenmessung `test-backend-ops -o MUL_MAT`

Stand llama.cpp `23b0202`, Treiber 470.256.02, CUDA 11.4, gcc 10, Skript `scripts/gegenmessung-k80.sh`. Gemessen am 2026-10-11 (Endfassung der Patches, 0001 mit Beschränkung auf `cc < 500`).

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

## 7. Offene Punkte

- **Mehrere GPUs gemischter Generation:** Mit RTX 3090 und K80 gleichzeitig sichtbar kommt `CUDA kernel mul_mat_vec_f has no device code compatible with CUDA arch 370. ggml-cuda.cu was compiled for: __CUDA_ARCH_LIST__`. Dass dort wörtlich `__CUDA_ARCH_LIST__` steht statt einer Architekturliste, deutet darauf hin, dass CUDA 11.4 dieses Makro noch nicht kennt (es kam mit CUDA 11.5). Noch nicht untersucht.
- **Fehler C auf M40 und P100:** vermutlich ebenfalls betroffen, ungetestet.
- **whisper K80 gegen CPU, Restpunkte:** faire Messung erledigt (Abschnitt 6.2). Offen sind noch: BLAS-Angabe des CPU-Builds, eine längere Audiodatei, der isolierte Governor-Effekt auf den K80-Lauf und die Ursache, warum der Nicht-Encoder-Anteil auf der K80-Variante langsamer ist als auf der CPU.
- **Upstream:** eingereicht als Pull Request [ggml-org/llama.cpp#30338](https://github.com/ggml-org/llama.cpp/pull/30338) (2026-10-11).

## 8. Quellen

- NVIDIA CUDA 9.0 Release Notes, Abschnitt cuBLAS: GemmEx nicht auf Kepler – https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes
- cuBLAS 11.4.4 Dokumentation, `CUBLAS_STATUS_ARCH_MISMATCH` – https://docs.nvidia.com/cuda/archive/11.4.4/cublas/index.html
- ggml CUDA-Backend (llama.cpp master): `ggml-cuda.cu`, `mmq.cu`, `convert.cu`, `common.cuh` – https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- llama.cpp Issue #12140 (K80 mit CUDA 11.4, Vulkan- vs. CUDA-Messung) – https://github.com/ggml-org/llama.cpp/issues/12140
