# Tesla K80 / Kepler: `CUBLAS_STATUS_ARCH_MISMATCH` in whisper.cpp und llama.cpp

**Betrifft:** alle Kepler-Karten (Compute Capability 3.x, z. B. Tesla K80 = sm_37, K40 = sm_35) mit ggml-basierten Programmen (whisper.cpp, llama.cpp und alles, was ggml einbettet).
**Status:** Patch vorhanden und auf einer Tesla K80 erfolgreich getestet (whisper.cpp, Modell large-v3).
**Patch:** [`patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch`](../patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch)

---

## 1. Symptom

Das Programm bricht während der Inferenz mit dieser Meldung ab:

```
CUDA error: CUBLAS_STATUS_ARCH_MISMATCH
  current device: 0, in function ggml_cuda_mul_mat_cublas_impl at .../ggml-cuda.cu:<Zeile>
  cublasGemmStridedBatchedEx(...)        <- oder cublasGemmBatchedEx / cublasGemmEx
ggml-cuda.cu:<Zeile>: CUDA error
```

Typisches Muster:

- Kleine Modelle (whisper tiny, oft auch medium) und quantisierte Modelle (Q4_K_M, Q8_0) laufen.
- Größere Modelle oder F16-Modelle (whisper large-v3, F16-GGUFs in llama.cpp) stürzen ab.

Das sieht zunächst nach einem Größenproblem aus, ist aber keins (siehe Abschnitt 2).

## 2. Ursache

### 2.1 cuBLAS: Die `…Ex`-Funktionen gibt es auf Kepler nicht

Seit CUDA 9.0 steht in den Release Notes von NVIDIA, dass die GemmEx-Routinen (Mixed-Precision-Erweiterungen von `cublas<t>gemm`) nur ab Maxwell unterstützt werden und **nicht auf Kepler, ausdrücklich genannt sind Tesla K40 und K80**. Das betrifft:

- `cublasGemmEx`
- `cublasGemmBatchedEx`
- `cublasGemmStridedBatchedEx`

Diese Funktionen liefern auf CC < 5.0 `CUBLAS_STATUS_ARCH_MISMATCH`, und zwar **auch dann, wenn alle Datentypen FP32 sind**.

Die klassischen, typisierten Funktionen wie `cublasSgemm`, `cublasSgemmBatched` und `cublasSgemmStridedBatched` laufen auf Kepler dagegen problemlos.

### 2.2 ggml: Batched-Matmuls nutzen immer die `…Ex`-Variante

In `ggml/src/ggml-cuda/ggml-cuda.cu`, Funktion `ggml_cuda_mul_mat_cublas_impl`, wählt ggml so aus:

| Fall | Aufruf | Kepler |
|---|---|---|
| F32, eine einzelne Matrix (`ne12==1 && ne13==1`) | `cublasSgemm` | ✅ läuft |
| F16/BF16, eine einzelne Matrix | `cublasGemmEx` | ❌ ARCH_MISMATCH |
| beliebiger Typ, batched ohne Broadcast | `cublasGemmStridedBatchedEx` | ❌ ARCH_MISMATCH |
| beliebiger Typ, batched mit Broadcast | `cublasGemmBatchedEx` | ❌ ARCH_MISMATCH |

Auf Kepler setzt ggml F16 zwar richtig auf F32 herunter, weil `fast_fp16_hardware_available()` dort false liefert. Das rettet aber nur den Einzelmatrix-Fall. **Batched** Matmuls (vor allem die Attention über mehrere Köpfe) landen weiter in einem `…Ex`-Aufruf.

### 2.3 Warum es nach „Modellgröße“ aussieht

Ob ein Matmul über cuBLAS-batched läuft oder über einen ggml-eigenen Kernel (mmvf/mmvq), hängt von Tensorform, Batchgröße und Gewichtstyp ab, nicht von der Dateigröße. Kleine oder quantisierte Modelle erreichen den kaputten Pfad zufällig nicht, große F16-Modelle schon.

**Diagnose:** In der Fehlermeldung steht unter `current device: …` die auslösende Anweisung. Steht dort ein `cublas…Ex(`, ist es genau dieses Problem.

## 3. Was nicht hilft (geprüft)

| Ansatz | Warum nicht |
|---|---|
| `-DGGML_CUDA_FORCE_MMQ=ON` | `ggml_cuda_should_use_mmq()` gibt auf Karten unter CC 6.1 (dp4a) `false` zurück, bevor das Flag überhaupt geprüft wird. MMQ greift auf Kepler nie, und es betrifft ohnehin nur quantisierte Gewichte, nicht die Attention. |
| `-DGGML_CUDA_FORCE_CUBLAS=ON` | Erzwingt cuBLAS erst recht, macht es also schlimmer. |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | Stellt nur den Datentyp um. Batched läuft trotzdem über `cublasGemmStridedBatchedEx` / `cublasGemmBatchedEx`. |
| cuBLAS aus CUDA 10.2 nachladen | Die Kepler-Sperre für `…Ex` gilt seit CUDA 9.0. Dazu kommt der ABI-Bruch zwischen `libcublas.so.10` und `.so.11`. |
| Neueres CUDA | Ab Treiber 495 / CUDA 11.5+ wird sm_37 nicht mehr unterstützt. Treiber 470 + CUDA 11.4 ist das Ende. |
| Modell quantisieren | Hilft teilweise bei llama.cpp, die Attention-Matmuls bleiben aber batched. Kein verlässlicher Fix. |

**Notlösung ohne Patch:** Vulkan-Backend (`-DGGML_VULKAN=1`). Treiber 470 bringt Vulkan für Kepler mit. Laut Messungen aus der Community (llama.cpp Issue #12140) ist Vulkan auf der K80 aber nur etwa halb so schnell wie CUDA (ca. 3 T/s gegenüber 5,5–6 T/s).

## 4. Der Fix

Der Patch ändert `ggml/src/ggml-cuda/ggml-cuda.cu` an drei Stellen:

1. **Rechenart auf Kepler immer F32.** In `ggml_cuda_mul_mat_cublas` wird bei NVIDIA mit `cc < 500` `compute_type = GGML_TYPE_F32` gesetzt. Das übersteuert auch BF16-Gewichte und die Umgebungsvariable `GGML_CUDA_CUBLAS_COMPUTE_TYPE`.
2. **Strided batched:** Für F32 wird `cublasSgemmStridedBatched` statt `cublasGemmStridedBatchedEx` verwendet.
3. **Pointer-batched:** Für F32 wird `cublasSgemmBatched` statt `cublasGemmBatchedEx` verwendet.

Damit gibt es auf Kepler keinen einzigen `…Ex`-Aufruf mehr. Die Rechnung ist mathematisch identisch (gleiche Matrizen, gleiche Strides, F32-Akkumulation). Die Änderungen 2 und 3 stehen in `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`, AMD- und Moore-Threads-Builds bleiben also unberührt.

## 5. Anleitung (whisper.cpp)

### 5.1 Voraussetzungen

| Komponente | Version |
|---|---|
| Treiber | NVIDIA 470-Serie (z. B. 470.256.02), die letzte mit Kepler-Unterstützung |
| CUDA Toolkit | 11.4 (Toolkit only, Treiber nicht überschreiben) |
| Compiler | gcc/g++ 10 (CUDA 11.4 akzeptiert keinen neueren Host-Compiler) |
| CMake | ≥ 3.18 |

### 5.2 Bauen

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
git checkout d1be6fd                      # Stand, gegen den der Patch erstellt wurde
git apply /pfad/zu/0001-cuda-kepler-avoid-cublas-ex.patch

CC=gcc-10 CXX=g++-10 cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 5.3 Testen

```bash
./models/download-ggml-model.sh large-v3
./build/bin/whisper-cli -m models/ggml-large-v3.bin -f samples/jfk.wav
```

Erwartet: Transkription ohne `CUBLAS_STATUS_ARCH_MISMATCH`.

### 5.4 Neuere Versionen von whisper.cpp / llama.cpp

Der ggml-Code ist in whisper.cpp und llama.cpp derselbe. Falls `git apply` auf einem neueren Stand fehlschlägt:

```bash
git apply -3 /pfad/zu/0001-cuda-kepler-avoid-cublas-ex.patch
```

Sonst die Stellen von Hand anpassen:

```bash
grep -n "cublasGemmEx\|cublasGemmBatchedEx\|cublasGemmStridedBatchedEx" ggml/src/ggml-cuda/ggml-cuda.cu
```

Jeder Treffer muss auf Kepler unerreichbar sein bzw. für F32 durch die `cublasSgemm*`-Entsprechung ersetzt werden.

## 6. Testprotokoll

| Datum | Karte | Treiber / CUDA | Software | Modell | Ergebnis |
|---|---|---|---|---|---|
| 2026-10-10 | Tesla K80 | 470 / 11.4 | whisper.cpp d1be6fd + Patch | large-v3 | ✅ läuft |

Weitere Messwerte (Laufzeit, Vergleich zu Vulkan/CPU) bitte hier ergänzen.

## 7. Quellen

- NVIDIA CUDA 9.0 Release Notes, Abschnitt cuBLAS: GemmEx nicht auf Kepler – https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes
- cuBLAS 11.4.4 Dokumentation, `CUBLAS_STATUS_ARCH_MISMATCH` – https://docs.nvidia.com/cuda/archive/11.4.4/cublas/index.html
- ggml CUDA-Backend (llama.cpp master): `ggml-cuda.cu`, `mmq.cu`, `common.cuh` – https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda
- llama.cpp Issue #12140 (K80 mit CUDA 11.4, Vulkan- vs. CUDA-Messung) – https://github.com/ggml-org/llama.cpp/issues/12140
