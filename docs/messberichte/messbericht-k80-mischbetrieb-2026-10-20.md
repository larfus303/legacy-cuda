# Messbericht: K80 / RTX 3090 — Mischbetrieb und LLM-Vergleich

**Datum:** 2026-10-20
**Durchgeführt von:** Max (OpenClaw), auf Server „bodo"
**Code-Stand:** llama.cpp Commit `6b3b84800` („cuda: only use MMQ for non-contiguous src0"), Mischbuild `-DCMAKE_CUDA_ARCHITECTURES="37;86"`
**Zweck:** Diese Messungen gehören zur Doku des Repos `larfus303/legacy-cuda` (Kepler-Fix für alte NVIDIA-Karten). Sie sollen drei Fragen beantworten.

---

## 0. Die drei Fragen

1. **Läuft der Mischbuild (Architekturen 37 + 86 in einem Binary) korrekt auf beiden Architekturen gleichzeitig?** — Das war der ursprüngliche Fehler `no device code compatible with CUDA arch 370`.
2. **Rechnet er im Mischbetrieb auch *richtig* (nicht nur schnell)?**
3. **Wie schlägt sich die K80 bei LLM-Inferenz im Vergleich zu CPU und RTX 3090?**

---

## 1. Messaufbau

**Hardware (Server „bodo"):**

| Gerät | CUDA-Index | Architektur (CC) | VRAM | Leistungsgrenze | Bemerkung |
|---|---|---|---|---|---|
| RTX 3090 | 0 | 8.6 | 24 266 MiB | 280 W | |
| Tesla K80, Hälfte „K80-A" | 1 | 3.7 | 11 441 MiB | 110 W | bewusst gedrosselt (heißere Hälfte) |
| Tesla K80, Hälfte „K80-B" | 2 | 3.7 | 11 441 MiB | 175 W | volle Leistung (kühlere Hälfte) |

CPU: Xeon E5-2699 v4 (Broadwell), 22 Kerne / 44 Threads, AVX2, 256 GB RAM. Treiber 470.256.02, CUDA 11.4.

**Build:** `build-mix`, `CMAKE_CUDA_ARCHITECTURES="37;86"`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`. Ein einziges Binary enthält Kernel für beide Architekturen.

**Flags (GPU-Läufe):** `-ngl 99 -fa 1 -ub 512 -r 3` (außer 70B: `-r 1`).
**Flags (CPU-Läufe):** `CUDA_VISIBLE_DEVICES=` (leer, keine GPU sichtbar), `-ngl 0 -t 22 -r 3` (70B: `-r 1`).

> **Wichtig bei CPU-Messungen:** Ein llama.cpp-Build mit CUDA-Backend verlagert bei `-ngl 0` große Batches (ab 32 Tokens) automatisch auf eine sichtbare GPU. Ohne leeres `CUDA_VISIBLE_DEVICES` ist der gemessene Prefill dann ein GPU-Wert, kein CPU-Wert. In einem ersten, fehlerhaften CPU-Lauf ergaben sich so 836 t/s (7B) bzw. 96,6 t/s (70B) Prefill — tatsächlich auf der 3090 gerechnet. Alle CPU-Werte in diesem Bericht sind mit leerem `CUDA_VISIBLE_DEVICES` gemessen; im Log steht dann `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`.

**Modelle:**
- Mistral-7B-Instruct-v0.3 Q4_K_M — 4,07 GiB (für die direkten Vergleiche)
- Dolphin-2.9.1-Llama-3-70B Q4_K_M — 39,59 GiB (für den „zu groß für eine Karte"-Test)

---

## 2. Fix-Verifikation: Mischbinary auf den K80-Hälften

**Ausgangspunkt:** Der `build-mix` war zunächst auf dem *alten* Stand `e0641ca9b` gebaut (vor dem Fix). Ein erster Lauf auf K80-A zeigte einen niedrigen Prefill:

| Stand | pp512 | tg128 |
|---|---|---|
| `e0641ca9b` (ohne Fix) | 38,95 ± 0,20 t/s | 8,86 ± 0,02 t/s |

`pp512 = 38,95 t/s` liegt in der Größenordnung der Prefill-Regression, die der Fix behebt. **Hinweis zur Vergleichbarkeit:** Dieser Wert wurde auf K80-A (110 W, gedrosselt) gemessen, während die frühere Regressionsmessung (42,8 t/s) auf K80-B (175 W) lief. Beides passt zusammen, die Hälften sind aber wegen der unterschiedlichen Leistungsgrenze nicht direkt vergleichbar. Der `build-mix` musste mit dem Fix neu gebaut werden.

**Nach Rebuild auf `6b3b84800`:**

| Gerät | pp512 | tg128 |
|---|---|---|
| K80-A (110 W, gedrosselt) | 101,93 ± 0,77 t/s | 8,96 ± 0,01 t/s |
| K80-B (175 W) | 140,96 ± 0,14 t/s | 12,17 ± 0,08 t/s |

**Erklärung:** Der Prefill springt mit dem Fix von 38,95 auf 101,93 (K80-A) bzw. 140,96 t/s (K80-B). Der Fix wirkt also auch im Mischbinary. K80-B liegt auf dem Niveau der früheren Fix-Verifikation (140,16 t/s; Differenz 0,6 %). Der Unterschied zwischen den Hälften ist die Leistungsgrenze (110 vs. 175 W), nicht der Patch.

**Befund 1:** ✅ Das Mischbinary rechnet auf beiden K80-Hälften, und der Fix ist darin wirksam.

---

## 3. Mischbetrieb: 3090 + K80 gleichzeitig

**Test:** `CUDA_VISIBLE_DEVICES=0,2`, Layer-Split (`-sm layer`) über 3090 und K80-B, Mistral-7B.

```
ggml_cuda_init: found 2 CUDA devices (Total VRAM: 35707 MiB):
  Device 0: RTX 3090, CC 8.6, VRAM 24266 MiB
  Device 1: K80, CC 3.7, VRAM 11441 MiB
```

| Test | Wert |
|---|---|
| pp512 | 490,00 ± 2,48 t/s |
| tg128 | 35,17 ± 0,05 t/s |

**Erklärung:** Beide Architekturen (8.6 + 3.7) laufen gleichzeitig, **ohne** den `arch 370`-Fehler. Der Mischbuild wählt die Kernel-Images pro GPU korrekt aus. Der Durchsatz (35,17 t/s) liegt zwischen „nur 3090" (135 t/s) und „nur K80-B" (12,17 t/s): Beim Layer-Split über zwei so unterschiedlich schnelle Karten wird die langsamere (K80-B) zum Flaschenhals, die 3090 wartet auf sie.

**Befund 2:** ✅ Der ursprüngliche Fehler (`no device code compatible with CUDA arch 370`) ist behoben. Der Mischbetrieb läuft, ist aber für reine Durchsatz-Aufgaben **langsamer als die 3090 allein**.

---

## 4. Korrektheit: perplexity im Mischbetrieb

**Test:** `llama-perplexity`, WikiText-2 Test-Set (4358 Zeilen), `-c 512`, `-sm layer`.

| Lauf | PPL |
|---|---|
| Referenz, nur 3090 | 6,2250 ± 0,03760 |
| Mischbetrieb (0,2) | 6,2249 ± 0,03760 |

**Erklärung:** Die Werte stimmen bis auf die vierte Nachkommastelle überein; die Differenz von 0,0001 liegt weit innerhalb der Messunsicherheit von ±0,038. Der Mischbetrieb rechnet also **praktisch identisch** zum Single-GPU-Referenzlauf.

**Befund 3:** ✅ Der Mischbetrieb rechnet korrekt.

---

## 5. Vollständiger Vergleich (Mistral-7B Q4_K_M)

Alle Werte gleicher Build, gleiche Flags, gleicher Tag:

| Konfiguration | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| RTX 3090 allein | 4466,67 ± 312,75 | **135,01 ± 0,34** |
| Mischbetrieb 3090 + K80-B | 490,00 ± 2,48 | 35,17 ± 0,05 |
| K80-B (175 W) allein | 140,96 ± 0,14 | 12,17 ± 0,08 |
| K80-A (110 W) allein | 101,93 ± 0,77 | 8,96 ± 0,01 |
| CPU (Xeon, 22 Kerne) | 80,05 ± 0,42 | **14,40 ± 0,06** |

**Erklärung — der differenzierte Befund:**

- **Prefill:** Die K80 gewinnt. K80-B (140,96 t/s) ist rund 1,76× schneller als die CPU (80,05 t/s); auch die gedrosselte K80-A (101,93 t/s) schlägt die CPU noch.
- **Generation:** Die CPU dreht das Bild. Sie liegt mit 14,40 t/s knapp vor K80-B (12,17 t/s) und deutlich vor K80-A (8,96 t/s).
- Die Drosselung ist **nicht** der Grund für das Generation-Ergebnis: auch die volle 175-W-Hälfte wird von der CPU knapp überholt.
- Der einzige klare Beschleuniger in beiden Metriken ist die RTX 3090 (135 t/s, etwa das Zehnfache von CPU und K80 bei der Generation).

**Einordnung:** Das deckt sich mit dem whisper-Vergleich (Abschnitt 6.2 der Doku), wo die K80 ~1,7× schneller war als dieselbe CPU — auch hier gewinnt die K80 beim batched Prefill. Der Unterschied liegt am Workload: der Prefill (große Matmuls) nutzt die K80 aus, die tokenweise Generation (memory-bound, ein Token pro Schritt) nicht.

---

## 6. Großes Modell: 70B über alle drei GPUs

**Motivation:** Der eigentliche Anwendungsfall des Mischbetriebs ist ein Modell, das auf keine einzelne Karte passt. Dolphin-70B Q4_K_M (39,59 GiB) passt nicht in die 24 GiB der 3090, aber in die 46 GiB Gesamt-VRAM (3090 + 2× K80).

**Test:** `CUDA_VISIBLE_DEVICES=0,1,2`, `-sm layer`, `-r 1`.

| Konfiguration | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| 3-Wege-Split (3090 + 2× K80) | **19,22** | **2,45** |
| CPU (22 Kerne, RAM) | 8,27 | 1,58 |

**Erklärung:** Das Modell läuft über alle drei GPUs — ohne die Kombination wäre es schlicht nicht lauffähig (39,59 GiB > 24 GiB der 3090). Der 3-Wege-Split ist dabei **in beiden Metriken schneller als die CPU**:

- Prefill: 2,3× schneller (19,22 vs. 8,27 t/s)
- Generation: 1,55× schneller (2,45 vs. 1,58 t/s)

**Befund 4:** Für ein Modell, das auf keine einzelne Karte passt, ist der 3-Wege-Split die einzige GPU-Option — und er schlägt die CPU klar in beiden Metriken. Das ist der eigentliche, greifbare Wert des Mischbetriebs.

---

## 7. Zusammenfassung der Kernaussagen

1. **Fix + Mischbinary korrekt:** `6b3b84800` wirkt im `37;86`-Build, beide Architekturen laufen gleichzeitig, perplexity praktisch identisch (6,2249 vs. 6,2250).
2. **K80 vs. CPU (7B):** Differenziert — die K80 ist beim Prefill 1,76× schneller als die CPU, bei der tokenweisen Generation aber knapp langsamer. Nur die 3090 ist in beiden Metriken klar vorn.
3. **70B über 3 GPUs:** Lauffähig (das ist der eigentliche Wert) und in beiden Metriken schneller als die CPU (Prefill 2,3×, Generation 1,55×).
4. **Praktischer Betrieb:** Für kleine Modelle ist die sinnvollste Nutzung *getrennt* (3090 trägt das Haupt-LLM, K80 übernimmt Nebenlast) — Layer-Split über beide bremst die 3090 nur aus. Für Modelle, die auf keine einzelne Karte passen, ist der 3-Wege-Split dagegen die richtige Wahl.

---

## Anhang: Roh-Outputs

Alle Zahlen stammen aus `llama-bench` / `llama-perplexity` des `build-mix` (`6b3b84800`). Die ±-Werte sind die von llama-bench gemeldeten Standardabweichungen über `-r 3` (bzw. `-r 1` bei 70B). Der perplexity-Lauf brauchte im Mischbetrieb ~37 min (K80-Flaschenhals) gegenüber ~1,5 min auf 3090-only. Die CPU-Läufe liefen mit leerem `CUDA_VISIBLE_DEVICES` (kein CUDA-Gerät sichtbar, siehe Hinweis in Abschnitt 1).
