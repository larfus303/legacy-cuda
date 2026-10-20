# legacy-cuda

KI auf alten NVIDIA-Rechenkarten (K80, M40, P40, P100, V100) – Treiber, Patches, Anleitungen.

## Worum geht es?

Ältere NVIDIA-Rechenkarten wie die Tesla K80 sind gebraucht sehr günstig zu haben und haben viel VRAM. Für lokale KI-Workloads (Spracherkennung, LLMs, Sprachsynthese) taugen sie noch, aber der Einstieg ist mühsam. Der Treibersupport endet mit der 470er-Serie, aktuelle Software setzt neuere Architekturen voraus, und viele Fehler sind nirgends dokumentiert.

Dieses Projekt sammelt, was nötig ist, damit diese Karten trotzdem laufen:

- funktionierende Treiber- und Toolkit-Kombinationen
- Patches für Software, die auf den alten Architekturen abstürzt
- Schritt-für-Schritt-Anleitungen, auch für Einsteiger

Das Ziel: Weniger Karten landen als Elektroschrott, weil sie sich einrichten lassen.

## Unterstützte Karten (Zielumfang)

| Karte | Architektur | Compute Capability | Status in diesem Repo |
|---|---|---|---|
| Tesla K80 | Kepler (GK210) | 3.7 | ggml-Patch v2 getestet: alle 18 477 Tests von `test-backend-ops` bestanden, wie auf der RTX 3090 |
| Tesla M40 | Maxwell (GM200) | 5.2 | geplant |
| Tesla P40 | Pascal (GP102) | 6.1 | geplant |
| Tesla P100 | Pascal (GP100) | 6.0 | geplant |
| Tesla V100 | Volta (GV100) | 7.0 | geplant |

## Inhalt

### Probleme und Lösungen

| Thema | Betrifft | Dokument | Patch |
|---|---|---|---|
| `CUBLAS_STATUS_ARCH_MISMATCH` in whisper.cpp / llama.cpp | Kepler (K80, K40) | [docs/kepler-cublas-arch-mismatch.md](docs/kepler-cublas-arch-mismatch.md) | [v2](patches/ggml/cuda-kepler-mul-mat-v2.patch) |
| `GGML_ASSERT(convert_func != nullptr)` bei quantisierten Modellen | Karten unter CC 6.1 (K80 getestet; M40, P100 vermutlich) | [docs/kepler-cublas-arch-mismatch.md](docs/kepler-cublas-arch-mismatch.md) | [v2](patches/ggml/cuda-kepler-mul-mat-v2.patch) |
| K80 und neuere Karte in einem Programm (`no kernel image` / `no device code compatible with CUDA arch 370`) | Mischbetrieb, z. B. K80 + RTX 3090 | [docs/kepler-cublas-arch-mismatch.md](docs/kepler-cublas-arch-mismatch.md), Abschnitt 6.6, und [Messbericht](docs/messberichte/messbericht-k80-mischbetrieb-2026-10-20.md) | Build-Einstellung `-DCMAKE_CUDA_ARCHITECTURES="37;86"` |

Der Patch v2 entspricht dem Stand des Pull Requests nach der ersten Review-Runde. Die älteren Patches 0001/0002 (Fassung 1) liegen in [`patches/ggml/v1/`](patches/ggml/v1/).

**Betriebstipp K80:** In llama.cpp `-fa 1` (Flash Attention) setzen, das macht die Generation rund 4 % schneller. `-ub` nicht unter 512 setzen. Messwerte im Dokument, Abschnitt 6.5. Vergleich K80 gegen CPU und 3090 in Abschnitt 6.7.

### Verzeichnisstruktur

```
docs/       Anleitungen und Fehleranalysen, ein Dokument pro Thema
patches/    Patches, nach Zielprojekt sortiert (z. B. patches/ggml/)
scripts/    Mess- und Hilfsskripte
```

## Todo

- [x] Kepler-Fix für `CUBLAS_STATUS_ARCH_MISMATCH` (ggml) mit Doku
- [x] Fix für fehlende Umrechner bei quantisierten Modellen ohne MMQ (ggml)
- [x] Gegenmessung mit `test-backend-ops` (ohne Patch / 0001 / 0001+0002)
- [x] Fix als Pull Request an ggml-org/llama.cpp eingereicht: [#30338](https://github.com/ggml-org/llama.cpp/pull/30338)
- [x] Review-Runde 1 umgesetzt (Patch v2), K80 und 3090 je 2444/2444 MUL_MAT, Prefill wie vorher
- [x] Alle Operationen auf der K80 geprüft: 18 477/18 477, keine Kepler-spezifischen Lücken
- [ ] Antwort des Reviewers auf v2 abwarten
- [ ] Voller `test-backend-ops`-Lauf mit dem Endstand `6b3b848`
- [ ] whisper mit Patch v2 messen
- [ ] Treiberbank: getestete Treiber-/CUDA-/Compiler-Kombinationen je Karte
- [ ] Installationshilfe für Einsteiger (vom leeren System bis zur ersten Inferenz)
- [ ] Benchmark-Doku: eigene Messwerte, u. a. Tesla K80 gegen RTX 3090
- [ ] Konzeptdatei: Ziele, Umfang und Aufbau des Projekts
- [ ] Weitere Karten (M40, P40, P100, V100) aufnehmen
- [ ] Fehler C auf M40 / P100 prüfen (vermutlich betroffen)
- [x] Mischbetrieb K80 + RTX 3090: Build `37;86` läuft auf beiden Karten gemeinsam und rechnet korrekt (Perplexity gleich)
- [x] LLM-Vergleich K80 / CPU / 3090 gemessen (Mistral-7B, Dolphin-70B über drei GPUs)

## Mitmachen

Erfahrungen mit diesen Karten sind willkommen, auch Fehlschläge. Am hilfreichsten sind Issues mit:

- Karte, Treiberversion, CUDA-Version, Betriebssystem
- Software und Version bzw. Commit
- vollständige Fehlermeldung

## Lizenz

MIT, siehe [LICENSE](LICENSE). Patches für Fremdprojekte stehen zusätzlich unter der Lizenz des jeweiligen Projekts.
