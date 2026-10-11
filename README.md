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
| Tesla K80 | Kepler (GK210) | 3.7 | ggml-Patches getestet (2423/2423 MUL_MAT-Tests) |
| Tesla M40 | Maxwell (GM200) | 5.2 | geplant |
| Tesla P40 | Pascal (GP102) | 6.1 | geplant |
| Tesla P100 | Pascal (GP100) | 6.0 | geplant |
| Tesla V100 | Volta (GV100) | 7.0 | geplant |

## Inhalt

### Probleme und Lösungen

| Thema | Betrifft | Dokument | Patch |
|---|---|---|---|
| `CUBLAS_STATUS_ARCH_MISMATCH` in whisper.cpp / llama.cpp | Kepler (K80, K40) | [docs/kepler-cublas-arch-mismatch.md](docs/kepler-cublas-arch-mismatch.md) | [0001](patches/ggml/0001-cuda-kepler-avoid-cublas-ex.patch) |
| `GGML_ASSERT(convert_func != nullptr)` bei quantisierten Modellen | Karten unter CC 6.1 (K80 getestet; M40, P100 vermutlich) | [docs/kepler-cublas-arch-mismatch.md](docs/kepler-cublas-arch-mismatch.md) | [0002](patches/ggml/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch) |

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
- [ ] Fix als Pull Request an ggml-org/llama.cpp einreichen
- [ ] Treiberbank: getestete Treiber-/CUDA-/Compiler-Kombinationen je Karte
- [ ] Installationshilfe für Einsteiger (vom leeren System bis zur ersten Inferenz)
- [ ] Benchmark-Doku: eigene Messwerte, u. a. Tesla K80 gegen RTX 3090
- [ ] Konzeptdatei: Ziele, Umfang und Aufbau des Projekts
- [ ] Weitere Karten (M40, P40, P100, V100) aufnehmen
- [ ] Fehler C auf M40 / P100 prüfen (vermutlich betroffen)
- [ ] Mischbetrieb K80 + neuere Karte: Fehler `no device code compatible with CUDA arch 370` untersuchen

## Mitmachen

Erfahrungen mit diesen Karten sind willkommen, auch Fehlschläge. Am hilfreichsten sind Issues mit:

- Karte, Treiberversion, CUDA-Version, Betriebssystem
- Software und Version bzw. Commit
- vollständige Fehlermeldung

## Lizenz

MIT, siehe [LICENSE](LICENSE). Patches für Fremdprojekte stehen zusätzlich unter der Lizenz des jeweiligen Projekts.
