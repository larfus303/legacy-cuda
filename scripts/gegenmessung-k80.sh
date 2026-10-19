#!/usr/bin/env bash
# Gegenmessung des ggml-Kepler-Patches (v2, PR #30338)
#
# Aufruf:
#   ./gegenmessung-k80.sh <llama.cpp-Verzeichnis> <Patch-Datei> [Modell.gguf]
#
#   llama.cpp-Verzeichnis: sauberer Stand, auf den der Patch passt (getestet: 23b0202)
#   Patch-Datei:           z. B. patches/ggml/cuda-kepler-mul-mat-v2.patch
#   Modell.gguf:           optional, dann zusaetzlich llama-bench mit/ohne Patch
#                          (fuer die K80 ein K-Quant-Modell nehmen, z. B. Q4_K_M)
#
# Teil 1 - Tesla K80 (eine GPU-Haelfte):  ohne Patch / mit Patch
# Teil 2 - neuere Karte, z. B. RTX 3090:  ohne Patch / mit Patch
# Jeweils test-backend-ops -o MUL_MAT, optional llama-bench.
#
# Umgebungsvariablen (Nummern wie in nvidia-smi):
#   K80_GPU=2   NEW_GPU=0   NEW_ARCH=86   SKIP_K80=1   SKIP_NEW=1
# K80_GPU: die kuehlere, nicht gedrosselte GPU-Haelfte nehmen
#   (nvidia-smi --query-gpu=index,name,temperature.gpu,power.limit --format=csv)
set -u
SRC=$(realpath "$1"); PATCH=$(realpath "$2"); MODEL=${3:-}
[ -n "$MODEL" ] && MODEL=$(realpath "$MODEL")
K80_GPU=${K80_GPU:-2}; NEW_GPU=${NEW_GPU:-0}; NEW_ARCH=${NEW_ARCH:-86}
LOG="$PWD/gegenmessung-$(date +%Y%m%d-%H%M)"; mkdir -p "$LOG"
SUM="$LOG/zusammenfassung.txt"

# Geraetenummern wie in nvidia-smi (CUDA sortiert sonst nach Geschwindigkeit)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$SRC"
D=ggml/src/ggml-cuda
git checkout -- "$D"
echo "llama.cpp $(git rev-parse --short HEAD), Patch $(basename "$PATCH"), $(date '+%F %T')" | tee "$SUM"
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,power.limit --format=csv,noheader >> "$SUM"

configure() {  # $1 = Build-Ordner, $2 = CUDA-Architektur
  [ -d "$1" ] || CC=gcc-10 CXX=g++-10 cmake -B "$1" -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=g++-10 \
    -DCMAKE_CUDA_ARCHITECTURES="$2" -DCMAKE_BUILD_TYPE=Release > "$LOG/cmake-$1.log" 2>&1
}

set_patch() {  # $1 = none | v2
  git checkout -- "$D"
  [ "$1" = v2 ] && git apply "$PATCH"
}

test_ops() {  # $1 = Build-Ordner, $2 = GPU, $3 = Name
  local name="$3"
  cmake --build "$1" -j --target test-backend-ops > "$LOG/build-$name.log" 2>&1
  local brc=$?
  echo "BUILD_EXIT=$brc" >> "$LOG/build-$name.log"
  if [ "$brc" -ne 0 ] || [ ! -x "./$1/bin/test-backend-ops" ]; then
    printf "%-18s BUILD FEHLGESCHLAGEN (exit=%s), siehe %s\n" "$name" "$brc" "$LOG/build-$name.log" | tee -a "$SUM"
    return 1
  fi
  CUDA_VISIBLE_DEVICES=$2 "./$1/bin/test-backend-ops" -o MUL_MAT > "$LOG/test-$name.log" 2>&1
  local rc=$?
  # ANSI-Farbcodes entfernen, sonst zaehlt grep die OK-Zeilen nicht
  sed 's/\x1b\[[0-9;]*m//g' "$LOG/test-$name.log" > "$LOG/test-$name.plain.log"
  local ok=$(grep -c "OK\s*$" "$LOG/test-$name.plain.log")
  local fail=$(grep -c "FAIL" "$LOG/test-$name.plain.log")
  printf "%-18s exit=%-4s OK-Zeilen=%-5s FAIL=%-4s %s\n" "$name" "$rc" "$ok" "$fail" \
    "$(grep -E 'tests passed|GGML_ASSERT|CUBLAS_STATUS' "$LOG/test-$name.plain.log" | tail -1)" | tee -a "$SUM"
}

bench() {  # $1 = Build-Ordner, $2 = GPU, $3 = Name
  [ -n "$MODEL" ] || return 0
  if ! cmake --build "$1" -j --target llama-bench > "$LOG/build-bench-$3.log" 2>&1; then
    echo "llama-bench $3: BUILD FEHLGESCHLAGEN" | tee -a "$SUM"; return 1
  fi
  CUDA_VISIBLE_DEVICES=$2 "./$1/bin/llama-bench" -m "$MODEL" -ngl 99 -r 5 > "$LOG/bench-$3.log" 2>&1
  echo "--- llama-bench $3 ---" | tee -a "$SUM"
  grep -E "^\|" "$LOG/bench-$3.log" | tee -a "$SUM"
}

if [ -z "${SKIP_K80:-}" ]; then
  echo "=== Tesla K80 (GPU $K80_GPU, sm_37) ===" | tee -a "$SUM"
  configure build-k80 37
  # ohne Patch bricht test-backend-ops auf der K80 ab, llama-bench kann ebenfalls abbrechen
  set_patch none; test_ops build-k80 "$K80_GPU" k80-ohne-patch; bench build-k80 "$K80_GPU" k80-ohne-patch
  set_patch v2;   test_ops build-k80 "$K80_GPU" k80-v2;         bench build-k80 "$K80_GPU" k80-v2
fi

if [ -z "${SKIP_NEW:-}" ]; then
  echo "=== Neuere Karte (GPU $NEW_GPU, sm_$NEW_ARCH) ===" | tee -a "$SUM"
  configure build-new "$NEW_ARCH"
  set_patch none; test_ops build-new "$NEW_GPU" new-ohne-patch; bench build-new "$NEW_GPU" new-ohne-patch
  set_patch v2;   test_ops build-new "$NEW_GPU" new-v2;         bench build-new "$NEW_GPU" new-v2
fi

git checkout -- "$D"   # Stand wieder sauber
echo "Logs: $LOG"
