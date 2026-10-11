#!/usr/bin/env bash
# Gegenmessung der ggml-Kepler-Patches 0001/0002
#
# Aufruf:
#   ./gegenmessung-k80.sh <llama.cpp-Verzeichnis> <Ordner mit 0001-/0002-Patch> [Modell.gguf]
#
# Teil 1 - Tesla K80 (Korrektheit), drei Laeufe auf demselben Stand:
#   ohne Patch, nur 0001, 0001+0002
# Teil 2 - neuere Karte, z. B. RTX 3090 (keine Regression):
#   test-backend-ops ohne Patch und mit 0001+0002,
#   optional llama-bench mit/ohne Patch, wenn ein Modell angegeben ist
#
# Umgebungsvariablen (Nummern wie in nvidia-smi):
#   K80_GPU=1   NEW_GPU=0   NEW_ARCH=86   SKIP_K80=1   SKIP_NEW=1
set -u
SRC=$(realpath "$1"); PD=$(realpath "$2"); MODEL=${3:-}
P1="$PD/0001-cuda-kepler-avoid-cublas-ex.patch"
P2="$PD/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch"
K80_GPU=${K80_GPU:-1}; NEW_GPU=${NEW_GPU:-0}; NEW_ARCH=${NEW_ARCH:-86}
LOG="$PWD/gegenmessung-$(date +%Y%m%d-%H%M)"; mkdir -p "$LOG"
SUM="$LOG/zusammenfassung.txt"

# Geraetenummern wie in nvidia-smi (CUDA sortiert sonst nach Geschwindigkeit)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$SRC"
F=ggml/src/ggml-cuda/ggml-cuda.cu
git checkout -- "$F"
echo "llama.cpp $(git rev-parse --short HEAD), $(date '+%F %T')" | tee "$SUM"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader >> "$SUM"

configure() {  # $1 = Build-Ordner, $2 = CUDA-Architektur
  [ -d "$1" ] || CC=gcc-10 CXX=g++-10 cmake -B "$1" -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="$2" -DCMAKE_BUILD_TYPE=Release > "$LOG/cmake-$1.log" 2>&1
}

set_patch() {  # $1 = none | 0001 | all
  git checkout -- "$F"
  case "$1" in
    0001) git apply "$P1" ;;
    all)  git apply "$P1" && git apply "$P2" ;;
  esac
}

test_ops() {  # $1 = Build-Ordner, $2 = GPU, $3 = Name
  local name="$3"
  cmake --build "$1" -j --target test-backend-ops > "$LOG/build-$name.log" 2>&1
  echo "BUILD_EXIT=$?" >> "$LOG/build-$name.log"
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
  cmake --build "$1" -j --target llama-bench > "$LOG/build-bench-$3.log" 2>&1
  CUDA_VISIBLE_DEVICES=$2 "./$1/bin/llama-bench" -m "$MODEL" -ngl 99 -r 5 > "$LOG/bench-$3.log" 2>&1
  echo "--- llama-bench $3 ---" | tee -a "$SUM"
  grep -E "^\|" "$LOG/bench-$3.log" | tee -a "$SUM"
}

if [ -z "${SKIP_K80:-}" ]; then
  echo "=== Tesla K80 (GPU $K80_GPU, sm_37) ===" | tee -a "$SUM"
  configure build-k80 37
  set_patch none; test_ops build-k80 "$K80_GPU" k80-ohne-patch
  set_patch 0001; test_ops build-k80 "$K80_GPU" k80-nur-0001
  set_patch all;  test_ops build-k80 "$K80_GPU" k80-0001+0002
fi

if [ -z "${SKIP_NEW:-}" ]; then
  echo "=== Neuere Karte (GPU $NEW_GPU, sm_$NEW_ARCH) ===" | tee -a "$SUM"
  configure build-new "$NEW_ARCH"
  set_patch none; test_ops build-new "$NEW_GPU" new-ohne-patch; bench build-new "$NEW_GPU" new-ohne-patch
  set_patch all;  test_ops build-new "$NEW_GPU" new-0001+0002;  bench build-new "$NEW_GPU" new-0001+0002
fi

git checkout -- "$F"   # Stand wieder sauber
echo "Logs: $LOG"
