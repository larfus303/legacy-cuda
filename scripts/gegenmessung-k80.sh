#!/usr/bin/env bash
# Gegenmessung Kepler-Patches auf Tesla K80
# Aufruf: ./gegenmessung-k80.sh <llama.cpp-Verzeichnis> <Ordner mit 0001-/0002-Patch>
# Drei Laeufe auf demselben Stand: ohne Patch, nur 0001, 0001+0002
set -u
SRC=$(realpath "$1"); PD=$(realpath "$2")
P1="$PD/0001-cuda-kepler-avoid-cublas-ex.patch"
P2="$PD/0002-cuda-no-mmq-skip-noncont-quant-without-nc-converter.patch"
LOG="$PWD/gegenmessung-$(date +%Y%m%d-%H%M)"; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # nur die K80

cd "$SRC"
git checkout -- ggml/src/ggml-cuda/ggml-cuda.cu
git rev-parse --short HEAD > "$LOG/commit.txt"

[ -d build ] || CC=gcc-10 CXX=g++-10 cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=37 -DCMAKE_BUILD_TYPE=Release > "$LOG/cmake.log" 2>&1

run() {  # $1 = Name
  cmake --build build -j --target test-backend-ops > "$LOG/build-$1.log" 2>&1
  echo "BUILD_EXIT=$?" >> "$LOG/build-$1.log"
  ./build/bin/test-backend-ops -o MUL_MAT > "$LOG/test-$1.log" 2>&1
  local rc=$?
  # ANSI-Farbcodes entfernen, sonst zaehlt grep die OK-Zeilen nicht
  sed 's/\x1b\[[0-9;]*m//g' "$LOG/test-$1.log" > "$LOG/test-$1.plain.log"
  local ok=$(grep -c "OK\s*$" "$LOG/test-$1.plain.log")
  local fail=$(grep -c "FAIL" "$LOG/test-$1.plain.log")
  printf "%-12s exit=%-4s OK-Zeilen=%-5s FAIL=%-4s %s\n" "$1" "$rc" "$ok" "$fail" \
    "$(grep -E 'tests passed|GGML_ASSERT|CUBLAS_STATUS' "$LOG/test-$1.plain.log" | tail -1)" | tee -a "$LOG/zusammenfassung.txt"
}

run ohne-patch
git apply "$P1" && run nur-0001
git apply "$P2" && run 0001+0002
git checkout -- ggml/src/ggml-cuda/ggml-cuda.cu   # Stand wieder sauber
echo "Logs: $LOG"
