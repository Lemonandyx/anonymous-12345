#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PREFIX="$(${PYTHON} -c 'import sys; print(sys.prefix)')"
PY_INCLUDE="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
TORCH_INCLUDE="$(${PYTHON} -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')"
EXT_SUFFIX="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${1:-${ROOT}/native/harmorag_native_scanner.cpp}"
OUTPUT="${2:-${ROOT}/harmorag_native_scanner${EXT_SUFFIX}}"

# Override when your Faiss build uses a different library name (for example
# faiss instead of the AVX2 build used for the paper's EPYC 7742 host).
FAISS_LIBRARY="${FAISS_LIBRARY:-faiss_avx2}"
if [[ ! -e "${PREFIX}/lib/lib${FAISS_LIBRARY}.so" && ! -e "${PREFIX}/lib/lib${FAISS_LIBRARY}.dylib" ]]; then
  echo "missing ${PREFIX}/lib/lib${FAISS_LIBRARY}.{so,dylib}" >&2
  echo "install Faiss 1.8.0 in the active environment or set FAISS_LIBRARY" >&2
  exit 2
fi

g++ -std=c++17 -O3 -march=native -fPIC -shared -fopenmp -pthread \
  -I"${PY_INCLUDE}" \
  -I"${TORCH_INCLUDE}" \
  -I"${PREFIX}/include" \
  -I"${ROOT}/native" \
  "${SOURCE}" \
  -L"${PREFIX}/lib" \
  -Wl,-rpath,"${PREFIX}/lib" \
  -l"${FAISS_LIBRARY}" \
  -o "${OUTPUT}"

echo "built ${OUTPUT} (${FAISS_LIBRARY})"
