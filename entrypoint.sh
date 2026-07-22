set -e

CUBLAS_DIR=$(python3 -c "import nvidia.cublas; print(list(nvidia.cublas.__path__)[0] + '/lib')" 2>/dev/null || echo "")
CUDNN_DIR=$(python3 -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + '/lib')" 2>/dev/null || echo "")

export LD_LIBRARY_PATH="${CUBLAS_DIR}:${CUDNN_DIR}:${LD_LIBRARY_PATH}"

exec "$@"
