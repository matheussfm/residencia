#!/usr/bin/env bash
# Roda um comando dentro do container do Experimento 01.
#
#   bash experimentos/01-linha-de-base/scripts/em_docker.sh <gpu> <comando...>
#   <gpu>: 0, 1 ou "nenhuma"
#
# Montagens:
#   repositório           -> /work            (leitura e escrita: resultados/ e listas/)
#   third_party/          -> /work/third_party (SOMENTE LEITURA: submódulos nunca são alterados)
#   $DADOS_DIR            -> /dados            (somente leitura)
#   $CKPT_DIR             -> /ckpt             (somente leitura)
# O diretório de trabalho é /work/experimentos/01-linha-de-base/scripts.
# Os caminhos vêm de experimentos/01-linha-de-base/config/servidor.env.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG="$RAIZ/experimentos/01-linha-de-base/config/servidor.env"
if [ ! -f "$CONFIG" ]; then
    echo "ERRO: crie $CONFIG a partir de servidor.env.example"
    exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG"
: "${DADOS_DIR:?defina DADOS_DIR em servidor.env}"
: "${CKPT_DIR:?defina CKPT_DIR em servidor.env}"
: "${IMAGEM:?defina IMAGEM em servidor.env}"

if [ $# -lt 2 ]; then
    echo "uso: $0 <gpu: 0|1|nenhuma> <comando...>"
    exit 1
fi
gpu="$1"
shift

opcoes=()
if [ "$gpu" != "nenhuma" ]; then
    opcoes+=(--gpus "device=$gpu")
fi
if [ -t 0 ] && [ -t 1 ]; then
    opcoes+=(-it)
fi

exec ${DOCKER:-docker} run --rm ${opcoes[@]+"${opcoes[@]}"} \
    --user "$(id -u):$(id -g)" \
    --shm-size=8g \
    -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e NUMBA_CACHE_DIR=/tmp/numba \
    -v "$RAIZ:/work" \
    -v "$RAIZ/third_party:/work/third_party:ro" \
    -v "$DADOS_DIR:/dados:ro" \
    -v "$CKPT_DIR:/ckpt:ro" \
    -w /work/experimentos/01-linha-de-base/scripts \
    "$IMAGEM" "$@"
