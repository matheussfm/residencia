#!/usr/bin/env bash
# Baixa os dois checkpoints usados no Experimento 01 e registra o SHA-256 de cada um.
#   bash experimentos/01-linha-de-base/scripts/baixar_checkpoints.sh <pasta_destino>
#
# - Best_LA_model_for_DF.pth: modelo DF pré-treinado do Tak et al. (2022), pelo mesmo ID do
#   Google Drive usado em BRSpeech-DF-Dataset/models/ssl_antispoof/Dockerfile (AKCIT).
# - xlsr2_300m.pt: XLS-R 300M do fairseq (front-end wav2vec 2.0).
# Precisa de wget e do gdown (pip install --user gdown==5.2.0). Se o Google Drive recusar o
# download por excesso de acessos, baixe pelo navegador e copie para a pasta destino.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "uso: $0 <pasta_destino>"
    exit 1
fi
RAIZ="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$1"
DEST="$(cd "$1" && pwd)"
case "$DEST/" in
    "$RAIZ/third_party/"*) echo "ERRO: destino dentro de third_party/"; exit 1 ;;
esac

if [ ! -f "$DEST/Best_LA_model_for_DF.pth" ]; then
    python3 -m gdown "https://drive.google.com/uc?id=1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG" \
        -O "$DEST/Best_LA_model_for_DF.pth"
fi
if [ ! -f "$DEST/xlsr2_300m.pt" ]; then
    wget --progress=dot:giga https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt \
        -O "$DEST/xlsr2_300m.pt"
fi

cd "$DEST"
sha256sum Best_LA_model_for_DF.pth xlsr2_300m.pt | tee checkpoints.sha256
ls -la "$DEST"
