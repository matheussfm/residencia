# Ambiente do Experimento 01 (linha de base acústica: SSL_Anti-spoofing, Tak et al., 2022).
#
# Baseado em third_party/BRSpeech-DF-Dataset/models/ssl_antispoof/Dockerfile (AKCIT, MIT), com
# três diferenças:
#   - o código do Tak NÃO é clonado: vem do submódulo, montado somente leitura em tempo de
#     execução (scripts/em_docker.sh), então a versão é exatamente a registrada no repositório;
#   - o fairseq é copiado do submódulo para dentro da imagem e instalado a partir da cópia
#     (nada é escrito no submódulo);
#   - sem Jupyter e sem checkpoints na imagem (os checkpoints são montados em /ckpt).
#
# Build, a partir da raiz do repositório (o .dockerignore da raiz limita o que é enviado):
#   docker build -f experimentos/01-linha-de-base/docker/Dockerfile -t residencia-exp01:pt181 .
# Se a RTX 4090 não funcionar com PyTorch 1.8.1 + CUDA 11.1, trocar a base:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime \
#       -f experimentos/01-linha-de-base/docker/Dockerfile -t residencia-exp01:pt1131 .
ARG BASE_IMAGE=pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# fairseq no commit fixado pelo Tak (a54021305d6b3c4c5959ac9395135f63202db8f1).
COPY third_party/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 /opt/fairseq
RUN pip install --no-cache-dir --editable /opt/fairseq

COPY experimentos/01-linha-de-base/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Permite que o usuário não-root do docker run (--user) use git no repositório montado.
RUN git config --system --add safe.directory '*'

WORKDIR /work/experimentos/01-linha-de-base/scripts
CMD ["bash"]
