#!/usr/bin/env bash
# Confere que os submódulos em third_party/ estão intactos: no commit registrado e
# sem nenhum arquivo alterado, novo ou ignorado. Sai com erro se algo mudou.
# Rodar no servidor (fora do container), antes e depois de cada cenário:
#   bash experimentos/01-linha-de-base/scripts/verificar_submodulos.sh
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$RAIZ"

echo "== git submodule status =="
git --no-optional-locks submodule status

falhou=0
if git --no-optional-locks submodule status | grep -q '^[-+U]'; then
    echo "ERRO: submódulo não inicializado ou fora do commit registrado (prefixo - + ou U acima)"
    falhou=1
fi
for sm in third_party/SSL_Anti-spoofing third_party/BRSpeech-DF-Dataset; do
    alteracoes="$(git --no-optional-locks -C "$sm" status --porcelain --ignored)"
    if [ -n "$alteracoes" ]; then
        echo "ERRO: $sm tem alterações:"
        echo "$alteracoes" | head -20
        falhou=1
    else
        echo "OK: $sm intacto"
    fi
done
exit $falhou
