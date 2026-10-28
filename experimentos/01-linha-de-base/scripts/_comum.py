"""Utilitários compartilhados pelos scripts do Experimento 01.

Regra: nada aqui (nem nos scripts) escreve dentro de third_party/.
"""
import csv
import hashlib
import os
import pickle
import sys

sys.dont_write_bytecode = True

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
THIRD_PARTY = os.path.join(RAIZ_REPO, 'third_party')
SSL_REPO = os.path.join(THIRD_PARTY, 'SSL_Anti-spoofing')
BRSPEECH_REPO = os.path.join(THIRD_PARTY, 'BRSpeech-DF-Dataset')

ROTULOS_VALIDOS = ('bonafide', 'spoof')


def erro(msg):
    sys.exit(f'ERRO: {msg}')


def garantir_fora_third_party(caminho):
    """Aborta se `caminho` (arquivo ou pasta de saída) estiver dentro de third_party/."""
    real = os.path.realpath(caminho)
    tp = os.path.realpath(THIRD_PARTY)
    if real == tp or real.startswith(tp + os.sep):
        erro(f'saída dentro de third_party/ não é permitida: {caminho}')


def escrever_tsv(caminho, cabecalho, linhas):
    """Escreve TSV de forma atômica (arquivo temporário + rename)."""
    garantir_fora_third_party(caminho)
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    tmp = caminho + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(cabecalho)
        w.writerows(linhas)
    os.replace(tmp, caminho)


def ler_tsv(caminho):
    """Lê TSV com cabeçalho; devolve lista de dicts."""
    with open(caminho, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def validar_id(id_):
    if not id_ or any(c.isspace() for c in id_):
        erro(f'id vazio ou com espaço (quebraria o formato "id pontuacao"): {id_!r}')


class _UnpicklerRestrito(pickle.Unpickler):
    """Só aceita as classes que os .pkl de pontuação do AKCIT usam (escalares numpy)."""
    PERMITIDAS = {('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')}

    def find_class(self, module, name):
        if (module, name) not in self.PERMITIDAS:
            raise pickle.UnpicklingError(f'classe não permitida no pickle: {module}.{name}')
        return super().find_class(module, name)


def ler_pontuacoes(caminho):
    """Lê pontuações como dict id -> float.

    Aceita o formato texto do Tak ("id pontuacao" por linha) ou um .pkl {id: escalar} do AKCIT.
    """
    if caminho.endswith('.pkl'):
        with open(caminho, 'rb') as f:
            return {k: float(v) for k, v in _UnpicklerRestrito(f).load().items()}
    pont = {}
    with open(caminho, encoding='utf-8') as f:
        for n, linha in enumerate(f, 1):
            partes = linha.split()
            if not partes:
                continue
            if len(partes) != 2:
                erro(f'{caminho}:{n}: esperado "id pontuacao", veio {linha.strip()!r}')
            if partes[0] in pont:
                erro(f'{caminho}:{n}: id repetido {partes[0]!r}')
            pont[partes[0]] = float(partes[1])
    return pont


def checar_taxas(caminhos, n=200):
    """Conta as taxas de amostragem de até n arquivos (o librosa reamostra o que não for 16 kHz)."""
    import soundfile
    contagem = {}
    for c in caminhos[:n]:
        taxa = soundfile.info(c).samplerate
        contagem[taxa] = contagem.get(taxa, 0) + 1
    return contagem


def conferir_contagem(nome, obtido, esperado):
    if obtido != esperado:
        print(f'AVISO: {nome} = {obtido}, esperado {esperado}')


def sha256(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, 'rb') as f:
        for parte in iter(lambda: f.read(bloco), b''):
            h.update(parte)
    return h.hexdigest()


def resumo(titulo, itens):
    """Imprime um bloco curto, fácil de copiar e colar."""
    print(f'\n==== RESUMO {titulo} ====')
    for k, v in itens:
        print(f'{k}: {v}')
    print('=' * (13 + len(titulo)))
