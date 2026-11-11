"""Fase 2: gera versões perturbadas de áudios do domínio de treino, um fator por vez.

Amostra: n genuínos e n falsos do ASVspoof 2021 DF eval, de origem VCTK/ASVspoof (coluna origem =
asvspoof) e sem codec (codec = nocodec), sorteados com semente fixa a partir da chave completa
(_diag.CHAVE_DF21). Cada áudio é gravado na condição "original" (só convertido para wav) e em cada
perturbação. Assim a análise compara o MESMO áudio antes e depois (f2_analisar.py).

Todas as saídas ficam a 16 kHz, em wav float32 (sem quantização extra), exceto onde o próprio
fator muda o sinal (codec, reamostragem). O pré-processamento do modelo (corte em 64.600 amostras
e preenchimento por repetição) não muda: é aplicado depois, pelo inferir.py do Experimento 01.

Grava em --saida:
  <condicao>/<trial>.wav, lista.tsv (id, caminho), rotulos.tsv (id, rotulo, grupo = condição),
  selecao.tsv (os trials sorteados) e perturbacoes.meta.json (parâmetros, versões, ffmpeg).
O id de cada áudio é "<condicao>__<trial>".

Roda FORA do container (precisa de numpy, scipy e soundfile e, para os codecs, do ffmpeg), como o
extrair_brspeech_hf.py, porque grava na pasta de dados, que o container monta somente leitura.
Sem ffmpeg, as condições de codec são puladas e isso fica registrado no .meta.json.

Exemplo (no servidor):
    python3 -B f2_perturbar.py --audio ~/matheus/dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac \
        --saida ~/matheus/dados/exp02_perturbacoes --raiz-container /dados/exp02_perturbacoes --n 400
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
import soundfile as sf
from scipy import signal

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))
from _comum import BRSPEECH_REPO, erro, escrever_tsv, garantir_fora_third_party, resumo, sha256  # noqa: E402

CHAVE = os.path.join(BRSPEECH_REPO, 'data', 'metadata', 'trial_metadata.txt')
SR = 16000


# ---------------------------------------------------------------------------------------------
# Perturbações: f(x, rng) -> y, com x e y em float32 a 16 kHz

def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def aparar(x, top_db, quadro=400, passo=160):
    """Remove o silêncio do início e do fim: quadros com energia abaixo de (máximo − top_db)."""
    if x.size < quadro:
        return x
    n = 1 + (x.size - quadro) // passo
    e = np.array([np.mean(x[i * passo:i * passo + quadro] ** 2) for i in range(n)])
    db = 10 * np.log10(e + 1e-12)
    ativos = np.flatnonzero(db > db.max() - top_db)
    if ativos.size == 0:
        return x
    return x[ativos[0] * passo: min(x.size, ativos[-1] * passo + quadro)]


def silencio_inicio(seg, nivel_dbfs=None):
    def f(x, rng):
        n = int(seg * SR)
        pre = np.zeros(n, np.float32) if nivel_dbfs is None else \
            (rng.standard_normal(n) * 10 ** (nivel_dbfs / 20)).astype(np.float32)
        return np.concatenate([pre, x])
    return f


def filtro(tipo, freqs, ordem=8):
    """Filtro Butterworth de fase zero. O RMS volta ao original, para que o volume (fator próprio,
    'ganho') não mude junto com a banda: um fator por vez."""
    sos = signal.butter(ordem, freqs, btype=tipo, fs=SR, output='sos')

    def f(x, rng):
        y = signal.sosfiltfilt(sos, x)
        return np.clip(y / rms(y) * rms(x), -1, 1).astype(np.float32)
    return f


def ruido(snr_db, cor):
    def f(x, rng):
        n = rng.standard_normal(x.size)
        if cor == 'rosa':  # espectro 1/f via FFT
            X = np.fft.rfft(n)
            k = np.arange(X.size)
            k[0] = 1
            n = np.fft.irfft(X / np.sqrt(k), x.size)
        n = n / rms(n) * rms(x) / 10 ** (snr_db / 20)
        return (x + n).astype(np.float32)
    return f


def reverberacao(rt60):
    def f(x, rng):
        n = int(1.2 * rt60 * SR)
        t = np.arange(n) / SR
        rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # −60 dB em rt60
        rir[0] = 1.0                                             # caminho direto
        rir /= np.sqrt(np.sum(rir ** 2))
        y = signal.fftconvolve(x, rir)[:x.size]
        return (y / rms(y) * rms(x)).astype(np.float32)
    return f


def ganho(db):
    return lambda x, rng: np.clip(x * 10 ** (db / 20), -1, 1).astype(np.float32)


def reamostrar(sr_meio):
    from math import gcd
    g = gcd(SR, sr_meio)
    up, down = sr_meio // g, SR // g
    return lambda x, rng: signal.resample_poly(signal.resample_poly(x, up, down), down, up).astype(np.float32)[:x.size]


def codec(args_ffmpeg, ext):
    def f(x, rng):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = os.path.join(d, 'a.wav'), os.path.join(d, 'b' + ext), os.path.join(d, 'c.wav')
            sf.write(a, x, SR, subtype='FLOAT')
            for cmd in (['ffmpeg', '-v', 'error', '-y', '-i', a] + args_ffmpeg + [b],
                        ['ffmpeg', '-v', 'error', '-y', '-i', b, '-ar', str(SR), '-ac', '1', '-c:a', 'pcm_f32le', c]):
                subprocess.run(cmd, check=True)
            y, sr = sf.read(c, dtype='float32')
            if sr != SR:
                erro(f'codec devolveu {sr} Hz')
            return y
    return f


def condicoes():
    """(nome, função, fator, precisa_ffmpeg). A ordem segue a prioridade do README, seção 6."""
    c = [('original', lambda x, rng: x, 'controle', False)]
    c += [(f'aparar_{t}db', (lambda t: lambda x, rng: aparar(x, t))(t), 'silencio', False) for t in (40, 60)]
    c += [(f'silencio_{s}s', silencio_inicio(s), 'silencio', False) for s in (0.25, 0.5, 1.0, 2.0)]
    c += [(f'ruido_fundo_{s}s', silencio_inicio(s, -60), 'silencio', False) for s in (0.5, 1.0)]
    c += [('banda_telefone', filtro('bandpass', [300, 3400]), 'banda', False),
          ('passa_baixas_4k', filtro('lowpass', 4000), 'banda', False),
          ('passa_baixas_6k', filtro('lowpass', 6000), 'banda', False)]
    c += [(f'mp3_{k}k', codec(['-c:a', 'libmp3lame', '-b:a', f'{k}k'], '.mp3'), 'codec', True) for k in (16, 32, 64)]
    c += [('opus_16k', codec(['-c:a', 'libopus', '-b:a', '16k'], '.opus'), 'codec', True)]
    c += [(f'ruido_{cor}_{s}db', ruido(s, cor), 'ruido', False) for cor in ('branco', 'rosa') for s in (30, 20, 10)]
    c += [(f'reverb_{r}s', reverberacao(r), 'reverberacao', False) for r in (0.3, 0.6, 1.0)]
    c += [(f'ganho_{d:+d}db', ganho(d), 'volume', False) for d in (-20, -10, 6)]
    c += [('reamostrar_8k', reamostrar(8000), 'reamostragem', False)]
    return c


# ---------------------------------------------------------------------------------------------

def selecionar(chave, n, semente):
    gen, fal = [], []
    with open(chave, encoding='utf-8') as f:
        for linha in f:
            col = linha.split()
            if col[7] == 'eval' and col[3] == 'asvspoof' and col[2] == 'nocodec':
                (gen if col[5] == 'bonafide' else fal).append(col)
    rng = np.random.default_rng(semente)
    # Há só 386 genuínos VCTK sem codec no eval: se n for maior, usa todos (registrado no meta).
    pick = lambda xs: [xs[i] for i in sorted(rng.choice(len(xs), min(n, len(xs)), replace=False))]  # noqa: E731
    return pick(gen) + pick(fal)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio', required=True, help='pasta com os .flac do 2021 DF eval')
    ap.add_argument('--saida', required=True)
    ap.add_argument('--n', type=int, default=400, help='máximo por classe (há só 386 genuínos VCTK sem codec)')
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--chave', default=CHAVE)
    ap.add_argument('--so', nargs='*', help='gera só estas condições')
    ap.add_argument('--raiz-container', default=None,
                    help='caminho de --saida visto de dentro do container (ex.: /dados/exp02_perturbacoes); '
                         'é o que vai na lista.tsv, porque o inferir.py roda no container')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)
    tem_ffmpeg = shutil.which('ffmpeg') is not None

    sel = selecionar(args.chave, args.n, args.semente)
    conds = [c for c in condicoes() if not args.so or c[0] in args.so or c[0] == 'original']
    puladas = [c[0] for c in conds if c[3] and not tem_ffmpeg]
    conds = [c for c in conds if not (c[3] and not tem_ffmpeg)]

    lista, rotulos = [], []
    for i, col in enumerate(sel):
        trial, rot = col[1], col[5]
        x, sr = sf.read(os.path.join(args.audio, trial + '.flac'), dtype='float32')
        if x.ndim != 1 or sr != SR:
            erro(f'{trial}: esperado mono a 16 kHz, veio {x.shape} a {sr} Hz')
        for k, (nome, fn, fator, _) in enumerate(conds):
            # semente própria por (áudio, condição): reproduzível e independente da ordem
            rng = np.random.default_rng([args.semente, i, k])
            destino = os.path.join(args.saida, nome, trial + '.wav')
            if not os.path.exists(destino):
                y = np.asarray(fn(x, rng), dtype=np.float32)
                if y.size == 0 or not np.all(np.isfinite(y)):
                    erro(f'{nome}/{trial}: saída vazia ou não finita')
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                sf.write(destino + '.tmp.wav', y, SR, subtype='FLOAT')
                os.replace(destino + '.tmp.wav', destino)
            caminho_lista = (f'{args.raiz_container.rstrip("/")}/{nome}/{trial}.wav' if args.raiz_container
                             else os.path.abspath(destino))
            lista.append((f'{nome}__{trial}', caminho_lista))
            rotulos.append((f'{nome}__{trial}', rot, nome))
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(sel)} áudios', flush=True)

    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'rotulos.tsv'), ['id', 'rotulo', 'grupo'], rotulos)
    escrever_tsv(os.path.join(args.saida, 'selecao.tsv'), ['trial', 'rotulo', 'locutor', 'ataque'],
                 [(c[1], c[5], c[0], c[4]) for c in sel])
    ffmpeg_v = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0] \
        if tem_ffmpeg else None
    meta = {'n_por_classe_max': args.n, 'n_genuinos': sum(c[5] == 'bonafide' for c in sel), 'n_falsos': sum(c[5] != 'bonafide' for c in sel), 'semente': args.semente, 'chave': args.chave, 'chave_sha256': sha256(args.chave),
            'condicoes': [{'nome': c[0], 'fator': c[2]} for c in conds], 'puladas_sem_ffmpeg': puladas,
            'ffmpeg': ffmpeg_v, 'numpy': np.__version__, 'soundfile': sf.__version__,
            'python': platform.python_version()}
    with open(os.path.join(args.saida, 'perturbacoes.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    resumo('f2_perturbar.py', [('áudios sorteados', len(sel)), ('condições', len(conds)),
                               ('arquivos', len(lista)), ('puladas (sem ffmpeg)', puladas or '-'),
                               ('saída', args.saida)])


if __name__ == '__main__':
    main()
