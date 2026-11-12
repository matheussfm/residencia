"""Medidas acústicas de cada áudio, sem modelo (liga a Fase 1 aos fatores da Fase 2).

Para cada áudio da lista (por padrão, a amostra da Fase 3: 2021 DF eval, hidden track,
In-the-Wild e BRSpeech-DF), na taxa de amostragem original:
  - silencio_inicio_s / silencio_fim_s: duração do trecho inicial/final com energia abaixo de
    (máximo − 40 dB), em quadros de 25 ms com passo de 10 ms (o mesmo critério do aparar_40db);
  - snr_est_db: percentil 95 menos percentil 10 da energia por quadro (estimativa grosseira da
    relação entre fala e fundo);
  - nivel_dbfs: RMS do áudio inteiro, em dBFS;
  - pico, frac_clipping: pico absoluto e fração de amostras com |x| >= 0,999;
  - banda_99_hz: frequência abaixo da qual está 99% da energia (Welch);
  - frac_graves_300hz: fração da energia abaixo de 300 Hz (a banda de telefone corta essa faixa);
  - sr, duracao_s.

Roda no servidor, dentro do container (o soundfile da imagem lê os .flac do 2021 DF), porque
precisa dos áudios. A análise é feita no PC (f1_qualidade_analise.py).

Exemplo:
    bash experimentos/01-linha-de-base/scripts/em_docker.sh nenhuma python \
        ../../02-diagnostico/scripts/f1_qualidade_audio.py \
        --lista ../../02-diagnostico/listas/fase3/lista.tsv \
        --saida ../../02-diagnostico/resultados/qualidade/medidas.tsv
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))

import numpy as np
import soundfile as sf
from scipy import signal

from _comum import escrever_tsv, garantir_fora_third_party, ler_tsv

LIMIAR_DB = 40.0


def medir(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    q, h = int(0.025 * sr), int(0.010 * sr)
    n = max(1, 1 + (x.size - q) // h)
    e = np.array([np.mean(x[i * h:i * h + q] ** 2) for i in range(n)]) + 1e-12
    db = 10 * np.log10(e)
    ativo = np.flatnonzero(db > db.max() - LIMIAR_DB)
    ini = ativo[0] * h / sr if ativo.size else x.size / sr
    fim = max(0.0, (x.size - (ativo[-1] * h + q)) / sr) if ativo.size else x.size / sr
    f, psd = signal.welch(x, fs=sr, nperseg=min(1024, x.size))
    acum = np.cumsum(psd) / max(psd.sum(), 1e-20)
    return {
        'sr': sr, 'duracao_s': x.size / sr,
        'silencio_inicio_s': ini, 'silencio_fim_s': fim,
        'snr_est_db': float(np.percentile(db, 95) - np.percentile(db, 10)),
        'nivel_dbfs': float(10 * np.log10(np.mean(x ** 2) + 1e-12)),
        'pico': float(np.abs(x).max()), 'frac_clipping': float(np.mean(np.abs(x) >= 0.999)),
        'banda_99_hz': float(f[min(np.searchsorted(acum, 0.99), f.size - 1)]),
        'frac_graves_300hz': float(psd[f < 300].sum() / max(psd.sum(), 1e-20)),
    }


CAMPOS = ['sr', 'duracao_s', 'silencio_inicio_s', 'silencio_fim_s', 'snr_est_db', 'nivel_dbfs', 'pico',
          'frac_clipping', 'banda_99_hz', 'frac_graves_300hz']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lista', required=True)
    ap.add_argument('--saida', required=True)
    ap.add_argument('--limite', type=int, default=None)
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)
    itens = [(r['id'], r['caminho']) for r in ler_tsv(args.lista)]
    if args.limite:
        itens = itens[:args.limite]
    linhas, falhas = [], []
    for n, (id_, c) in enumerate(itens, 1):
        try:
            x, sr = sf.read(c, dtype='float32')
            m = medir(x, sr)
            linhas.append([id_] + [f'{m[k]:.6g}' for k in CAMPOS])
        except Exception as e:  # registra e segue
            falhas.append((id_, str(e)[:80]))
        if n % 1000 == 0:
            print(f'  {n}/{len(itens)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    escrever_tsv(args.saida, ['id'] + CAMPOS, linhas)
    if falhas:
        escrever_tsv(args.saida + '.falhas.tsv', ['id', 'erro'], falhas)
    print(f'ok: {len(linhas)} medidos, {len(falhas)} falhas; {args.saida}')


if __name__ == '__main__':
    main()
