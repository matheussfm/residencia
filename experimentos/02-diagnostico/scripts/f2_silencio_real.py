"""Fase 2b: normalização do silêncio em dados reais fora do domínio (In-the-Wild e BRSpeech-DF).

A Fase 2 mostrou que o silêncio é um atalho causal, e as medidas acústicas (README 4.10) mostraram
que no In-the-Wild ele aponta para o lado errado (genuínos sem silêncio inicial, falsos com).
Aqui o silêncio é normalizado em TODOS os áudios, nas duas classes, e os áudios são pontuados de
novo. Se a EER cair, essa parte da perda vem do atalho do silêncio.

Condições:
  - aparar:     corta o silêncio do início e do fim (quadros abaixo de máximo − 40 dB; quadros de
                25 ms com passo de 10 ms, o mesmo critério do f1_qualidade_audio.py);
  - padronizar: corta como acima e acrescenta exatamente 0,5 s de silêncio digital no início de
                todos os áudios (remove a diferença entre classes sem tirar o silêncio inicial que
                o modelo espera).

Os áudios são gravados em FLAC 16 bits na taxa original (cortar não muda os valores das amostras;
as fontes são de 16 bits). Os ids são os do Experimento 01, para usar os mesmos rotulos.tsv.

Entrada: as listas do Experimento 01 (caminhos do container, /dados/...), traduzidos para o host
por --raiz-host-dados. Roda FORA do container (grava na pasta de dados), no ambiente com numpy,
scipy e soundfile; recorre ao ffmpeg se a libsndfile recusar um arquivo.

Saída em --saida: <condicao>/<cenario>/... (áudios) e <condicao>_<cenario>.lista.tsv (caminhos do
container, para o inferir.py), mais silencio_real.meta.json.

Exemplo (no servidor):
    python -B f2_silencio_real.py --raiz-host-dados ~/matheus/dados \
        --saida ~/matheus/dados/exp02_silencio --raiz-container /dados/exp02_silencio
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))

import numpy as np
import soundfile as sf

from _comum import erro, escrever_tsv, garantir_fora_third_party, ler_tsv, resumo

LIS01 = os.path.join(AQUI, '..', '..', '01-linha-de-base', 'listas')
CENARIOS = {'itw': 'in_the_wild', 'brs': 'brspeech_test_hf'}
LIMIAR_DB = 40.0


def aparar(x, sr):
    q, h = int(0.025 * sr), int(0.010 * sr)
    if x.size < q:
        return x
    n = 1 + (x.size - q) // h
    e = np.array([np.mean(x[i * h:i * h + q] ** 2) for i in range(n)]) + 1e-12
    db = 10 * np.log10(e)
    ativo = np.flatnonzero(db > db.max() - LIMIAR_DB)
    if ativo.size == 0:
        return x
    y = x[ativo[0] * h: min(x.size, ativo[-1] * h + q)]
    return y if y.size >= int(0.1 * sr) else x  # não deixa um áudio praticamente vazio


CONDICOES = {
    'aparar': lambda x, sr: aparar(x, sr),
    'padronizar': lambda x, sr: np.concatenate([np.zeros(int(0.5 * sr), dtype=x.dtype), aparar(x, sr)]),
}


def ler(caminho):
    try:
        x, sr = sf.read(caminho, dtype='float32')
        via = False
    except Exception:
        import subprocess
        r = subprocess.run(['ffmpeg', '-v', 'error', '-i', caminho, '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1', '-'],
                           capture_output=True, check=True)
        sr = int(subprocess.run(['ffmpeg', '-hide_banner', '-i', caminho], capture_output=True, text=True).stderr
                 .split(' Hz')[0].rsplit(' ', 1)[-1])
        x, via = np.frombuffer(r.stdout, dtype=np.float32).copy(), True
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr, via


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz-host-dados', required=True, help='pasta do host montada como /dados no container')
    ap.add_argument('--saida', required=True)
    ap.add_argument('--raiz-container', required=True, help='--saida vista de dentro do container')
    ap.add_argument('--listas', default=LIS01)
    ap.add_argument('--limite', type=int, default=None, help='só os N primeiros de cada cenário (teste)')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)
    host = os.path.expanduser(args.raiz_host_dados).rstrip('/')

    stats = {}
    for cen, pasta in CENARIOS.items():
        itens = ler_tsv(os.path.join(args.listas, pasta, 'lista.tsv'))
        if args.limite:
            itens = itens[:args.limite]
        listas = {c: [] for c in CONDICOES}
        cortado, via_ffmpeg = [], 0
        for n, r in enumerate(itens, 1):
            if not r['caminho'].startswith('/dados/'):
                erro(f"caminho fora de /dados: {r['caminho']}")
            origem = host + r['caminho'][len('/dados'):]
            x, sr, via = ler(origem)
            via_ffmpeg += via
            for c, f in CONDICOES.items():
                rel = os.path.join(c, cen, f'{n:06d}.flac')
                destino = os.path.join(args.saida, rel)
                if not os.path.exists(destino):
                    y = f(x, sr)
                    os.makedirs(os.path.dirname(destino), exist_ok=True)
                    sf.write(destino + '.tmp.flac', y, sr, subtype='PCM_16')
                    os.replace(destino + '.tmp.flac', destino)
                    if c == 'aparar':
                        cortado.append((x.size - y.size) / sr)
                listas[c].append((r['id'], f"{args.raiz_container.rstrip('/')}/{rel.replace(os.sep, '/')}"))
            if n % 2000 == 0:
                print(f'  {cen}: {n}/{len(itens)}', flush=True)
        for c in CONDICOES:
            escrever_tsv(os.path.join(args.saida, f'{c}_{cen}.lista.tsv'), ['id', 'caminho'], listas[c])
        cortado = np.array(cortado) if cortado else np.zeros(1)
        stats[cen] = {'n': len(itens), 'lidos_via_ffmpeg': via_ffmpeg,
                      'segundos_cortados_mediana': float(np.median(cortado)),
                      'segundos_cortados_p90': float(np.percentile(cortado, 90))}
        print(f"{cen}: {len(itens)} áudios; corte mediano {stats[cen]['segundos_cortados_mediana']:.2f} s", flush=True)

    with open(os.path.join(args.saida, 'silencio_real.meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'condicoes': list(CONDICOES), 'limiar_db': LIMIAR_DB, 'silencio_padronizado_s': 0.5,
                   'cenarios': stats}, f, indent=2, ensure_ascii=False)
    resumo('f2_silencio_real.py', [(c, v) for c, v in stats.items()] + [('saída', args.saida)])


if __name__ == '__main__':
    main()
