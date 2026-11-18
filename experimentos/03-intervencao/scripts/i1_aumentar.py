"""Etapa 2 (I1): cópias aumentadas da partição de treino do df21_eval, nas duas classes.

Para cada trial da partição de treino (listas/divisao_df21_eval.tsv) e cada variante, grava uma
cópia aumentada, com semente própria por (áudio, variante) e a condição sorteada registrada:

  controle  só lê, converte para 16 kHz e regrava, sem aumento. É o controle do caminho de leitura
            e gravação: o probe "limpo + controle" separa o efeito do aumento do efeito de duplicar
            o treino e de passar pelo mesmo caminho de E/S das cópias aumentadas;
  a         +reverberação: RT60 sorteado em [0,2; 1,0] s (resposta ao impulso sintética do
            f2_perturbar.py, RMS restaurado);
  b         +silêncio/volume: corta o silêncio das pontas (máximo − 40 dB) ou acrescenta de 0,25 a
            2 s de silêncio digital no início (50% cada); depois, ganho sorteado em [−20; +6] dB;
  c         +canal: banda de telefone (300–3.400 Hz, RMS restaurado) ou MP3 a 16 kbps (50% cada);
  d         tudo: cada um de (a), (b) e (c) entra com probabilidade 1/2, sorteados de forma
            independente (se nenhum for sorteado, um deles é escolhido ao acaso), na ordem
            sala → canal → silêncio/volume.

As funções de perturbação são as do f2_perturbar.py do Exp. 02 (importadas, não duplicadas); a
leitura é a do f2_silencio_real.py (soundfile, com o ffmpeg como alternativa).

Saídas em --saida: <variante>/<trial>.wav (16 kHz, float32), lista.tsv (id, caminho visto do
container), condicoes.tsv (id, variante, parâmetros sorteados em JSON) e i1_aumentar.meta.json.
O id de cada cópia é "i1_<variante>::<trial>".

Roda FORA do container, no ambiente isolado (numpy, scipy, soundfile e ffmpeg no PATH), porque
grava na pasta de dados, que o container monta somente leitura.

Exemplo (no servidor, a partir de experimentos/03-intervencao/scripts):
    python -B i1_aumentar.py --raiz-host-dados ~/matheus/dados \
        --saida ~/matheus/dados/exp03_i1 --raiz-container /dados/exp03_i1 --processos 16
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
EXP02_SCRIPTS = os.path.abspath(os.path.join(AQUI, '..', '..', '02-diagnostico', 'scripts'))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))
sys.path.insert(0, EXP02_SCRIPTS)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import f2_perturbar as F2  # noqa: E402  (perturbações da Fase 2)
from f2_silencio_real import ler  # noqa: E402
from _comum import erro, escrever_tsv, garantir_fora_third_party, ler_tsv, resumo, sha256  # noqa: E402

SR = F2.SR
VARIANTES = ['controle', 'a', 'b', 'c', 'd']
DIVISAO = os.path.join(AQUI, '..', 'listas', 'divisao_df21_eval.tsv')
LISTA_F3 = os.path.join(EXP02_SCRIPTS, '..', 'listas', 'fase3', 'lista.tsv')
MP3_16K = F2.codec(['-c:a', 'libmp3lame', '-b:a', '16k'], '.mp3')
TELEFONE = F2.filtro('bandpass', [300, 3400])


def para_16k(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    from math import gcd
    from scipy import signal
    g = gcd(SR, sr)
    return signal.resample_poly(x, SR // g, sr // g).astype(np.float32)


def reverb(x, rng):
    rt60 = float(rng.uniform(0.2, 1.0))
    return F2.reverberacao(rt60)(x, rng), {'rt60_s': round(rt60, 4)}


def silencio_volume(x, rng):
    p = {}
    if rng.random() < 0.5:
        y = F2.aparar(x, 40)
        p['silencio'] = 'cortar_40db'
    else:
        seg = float(rng.uniform(0.25, 2.0))
        y = F2.silencio_inicio(seg)(x, rng)
        p['silencio'] = f'acrescentar_{seg:.3f}s'
    g = float(rng.uniform(-20, 6))
    p['ganho_db'] = round(g, 3)
    return np.clip(y * 10 ** (g / 20), -1, 1).astype(np.float32), p


def canal(x, rng):
    if rng.random() < 0.5:
        return TELEFONE(x, rng), {'canal': 'telefone'}
    return MP3_16K(x, rng), {'canal': 'mp3_16k'}


def aumentar(variante, x, rng):
    """Devolve (y, parâmetros sorteados)."""
    if variante == 'controle':
        return x, {}
    if variante == 'a':
        return reverb(x, rng)
    if variante == 'b':
        return silencio_volume(x, rng)
    if variante == 'c':
        return canal(x, rng)
    if variante == 'd':
        usar = rng.random(3) < 0.5
        if not usar.any():
            usar[rng.integers(3)] = True
        y, p = x, {'fatores': [n for n, u in zip(('reverberacao', 'canal', 'silencio_volume'), usar) if u]}
        for u, f in zip(usar, (reverb, canal, silencio_volume)):
            if u:
                y, q = f(y, rng)
                p.update(q)
        return y, p
    erro(f'variante desconhecida: {variante}')


def processar(tarefa):
    """tarefa: (n, trial, caminho_host, [(k, variante, destino)], semente). Devolve linhas de condição."""
    n, trial, origem, destinos, semente = tarefa
    x, sr, via = ler(origem)
    x = para_16k(x, sr)
    linhas = []
    for k, v, destino in destinos:
        rng = np.random.default_rng([semente, n, k])
        y, p = aumentar(v, x, rng)
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0 or not np.all(np.isfinite(y)):
            erro(f'{v}/{trial}: saída vazia ou não finita')
        if not os.path.exists(destino):
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            sf.write(destino + '.tmp.wav', y, SR, subtype='FLOAT')
            os.replace(destino + '.tmp.wav', destino)
        linhas.append((f'i1_{v}::{trial}', v, json.dumps(p, ensure_ascii=False, sort_keys=True), f'{y.size / SR:.3f}'))
    return linhas, int(via)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz-host-dados', required=True, help='pasta do host montada como /dados no container')
    ap.add_argument('--saida', required=True)
    ap.add_argument('--raiz-container', required=True, help='--saida vista de dentro do container')
    ap.add_argument('--lista-fase3', default=LISTA_F3, help='lista.tsv da Fase 3 (caminhos vistos do container)')
    ap.add_argument('--divisao', default=DIVISAO)
    ap.add_argument('--variantes', nargs='+', default=VARIANTES, choices=VARIANTES)
    ap.add_argument('--particao', default='treino')
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--processos', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--limite', type=int, default=None, help='só os N primeiros trials (teste rápido)')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)
    if any(v in ('c', 'd') for v in args.variantes) and shutil.which('ffmpeg') is None:
        erro('as variantes c e d precisam do ffmpeg no PATH (MP3 a 16 kbps)')
    host = os.path.expanduser(args.raiz_host_dados).rstrip('/')

    caminhos = {r['id']: r['caminho'] for r in ler_tsv(args.lista_fase3)}
    ids = sorted(r['id'] for r in ler_tsv(args.divisao) if r['particao'] == args.particao)
    if args.limite:
        ids = ids[:args.limite]
    tarefas, lista = [], []
    for n, i in enumerate(ids):
        c = caminhos.get(i)
        if c is None or not c.startswith('/dados/'):
            erro(f'{i}: sem caminho em /dados na lista da Fase 3 ({c})')
        trial = i.split('::', 1)[1]
        destinos = []
        for v in args.variantes:
            k = VARIANTES.index(v)  # semente independe de quais variantes foram pedidas
            destinos.append((k, v, os.path.join(args.saida, v, trial + '.wav')))
            lista.append((f'i1_{v}::{trial}', f"{args.raiz_container.rstrip('/')}/{v}/{trial}.wav"))
        tarefas.append((n, trial, host + c[len('/dados'):], destinos, args.semente))

    condicoes, via = [], 0
    with Pool(args.processos) as pool:
        for k, (linhas, v) in enumerate(pool.imap(processar, tarefas, chunksize=4), 1):
            condicoes += linhas
            via += v
            if k % 200 == 0:
                print(f'  {k}/{len(tarefas)} trials', flush=True)

    os.makedirs(args.saida, exist_ok=True)
    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'condicoes.tsv'), ['id', 'variante', 'parametros', 'duracao_s'], condicoes)
    ffmpeg_v = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0] \
        if shutil.which('ffmpeg') else None
    meta = {'variantes': args.variantes, 'particao': args.particao, 'n_trials': len(ids), 'n_arquivos': len(lista),
            'semente': args.semente, 'divisao_sha256': sha256(args.divisao), 'lista_fase3_sha256': sha256(args.lista_fase3),
            'condicoes_sha256': sha256(os.path.join(args.saida, 'condicoes.tsv')), 'lidos_via_ffmpeg': via,
            'parametros': {'a': 'RT60 ~ U(0,2; 1,0) s', 'b': 'cortar 40 dB ou +U(0,25; 2,0) s de silêncio (1/2 cada); ganho ~ U(-20; +6) dB',
                           'c': 'telefone 300-3400 Hz ou MP3 16 kbps (1/2 cada)',
                           'd': 'cada fator com p = 1/2, ao menos um; ordem sala, canal, silêncio/volume'},
            'ffmpeg': ffmpeg_v, 'numpy': np.__version__, 'soundfile': sf.__version__, 'python': platform.python_version()}
    with open(os.path.join(args.saida, 'i1_aumentar.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    cont = {}
    for _, v, p, _ in condicoes:
        cont.setdefault(v, {})
        for x in json.loads(p).get('fatores', []) + ([json.loads(p)['canal']] if 'canal' in json.loads(p) else []):
            cont[v][x] = cont[v].get(x, 0) + 1
    resumo('i1_aumentar.py', [('trials', len(ids)), ('arquivos', len(lista)), ('lidos via ffmpeg', via),
                              ('contagens', cont), ('saída', args.saida)])


if __name__ == '__main__':
    main()
