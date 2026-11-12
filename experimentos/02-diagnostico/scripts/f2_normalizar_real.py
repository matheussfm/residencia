"""Fase 2c: intervenções no áudio de teste, em dados reais (In-the-Wild e BRSpeech-DF).

Mesma lógica da Fase 2b (f2_silencio_real.py): uma transformação é aplicada a TODOS os áudios,
nas duas classes, e os áudios são pontuados de novo. Se a EER muda, o fator transformado responde
por parte da perda nesse cenário. Uma condição por vez:

  - controle:    só reamostra para 16 kHz e regrava (mede o efeito do próprio caminho de E/S);
  - dereverb:    remove reverberação com WPE (pacote nara_wpe; STFT de 512 pontos, passo de 128,
                 taps = 10, delay = 3, 3 iterações), RMS restaurado. FORA DO PADRÃO: nos testes com
                 reverberação sintética, o WPE de um canal melhorou só 0,3–0,8 dB de SDR (mesmo
                 com 40 taps), então um resultado nulo não seria interpretável (PROCESSO);
  - equalizar:   iguala o espectro médio de longo prazo do áudio ao dos genuínos VCTK sem codec do
                 2021 DF (a "cor" do canal do treino), por um filtro FIR de fase linear; ganho por
                 faixa limitado a ±15 dB; RMS restaurado (só a forma do espectro muda);
  - volume:      leva o RMS do áudio a −18 dBFS (mediana dos genuínos do 2021 DF eval, README 4.10);
  - janela_meio: o trecho de 64.600 amostras do MEIO do áudio (áudios mais curtos ficam iguais);
  - janela_fim:  o trecho de 64.600 amostras do FIM do áudio;
  - padronizar:  silêncio das pontas cortado (máximo − 40 dB) e 0,5 s de silêncio digital no início
                 (a mesma condição da Fase 2b, aqui com o caminho de E/S do controle);
  - volume_padronizar: corta o silêncio, leva o RMS da fala a −18 dBFS e acrescenta 0,5 s de
                 silêncio no início (as duas intervenções que ajudaram no In-the-Wild, juntas).

Cenários (--cenarios): itw e brs (listas do Exp. 01) e df21 (amostra do 2021 DF da Fase 3: 3.000
trials do eval e 2.000 da hidden track, para verificar que a intervenção não piora o domínio de
treino; ids com o prefixo da Fase 3, ex.: df21_eval::DF_E_...).

Saídas em 16 kHz, FLAC 16 bits. Ids do Experimento 01, para usar os mesmos rotulos.tsv.

Roda FORA do container, no ambiente isolado (numpy, scipy, soundfile e, para dereverb, nara_wpe).
A referência espectral vem dos genuínos da amostra da Fase 2 (resultados/fase2/selecao.tsv), lidos
de --audio-df21.

Exemplo (no servidor):
    python -B f2_normalizar_real.py --raiz-host-dados ~/matheus/dados \
        --audio-df21 ~/matheus/dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac \
        --saida ~/matheus/dados/exp02_normalizar --raiz-container /dados/exp02_normalizar --processos 16
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))
sys.path.insert(0, AQUI)

import numpy as np
import soundfile as sf
from scipy import signal

from _comum import erro, escrever_tsv, garantir_fora_third_party, ler_tsv, resumo
from f2_silencio_real import CENARIOS, LIS01, aparar, ler

SR = 16000
JANELA = 64600
CONDICOES = ['controle', 'dereverb', 'equalizar', 'volume', 'janela_meio', 'janela_fim', 'padronizar', 'volume_padronizar']
PADRAO = ['controle', 'equalizar', 'volume', 'janela_meio', 'janela_fim']
NPERSEG = 512
REF = None  # espectro de referência (preenchido em cada processo)


def para_16k(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    from math import gcd
    g = gcd(SR, sr)
    return signal.resample_poly(x, SR // g, sr // g).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def espectro(x):
    f, p = signal.welch(x, fs=SR, nperseg=min(NPERSEG, x.size))
    if p.size != NPERSEG // 2 + 1:  # áudio mais curto que a janela de Welch
        f2 = np.linspace(0, SR / 2, NPERSEG // 2 + 1)
        p = np.interp(f2, f, p)
    return p / max(p.sum(), 1e-20)


def dereverb(x):
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe
    Y = stft(x[None].astype(np.float64), size=NPERSEG, shift=128).transpose(2, 0, 1)
    Z = wpe(Y, taps=10, delay=3, iterations=3, statistics_mode='full').transpose(1, 2, 0)
    z = istft(Z, size=NPERSEG, shift=128)[0][:x.size]
    if z.size < x.size:
        z = np.pad(z, (0, x.size - z.size))
    return (z / rms(z) * rms(x)).astype(np.float32)


def equalizar(x):
    ganho_db = np.clip(10 * np.log10((REF + 1e-12) / (espectro(x) + 1e-12)), -15, 15)
    freqs = np.linspace(0, 1, ganho_db.size)
    h = signal.firwin2(513, freqs, 10 ** (ganho_db / 20))
    y = signal.fftconvolve(x, h, mode='same')
    return np.clip(y / rms(y) * rms(x), -1, 1).astype(np.float32)


def transformar(cond, x):
    if cond == 'controle':
        return x
    if cond == 'dereverb':
        return dereverb(x)
    if cond == 'equalizar':
        return equalizar(x)
    if cond == 'volume':
        return np.clip(x * (10 ** (-18 / 20) / rms(x)), -1, 1).astype(np.float32)
    if cond in ('padronizar', 'volume_padronizar'):
        y = aparar(x, SR)
        if cond == 'volume_padronizar':
            y = np.clip(y * (10 ** (-18 / 20) / rms(y)), -1, 1)
        return np.concatenate([np.zeros(int(0.5 * SR), dtype=np.float32), y.astype(np.float32)])
    if x.size <= JANELA:
        return x
    if cond == 'janela_meio':
        i = (x.size - JANELA) // 2
        return x[i:i + JANELA]
    if cond == 'janela_fim':
        return x[-JANELA:]
    erro(f'condição desconhecida: {cond}')


def _init(ref):
    global REF
    REF = ref


def processar(tarefa):
    origem, destinos = tarefa
    pendentes = [(c, d) for c, d in destinos if not os.path.exists(d)]
    if not pendentes:
        return 0
    x, sr, via = ler(origem)
    x = para_16k(x, sr)
    for c, d in pendentes:
        y = transformar(c, x)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        sf.write(d + '.tmp.flac', y, SR, subtype='PCM_16')
        os.replace(d + '.tmp.flac', d)
    return int(via)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz-host-dados', required=True)
    ap.add_argument('--audio-df21', required=True, help='pasta com os .flac do 2021 DF eval (referência espectral)')
    ap.add_argument('--saida', required=True)
    ap.add_argument('--raiz-container', required=True)
    ap.add_argument('--condicoes', nargs='+', default=PADRAO, choices=CONDICOES)
    ap.add_argument('--processos', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--listas', default=LIS01)
    ap.add_argument('--cenarios', nargs='+', default=['itw', 'brs'], choices=['itw', 'brs', 'df21'])
    ap.add_argument('--limite', type=int, default=None)
    ap.add_argument('--selecao', default=os.path.join(AQUI, '..', 'resultados', 'fase2', 'selecao.tsv'),
                    help='seleção da Fase 2 (os genuínos dão a referência espectral)')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)
    host = os.path.expanduser(args.raiz_host_dados).rstrip('/')

    # referência espectral: genuínos VCTK sem codec da amostra da Fase 2
    sel = ler_tsv(args.selecao)
    trials = [r['trial'] for r in sel if r['rotulo'] == 'bonafide']
    esp = []
    for t in trials:
        x, sr, _ = ler(os.path.join(os.path.expanduser(args.audio_df21), t + '.flac'))
        esp.append(espectro(para_16k(x, sr)))
    ref = np.mean(esp, axis=0)
    os.makedirs(args.saida, exist_ok=True)
    np.save(os.path.join(args.saida, 'espectro_referencia.npy'), ref)
    print(f'referência espectral: {len(trials)} genuínos VCTK sem codec')

    stats = {}
    fontes = dict(CENARIOS, df21=None)
    for cen in args.cenarios:
        if cen == 'df21':  # amostra da Fase 3, só os trials do 2021 DF
            itens = [r for r in ler_tsv(os.path.join(AQUI, '..', 'listas', 'fase3', 'lista.tsv'))
                     if r['id'].startswith(('df21_eval::', 'df21_hidden::'))]
        else:
            itens = ler_tsv(os.path.join(args.listas, fontes[cen], 'lista.tsv'))
        if args.limite:
            itens = itens[:args.limite]
        tarefas, listas = [], {c: [] for c in args.condicoes}
        for n, r in enumerate(itens, 1):
            if not r['caminho'].startswith('/dados/'):
                erro(f"caminho fora de /dados: {r['caminho']}")
            destinos = []
            for c in args.condicoes:
                rel = f'{c}/{cen}/{n:06d}.flac'
                destinos.append((c, os.path.join(args.saida, rel)))
                listas[c].append((r['id'], f"{args.raiz_container.rstrip('/')}/{rel}"))
            tarefas.append((host + r['caminho'][len('/dados'):], destinos))
        via = 0
        with Pool(args.processos, initializer=_init, initargs=(ref,)) as pool:
            for k, v in enumerate(pool.imap(processar, tarefas, chunksize=8), 1):
                via += v
                if k % 2000 == 0:
                    print(f'  {cen}: {k}/{len(tarefas)}', flush=True)
        for c in args.condicoes:
            escrever_tsv(os.path.join(args.saida, f'{c}_{cen}.lista.tsv'), ['id', 'caminho'], listas[c])
        stats[cen] = {'n': len(itens), 'lidos_via_ffmpeg': via}
        print(f'{cen}: {len(itens)} áudios', flush=True)

    with open(os.path.join(args.saida, 'normalizar_real.meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'condicoes': args.condicoes, 'sr': SR, 'janela': JANELA,
                   'wpe': {'nperseg': NPERSEG, 'shift': 128, 'taps': 10, 'delay': 3, 'iteracoes': 3},
                   'equalizar': {'numtaps': 513, 'ganho_max_db': 15, 'referencia': f'{len(trials)} genuínos VCTK sem codec'},
                   'volume_dbfs': -18, 'cenarios': stats}, f, indent=2, ensure_ascii=False)
    resumo('f2_normalizar_real.py', [(c, v) for c, v in stats.items()] + [('saída', args.saida)])


if __name__ == '__main__':
    main()
