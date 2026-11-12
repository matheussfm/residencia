"""Análise das medidas acústicas (f1_qualidade_audio.py) contra rótulos e pontuações do detector.

1. Mediana de cada medida por cenário e classe: os genuínos fora do domínio diferem dos genuínos
   do 2021 DF nos fatores que a Fase 2 mostrou causais (silêncio, graves, nível)?
2. Atalho do silêncio em dados reais: EER usando só a duração do silêncio inicial como pontuação
   (mais silêncio = genuíno), em cada cenário.
3. Dentro dos genuínos de cada cenário: correlação de Spearman entre cada medida e a pontuação do
   detector (a pontuação baixa acompanha pouco silêncio, poucos graves, nível baixo?).
4. BRSpeech-DF: genuínos dos 3 locutores "ruins" contra os dos 7 "bons".

Saída: resultados/qualidade/analise.json.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import spearmanr

import _diag as D

MEDIDAS = ['silencio_inicio_s', 'silencio_fim_s', 'snr_est_db', 'nivel_dbfs', 'banda_99_hz', 'frac_graves_300hz',
           'frac_clipping', 'duracao_s']
RUINS = {'5739', '13063', '12626'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--medidas', required=True)
    ap.add_argument('--amostra', required=True)
    args = ap.parse_args()
    med = {r['id']: r for r in D.ler_tsv(D._usar(args.medidas))}
    am = [r for r in D.ler_tsv(D._usar(args.amostra)) if r['id'] in med]
    cen = np.array([r['cenario'] for r in am])
    y = np.array([r['rotulo'] == 'bonafide' for r in am])
    loc = np.array([r['locutor'] for r in am])
    p = np.array([float(r['pontuacao_exp01']) for r in am])
    M = {k: np.array([float(med[r['id']][k]) for r in am]) for k in MEDIDAS}
    res = {'n': len(am), 'medianas': {}, 'atalho_silencio_inicial': {}, 'spearman_genuinos': {}, 'brs_locutores': {}}

    print(f"{'cenário/classe':24s}" + ''.join(f'{k[:14]:>15s}' for k in MEDIDAS))
    for c in ('df21_eval', 'df21_hidden', 'itw', 'brs'):
        for nome, m in (('genuínos', y), ('falsos', ~y)):
            mm = (cen == c) & m
            d = {k: float(np.median(M[k][mm])) for k in MEDIDAS}
            res['medianas'][f'{c}/{nome}'] = d
            print(f'{c + "/" + nome:24s}' + ''.join(f'{d[k]:15.3g}' for k in MEDIDAS))
        mc = cen == c
        sb, ss = M['silencio_inicio_s'][mc & y], M['silencio_inicio_s'][mc & ~y]
        res['atalho_silencio_inicial'][c] = {'eer_so_silencio_inicial': D.eer(sb, ss)[0], 'auc': D.auc(sb, ss)}
        res['spearman_genuinos'][c] = {k: float(spearmanr(M[k][mc & y], p[mc & y]).correlation) for k in MEDIDAS}
    print('\nAtalho do silêncio inicial (EER usando só o silêncio inicial; 50% = não separa):')
    for c, d in res['atalho_silencio_inicial'].items():
        print(f"  {c:12s} EER {D.pct(d['eer_so_silencio_inicial'])}  AUC {d['auc']:.3f}")
    print('\nSpearman (medida × pontuação do detector), só genuínos:')
    for c, d in res['spearman_genuinos'].items():
        print(f'  {c:12s} ' + ' '.join(f'{k[:12]}={v:+.2f}' for k, v in d.items()))

    mb = (cen == 'brs') & y
    ruim = np.isin(loc, list(RUINS))
    for nome, m in (('ruins', mb & ruim), ('bons', mb & ~ruim)):
        res['brs_locutores'][nome] = {k: float(np.median(M[k][m])) for k in MEDIDAS}
        res['brs_locutores'][nome]['pontuacao_mediana'] = float(np.median(p[m]))
        res['brs_locutores'][nome]['n'] = int(m.sum())
    print('\nBRSpeech-DF, genuínos dos locutores ruins × bons (medianas):')
    for nome, d in res['brs_locutores'].items():
        print(f'  {nome:6s} n={d["n"]:4d} pont {d["pontuacao_mediana"]:+.2f} ' +
              ' '.join(f'{k[:12]}={d[k]:.3g}' for k in MEDIDAS))
    por_loc = {}
    for l in sorted(np.unique(loc[mb])):
        m = mb & (loc == l)
        por_loc[l] = {k: float(np.median(M[k][m])) for k in MEDIDAS}
        por_loc[l]['pontuacao_mediana'] = float(np.median(p[m]))
    res['brs_por_locutor'] = por_loc
    print('\nBRSpeech-DF por locutor (genuínos):')
    for l, d in sorted(por_loc.items(), key=lambda kv: kv[1]['pontuacao_mediana']):
        print(f"  {l:6s} {'RUIM' if l in RUINS else '    '} pont {d['pontuacao_mediana']:+.2f} sil_ini {d['silencio_inicio_s']:.2f}"
              f" snr {d['snr_est_db']:.1f} nível {d['nivel_dbfs']:.1f} banda {d['banda_99_hz']:.0f} graves {d['frac_graves_300hz']:.3f}")

    D.salvar_json(os.path.join(D.pasta_saida('qualidade'), 'analise.json'), res, __file__, {'limiar_silencio_db': 40})


if __name__ == '__main__':
    main()
