"""Fase 2: figura do efeito de cada perturbação (a partir do perturbacoes.json do f2_analisar.py).

Painel esquerdo: deslocamento mediano da pontuação de cada áudio em relação a ele mesmo na
condição original (genuínos em azul, falsos em laranja), com IC de 95% por bootstrap.
Painel direito: EER com só os genuínos perturbados (contra os falsos originais) e com as duas
classes perturbadas. As condições ficam agrupadas por fator.

Exemplo:
    python f2_figura.py --resultado ../resultados/fase2/perturbacoes.json \
        --meta ../resultados/fase2/perturbacoes.meta.json
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F

ORDEM_FATORES = ['controle', 'reverberacao', 'silencio', 'codec', 'banda', 'volume', 'reamostragem', 'ruido']
NOMES_FATORES = {'controle': 'controle', 'reverberacao': 'reverberação', 'silencio': 'silêncio', 'codec': 'codec',
                 'banda': 'banda', 'volume': 'volume', 'reamostragem': 'reamostragem', 'ruido': 'ruído'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resultado', required=True)
    ap.add_argument('--meta', required=True, help='perturbacoes.meta.json do f2_perturbar.py (fator de cada condição)')
    ap.add_argument('--saida', default=None)
    args = ap.parse_args()
    res = json.load(open(args.resultado, encoding='utf-8'))['condicoes']
    fator = {c['nome']: c['fator'] for c in json.load(open(args.meta, encoding='utf-8'))['condicoes']}
    conds = sorted(res, key=lambda c: (ORDEM_FATORES.index(fator[c]), c))
    y = np.arange(len(conds))[::-1]

    fig, axs = F.plt.subplots(1, 2, figsize=(11, 8.5), sharey=True)
    ax = axs[0]
    for k, cor, desl, rot in (('delta_bonafide', F.COR_GEN, 0.15, 'genuínos'), ('delta_spoof', F.COR_FAL, -0.15, 'falsos')):
        v = np.array([res[c][k]['mediana'] for c in conds])
        lo = np.array([res[c][k]['ic95_mediana'][0] for c in conds])
        hi = np.array([res[c][k]['ic95_mediana'][1] for c in conds])
        ax.errorbar(v, y + desl, xerr=[v - lo, hi - v], fmt='o', color=cor, ecolor=F.EIXO, markersize=5, label=rot)
    ax.axvline(0, color=F.TINTA2, linewidth=1)
    ax.set_xlabel('deslocamento mediano da pontuação (perturbado − original)')
    ax.set_title('Quanto a pontuação de cada áudio muda', loc='left')
    ax.legend(loc='lower left')
    ax = axs[1]
    for k, cor, desl, rot in (('eer_gen_perturbado_vs_fal_original', F.AZUL, 0.15, 'só genuínos perturbados'),
                              ('eer_ambos_perturbados', F.AQUA, -0.15, 'genuínos e falsos perturbados')):
        ax.plot([100 * res[c][k] for c in conds], y + desl, 'o', color=cor, markersize=6, label=rot)
    ax.axvline(100 * res['original']['eer_ambos_perturbados'], color=F.TINTA2, linewidth=1, linestyle='--')
    ax.set_xlabel('EER (%)')
    ax.set_title('EER em cada condição (controle: tracejado)', loc='left')
    ax.legend(loc='lower right')
    axs[0].set_yticks(y, [c.replace('_', ' ') for c in conds], fontsize=8)
    # separadores e rótulos dos fatores
    for ax in axs:
        for i in range(1, len(conds)):
            if fator[conds[i]] != fator[conds[i - 1]]:
                ax.axhline(y[i] + 0.5, color=F.GRADE, linewidth=1)
    for f in ORDEM_FATORES:
        ys = [y[i] for i, c in enumerate(conds) if fator[c] == f]
        if ys:
            axs[1].annotate(NOMES_FATORES[f], xy=(1.01, np.mean(ys)), xycoords=('axes fraction', 'data'),
                            va='center', fontsize=8, color=F.TINTA2)
    saida = args.saida or os.path.join(D.pasta_saida('fase2', 'figuras'), 'perturbacoes.png')
    F.salvar(fig, saida)
    print('figura:', saida)


if __name__ == '__main__':
    main()
