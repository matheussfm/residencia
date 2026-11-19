"""Fechamento: figuras do Experimento 03, a partir dos JSON das Etapas 1 a 4 (sem recalcular nada).

  metodos.png  EER [IC 95% por locutor] de cada linha de base e da I1 em cada cenário, com o
               detector (B0) como referência tracejada. Um painel por cenário.
  i2.png       EER mediana [p10, p90] por k na I2, nas camadas 12 (a escolhida pela origem) e 5,
               com os locutores sorteados (Etapa 3) e com a adaptação na validação (Etapa 4). Em k = 0,
               os probes de origem sem adaptação; o tracejado é o detector nos mesmos subconjuntos.

Estilo do Exp. 02 (_figuras.py). Saídas em resultados/figuras/.
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
import _figuras as F  # noqa: E402  (Exp. 02)

RES = os.path.join(I.EXP03, 'resultados')
CENARIOS = [('itw', 'In-the-Wild'), ('brs', 'BRSpeech-DF\n(teste oficial)'), ('df21_hidden', 'Hidden track\n(completa)'),
            ('df21_hidden_teste', 'Hidden track\n(só locutores de teste)'), ('df21_teste', 'df21 teste\n(domínio de origem)')]
METODOS = [('B0', 'B0 detector'), ('B1', 'B1 detector + normalização'), ('B2', 'B2 probe limpo'),
           ('I1_d', 'I1 (d) probe com aumento'), ('I1_d+B1', 'I1 (d) + normalização')]


def ler(*p):
    return json.load(open(os.path.join(RES, *p), encoding='utf-8'))


def fig_metodos(i1):
    fig, axs = F.plt.subplots(1, len(CENARIOS), figsize=(13, 3.7), sharey=True)
    y = np.arange(len(METODOS))[::-1]
    for ax, (c, nome) in zip(axs, CENARIOS):
        m = i1['metricas'][c]
        v = np.array([100 * m[k]['eer']['valor'] for k, _ in METODOS])
        lo = np.array([100 * m[k]['eer']['ic95'][0] for k, _ in METODOS])
        hi = np.array([100 * m[k]['eer']['ic95'][1] for k, _ in METODOS])
        ax.hlines(y, lo, hi, color=F.MUDO, linewidth=1.2)
        ax.plot(v, y, 'o', color=F.AZUL, markersize=7, markeredgecolor=F.FUNDO, markeredgewidth=1.5, zorder=3)
        ax.axvline(v[0], color=F.TINTA2, linestyle='--', linewidth=1)
        for yy, vv in zip(y, v):
            ax.annotate(f'{vv:.1f}', (vv, yy), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=7.5, color=F.TINTA2)
        ax.set_title(nome, loc='left', fontsize=9)
        ax.set_xlim(0, max(hi.max() * 1.08, 5))
        ax.set_xlabel('EER (%)')
        ax.grid(axis='y', visible=False)
        ax.set_ylim(-0.6, len(METODOS) - 0.4)
    axs[0].set_yticks(y, [n for _, n in METODOS])
    fig.suptitle('Linhas de base e I1: EER com IC de 95% por locutor (tracejado: o detector, B0)', x=0.01,
                 ha='left', fontsize=10)
    F.salvar(fig, os.path.join(RES, 'figuras', 'metodos.png'))


def fig_i2(e3, e4):
    ks = [0, 10, 50, 200]
    x = np.arange(len(ks))
    series = [('A_limpo', 'A limpo (origem + alvo)', F.AZUL, 'B2', -0.1), ('A_d', 'A (d) (origem aumentada + alvo)', F.LARANJA, 'I1_d', 0.0),
              ('B_alvo', 'B (só alvo)', F.AQUA, None, 0.1)]
    fig, axs = F.plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    for i, (j, ref_k0, resumo, titulo) in enumerate((
            (e3, e3['referencias_k0'], e3['resumo'], 'Etapa 3: locutores e TTS sorteados (teste com 5 locutores)'),
            (e4['i2_validacao'], e4['i2_validacao']['referencias_k0'], e4['i2_validacao']['resumo'],
             'Etapa 4: adaptação na validação, teste com os 10 locutores do teste'))):
        for col, rep in enumerate(('camada_12', 'camada_5')):
            ax = axs[i, col]
            b0 = 100 * ref_k0['B0']['eer']['mediana']
            ax.axhline(b0, color=F.TINTA2, linestyle='--', linewidth=1)
            ax.annotate(f'detector {b0:.1f}%', (x[-1], b0), xytext=(0, 4), textcoords='offset points', ha='right',
                        fontsize=7.5, color=F.TINTA2)
            for nome, rotulo, cor, k0, dx in series:
                med, p10, p90, xx = [], [], [], []
                if k0:
                    r0 = ref_k0[f'{k0}@{rep}']['eer']
                    med.append(r0['mediana']), p10.append(r0['p10']), p90.append(r0['p90']), xx.append(0)
                for n, k in enumerate(ks[1:], 1):
                    r = resumo[str(k)]['metodos'][f'{nome}@{rep}']['eer']
                    med.append(r['mediana']), p10.append(r['p10']), p90.append(r['p90']), xx.append(n)
                med, p10, p90 = 100 * np.array(med), 100 * np.array(p10), 100 * np.array(p90)
                xx = np.array(xx) + dx
                ax.vlines(xx, p10, p90, color=cor, linewidth=1.2, alpha=0.6)
                ax.plot(xx, med, '-o', color=cor, markersize=6, markeredgecolor=F.FUNDO, markeredgewidth=1.5,
                        label=rotulo)
            ax.set_title(f"{'Camada 12 (escolhida pela origem)' if rep == 'camada_12' else 'Camada 5'}", loc='left', fontsize=9)
            ax.set_ylim(0, 60)
            if col == 0:
                ax.set_ylabel(f'EER mediana (%)\n{titulo.split(":")[0]}')
        axs[i, 0].annotate(titulo, (0, 1.13), xycoords='axes fraction', fontsize=8.5, color=F.TINTA2)
    for ax in axs[-1]:
        ax.set_xticks(x, [str(k) for k in ks])
        ax.set_xlabel('exemplos do português por classe (k)')
    axs[0, 1].legend(loc='upper right', fontsize=7.5)
    fig.suptitle('I2: poucos exemplos em português, com locutores e TTS fora da adaptação (barras: p10–p90 dos 20 sorteios)',
                 x=0.01, ha='left', fontsize=10)
    F.salvar(fig, os.path.join(RES, 'figuras', 'i2.png'))


def main():
    os.makedirs(os.path.join(RES, 'figuras'), exist_ok=True)
    fig_metodos(ler('etapa2', 'i1.json'))
    fig_i2(ler('etapa3', 'i2.json'), ler('etapa4', 'brs_val.json'))
    print('ok:', os.path.join(RES, 'figuras'))


if __name__ == '__main__':
    main()
