"""Fase 3, teste de assinatura: o deslocamento observado nos dados reais se parece com qual fator?

Cada perturbação da Fase 2 desloca a representação dos genuínos numa direção própria (a
"assinatura" do fator): s_f = média(genuínos sob o fator f) − média(genuínos originais), nos
mesmos áudios VCTK. O deslocamento observado nos dados reais é t = média(genuínos do cenário) −
média(genuínos de referência). O cosseno entre t e s_f, camada a camada, diz com que fator o
deslocamento real mais se parece.

Deslocamentos observados (alvos):
  - hidden:      genuínos da hidden track − genuínos VCTK do 2021 DF eval (CONTROLE POSITIVO: deve
                 se parecer com a assinatura de "aparar_40db", porque a hidden track é o próprio
                 corte de silêncio);
  - itw:         genuínos do In-the-Wild − genuínos VCTK do 2021 DF eval;
  - brs:         genuínos do BRSpeech-DF − genuínos VCTK do 2021 DF eval;
  - brs_ruins:   genuínos dos 3 locutores "ruins" − genuínos dos 7 "bons" do BRSpeech-DF.

Espaço: padronizado pelas features do 2021 DF eval da Fase 3 (mesmo modelo e mesmo pré-processamento
nas duas extrações).

Entradas: features da Fase 3 (features.npz + amostra.tsv) e features das condições da Fase 2
(features_assinaturas.npz, extraídas com f3_extrair.py sobre a lista da Fase 2 filtrada), com a
selecao.tsv da Fase 2 (rótulos).
Saídas: resultados/fase3/assinaturas.json e figuras/assinaturas.png.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F
import f3_probing as P

FATORES = ['reverb_0.6s', 'aparar_40db', 'mp3_16k', 'banda_telefone', 'ganho_-20db', 'silencio_1.0s', 'ruido_branco_10db']
RUINS = {'5739', '13063', '12626'}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True, help='features.npz da Fase 3')
    ap.add_argument('--amostra', required=True)
    ap.add_argument('--features-f2', required=True, help='features das condições da Fase 2')
    ap.add_argument('--selecao', required=True, help='selecao.tsv da Fase 2')
    args = ap.parse_args()
    from sklearn.preprocessing import StandardScaler

    z = np.load(D._usar(args.features))
    am = {r['id']: r for r in D.ler_tsv(D._usar(args.amostra))}
    ids = [str(i) for i in z['ids']]
    cen = np.array([am[i]['cenario'] for i in ids])
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids])
    loc = np.array([am[i]['locutor'] for i in ids])
    orig = np.array([am[i]['grupo'] for i in ids])
    cam3 = P.memmap_npz(args.features, 'camadas')

    z2 = np.load(D._usar(args.features_f2))
    sel = {r['trial']: r['rotulo'] == 'bonafide' for r in D.ler_tsv(D._usar(args.selecao))}
    ids2 = [str(i) for i in z2['ids']]
    cond2 = np.array([i.split('__', 1)[0] for i in ids2])
    gen2 = np.array([sel[i.split('__', 1)[1]] for i in ids2])
    cam2 = z2['camadas']
    faltam = [f for f in FATORES + ['original'] if f not in set(cond2)]
    if faltam:
        D.erro(f'condições ausentes nas features da Fase 2: {faltam}')

    m_ref = (cen == 'df21_eval') & y & (orig == 'asvspoof')
    alvos = {
        'hidden': ((cen == 'df21_hidden') & y, m_ref),
        'itw': ((cen == 'itw') & y, m_ref),
        'brs': ((cen == 'brs') & y, m_ref),
        'brs_ruins': ((cen == 'brs') & y & np.isin(loc, list(RUINS)), (cen == 'brs') & y & ~np.isin(loc, list(RUINS))),
    }
    nomes = [f'camada_{i + 1}' for i in range(cam3.shape[1])] + ['final']
    res = {n: {} for n in nomes}
    for k, nome in enumerate(nomes):
        if nome == 'final':
            X3, X2 = z['final'].astype(np.float32), z2['final'].astype(np.float32)
        else:
            X3 = np.ascontiguousarray(cam3[:, k, :], dtype=np.float32)
            X2 = cam2[:, k, :].astype(np.float32)
        sc = StandardScaler().fit(X3[cen == 'df21_eval'])
        X3, X2 = sc.transform(X3), sc.transform(X2)
        base = X2[(cond2 == 'original') & gen2].mean(0)
        assin = {f: X2[(cond2 == f) & gen2].mean(0) - base for f in FATORES}
        for a, (m_alvo, m_base) in alvos.items():
            t = X3[m_alvo].mean(0) - X3[m_base].mean(0)
            res[nome][a] = {f: cos(t, s) for f, s in assin.items()}
        del X3, X2
    D.salvar_json(os.path.join(D.pasta_saida('fase3'), 'assinaturas.json'), res, __file__,
                  {'fatores': FATORES, 'referencia': 'genuínos VCTK do 2021 DF eval (amostra da Fase 3)'})

    for nome in ('camada_3', 'camada_5', 'camada_12', 'camada_24', 'final'):
        print(f'\n{nome}: cosseno entre o deslocamento observado e a assinatura de cada fator')
        print(f"{'alvo':10s}" + ''.join(f'{f[:14]:>16s}' for f in FATORES))
        for a in alvos:
            print(f'{a:10s}' + ''.join(f'{res[nome][a][f]:16.2f}' for f in FATORES))

    fig, axs = F.plt.subplots(1, len(alvos), figsize=(14, 3.6), sharey=True)
    x = np.arange(len(nomes))
    titulos = {'hidden': 'hidden track (controle +)', 'itw': 'In-the-Wild', 'brs': 'BRSpeech-DF',
               'brs_ruins': 'BRS: 3 ruins − 7 bons'}
    for ax, a in zip(axs, alvos):
        for cor, f in zip(F.CORES, FATORES):
            ax.plot(x, [res[n][a][f] for n in nomes], '-', color=cor, label=f.replace('_', ' '), linewidth=1.6)
        ax.axhline(0, color=F.TINTA2, linewidth=1)
        ax.set_title(titulos[a], loc='left')
        ax.set_xticks(list(range(0, 24, 4)) + [24], [str(i + 1) for i in range(0, 24, 4)] + ['fin.'], fontsize=7)
        ax.set_xlabel('camada')
    axs[0].set_ylabel('cosseno (deslocamento × assinatura)')
    axs[-1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
    F.salvar(fig, os.path.join(D.pasta_saida('fase3', 'figuras'), 'assinaturas.png'))


if __name__ == '__main__':
    main()
