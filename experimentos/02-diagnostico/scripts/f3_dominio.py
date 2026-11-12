"""Fase 3: em que camada a diferença de domínio aparece, e para que lado ela empurra?

Usa as features da Fase 3 (f3_extrair.py). Para cada representação (24 camadas do XLS-R, saída
final e embedding do AASIST):

1. TAMANHO DA DIFERENÇA DE DOMÍNIO: AUC de um classificador linear que distingue os genuínos do
   2021 DF eval dos genuínos de outro cenário (hidden track, In-the-Wild, BRSpeech-DF), com
   validação cruzada por locutor (3 partes). O mesmo para os falsos. AUC = 0,5: domínios
   indistinguíveis; AUC = 1: totalmente separáveis.

2. DIREÇÃO DO DESLOCAMENTO: no espaço padronizado pelo 2021 DF, a direção "real × falso" é o vetor
   de pesos w do probe treinado no 2021 DF (f3_probing). O deslocamento dos genuínos de um
   cenário é d = média(genuínos do cenário) − média(genuínos do 2021 DF). A projeção d·w/|w|,
   dividida pela distância entre as médias de genuínos e falsos do 2021 DF na mesma direção, diz
   que fração do caminho até o lado "falso" o deslocamento de domínio percorre:
   0 = não empurra; 1 = leva a média dos genuínos até a média dos falsos; negativo = empurra para
   o lado "genuíno". Calculada também para os falsos.

Saídas: resultados/fase3/dominio.json e figuras/dominio.png.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F
import f3_probing as P

ALVOS = ['df21_hidden', 'itw', 'brs']
NOMES = {'df21_hidden': 'hidden track', 'itw': 'In-the-Wild', 'brs': 'BRSpeech-DF'}


def auc_dominio(X, d, g, C):
    """AUC de separar d==1 (alvo) de d==0 (2021 DF), CV por locutor."""
    from sklearn.model_selection import GroupKFold
    s = np.zeros(d.size)
    for tr, te in GroupKFold(n_splits=3).split(X, d, g):
        s[te] = P.classificador(C).fit(X[tr], d[tr]).decision_function(X[te])
    return D.auc(s[d == 1], s[d == 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True)
    ap.add_argument('--amostra', required=True)
    ap.add_argument('--C', type=float, default=0.1)
    args = ap.parse_args()
    from sklearn.preprocessing import StandardScaler

    z = np.load(D._usar(args.features))
    am = {r['id']: r for r in D.ler_tsv(D._usar(args.amostra))}
    ids = [str(i) for i in z['ids']]
    cen = np.array([am[i]['cenario'] for i in ids])
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    loc = np.array([f"{am[i]['cenario']}:{am[i]['locutor']}" for i in ids])
    cam = P.memmap_npz(args.features, 'camadas')
    reps = [(f'camada_{i + 1}', lambda i=i: np.ascontiguousarray(cam[:, i, :], dtype=np.float32)) for i in range(cam.shape[1])]
    reps += [('final', lambda: z['final'].astype(np.float32)), ('aasist', lambda: z['aasist'].astype(np.float32))]
    m_df = cen == 'df21_eval'

    res = {}
    for nome, carregar in reps:
        X = carregar()
        r = {'auc_dominio': {}, 'deslocamento_rumo_falso': {}}
        # direção real x falso aprendida no 2021 DF (espaço padronizado pelo 2021 DF)
        sc = StandardScaler().fit(X[m_df])
        Xs = sc.transform(X)
        clf = P.classificador(args.C).fit(X[m_df], y[m_df])
        w = clf[-1].coef_[0]
        w = w / np.linalg.norm(w)
        mg = Xs[m_df & (y == 1)].mean(0)
        mf = Xs[m_df & (y == 0)].mean(0)
        sep = float((mg - mf) @ w)  # distância genuíno-falso no 2021 DF, ao longo de w
        for alvo in ALVOS:
            for classe, rot in (('genuinos', 1), ('falsos', 0)):
                m = ((cen == alvo) | m_df) & (y == rot)
                d = (cen[m] == alvo).astype(int)
                r['auc_dominio'][f'{alvo}/{classe}'] = auc_dominio(X[m], d, loc[m], args.C)
                ref = mg if rot == 1 else mf
                desl = Xs[(cen == alvo) & (y == rot)].mean(0) - ref
                # positivo = empurra para o lado "falso" (para genuínos) ou "genuíno" (para falsos)
                proj = float(desl @ w)
                r['deslocamento_rumo_falso'][f'{alvo}/{classe}'] = (-proj if rot == 1 else proj) / sep
        r['separacao_df21'] = sep
        res[nome] = r
        print(f"{nome:10s} AUC domínio (gen): " + ' '.join(f"{NOMES[a]}={r['auc_dominio'][a + '/genuinos']:.3f}" for a in ALVOS)
              + ' | deslocamento rumo ao falso (gen): '
              + ' '.join(f"{NOMES[a]}={r['deslocamento_rumo_falso'][a + '/genuinos']:+.2f}" for a in ALVOS), flush=True)
        del X, Xs

    out = D.pasta_saida('fase3')
    D.salvar_json(os.path.join(out, 'dominio.json'), res, __file__, {'C': args.C, 'cv': 'GroupKFold(3) por locutor'})

    nomes = list(res)
    x = np.arange(len(nomes))
    fig, axs = F.plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for cor, a in zip((F.AZUL, F.AQUA, F.LARANJA), ALVOS):
        axs[0].plot(x, [res[n]['auc_dominio'][a + '/genuinos'] for n in nomes], '-', color=cor, label=NOMES[a])
        axs[1].plot(x, [res[n]['deslocamento_rumo_falso'][a + '/genuinos'] for n in nomes], '-', color=cor, label=NOMES[a])
    axs[0].set_ylabel('AUC (genuínos do cenário × do 2021 DF)')
    axs[0].set_title('Quanto o domínio dos genuínos é distinguível', loc='left')
    axs[1].axhline(0, color=F.TINTA2, linewidth=1)
    axs[1].axhline(1, color=F.MUDO, linewidth=1, linestyle=':')
    axs[1].set_ylabel('fração do caminho até os falsos')
    axs[1].set_title('Para onde o domínio empurra os genuínos (1 = até os falsos)', loc='left')
    nc = len(nomes) - 2
    for ax in axs:
        ax.set_xticks(list(range(0, nc, 2)) + [nc, nc + 1], [str(i + 1) for i in range(0, nc, 2)] + ['fin.', 'AAS.'], fontsize=7)
        ax.set_xlabel('camada do XLS-R, saída final e AASIST')
    axs[0].legend(loc='lower right', fontsize=8)
    F.salvar(fig, os.path.join(D.pasta_saida('fase3', 'figuras'), 'dominio.png'))


if __name__ == '__main__':
    main()
