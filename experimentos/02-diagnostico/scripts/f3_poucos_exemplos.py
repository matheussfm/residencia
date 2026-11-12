"""Fase 3: quanto um classificador linear recupera com poucos exemplos do domínio-alvo?

Para o In-the-Wild e o BRSpeech-DF, nas camadas 5 e 24 do XLS-R: sorteia metade dos locutores do
cenário como "fonte", treina o probe linear (o mesmo do f3_probing.py) com k exemplos por classe
desses locutores e testa nos locutores restantes (nunca vistos). Repete 20 vezes por k.

Responde à pergunta prática da Semana 10: quantos exemplos rotulados do domínio-alvo bastam
para recuperar a separação que existe nas representações?

Saída: resultados/fase3/poucos_exemplos.json.

Exemplo:
    python f3_poucos_exemplos.py --features ../resultados/fase3/features.npz \
        --amostra ../listas/fase3/amostra.tsv
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import f3_probing as P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True)
    ap.add_argument('--amostra', required=True)
    ap.add_argument('--camadas', type=int, nargs='+', default=[5, 24])
    ap.add_argument('--k', type=int, nargs='+', default=[10, 50, 200])
    ap.add_argument('--sorteios', type=int, default=20)
    ap.add_argument('--C', type=float, default=0.1)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()

    z = np.load(D._usar(args.features))
    am = {r['id']: r for r in D.ler_tsv(D._usar(args.amostra))}
    ids = [str(i) for i in z['ids']]
    cen = np.array([am[i]['cenario'] for i in ids])
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    loc = np.array([am[i]['locutor'] for i in ids])
    cam = P.memmap_npz(args.features, 'camadas')
    rng = np.random.default_rng(args.semente)

    res = {}
    for camada in args.camadas:
        X = np.ascontiguousarray(cam[:, camada - 1, :], dtype=np.float32)
        for c in ('brs', 'itw'):
            m = np.flatnonzero(cen == c)
            locs = np.unique(loc[m])
            for k in args.k:
                eers = []
                for _ in range(args.sorteios):
                    fonte = set(rng.choice(locs, locs.size // 2, replace=False))
                    mf = np.array([l in fonte for l in loc[m]])
                    idx_f, idx_t = m[mf], m[~mf]
                    gb, gs = idx_f[y[idx_f] == 1], idx_f[y[idx_f] == 0]
                    if gb.size < k or gs.size < k or np.unique(y[idx_t]).size < 2:
                        continue
                    tr = np.concatenate([rng.choice(gb, k, replace=False), rng.choice(gs, k, replace=False)])
                    s = P.classificador(args.C).fit(X[tr], y[tr]).decision_function(X[idx_t])
                    eers.append(D.eer(s[y[idx_t] == 1], s[y[idx_t] == 0])[0])
                e = np.array(eers)
                res[f'{c}_camada{camada}_k{k}'] = {'mediana': float(np.median(e)), 'p10': float(np.percentile(e, 10)),
                                                   'p90': float(np.percentile(e, 90)), 'n': int(e.size)}
                print(f'{c} camada {camada:2d} k={k:3d} por classe: EER mediana {100 * np.median(e):5.1f}% '
                      f'[p10 {100 * np.percentile(e, 10):4.1f}, p90 {100 * np.percentile(e, 90):4.1f}] ({e.size} sorteios)',
                      flush=True)

    D.salvar_json(os.path.join(D.pasta_saida('fase3'), 'poucos_exemplos.json'),
                  {'descricao': 'probe linear treinado com k exemplos por classe de metade dos locutores do '
                                'cenário-alvo e testado na outra metade', 'resultados': res}, __file__,
                  {'camadas': args.camadas, 'k': args.k, 'sorteios': args.sorteios, 'C': args.C, 'semente': args.semente})


if __name__ == '__main__':
    main()
