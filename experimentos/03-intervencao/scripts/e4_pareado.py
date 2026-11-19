"""Etapa 4, complemento: diferença pareada I1 (d) − B2 na validação do BRSpeech-DF.

O e4_brs_val.py compara B2 e I1 (d) com o detector (B0), mas não entre si. Este script retreina os
mesmos probes (mesmos dados, mesmos C das Etapas 1 e 2) e mede, em cada representação do
pré-registro, a diferença pareada de EER e de minDCF entre I1 (d) e B2 na validação, com IC por
bootstrap de locutores (os mesmos locutores sorteados nos dois). Confere também que as EER de B2 e
I1 (d) reproduzem as do brs_val.json.

Saída: resultados/etapa4/pareado.json (+ .meta.json). Roda no servidor, no ambiente isolado:
    python -B e4_pareado.py --features-val ../resultados/etapa4/features_brs_val.npz --lista-val ../listas/brs_val
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _interv import D
import f3_probing as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', default=os.path.join(D.EXP02, 'resultados', 'fase3', 'features.npz'))
    ap.add_argument('--features-val', required=True)
    ap.add_argument('--lista-val', required=True)
    ap.add_argument('--etapa1', default=os.path.join(I.EXP03, 'resultados', 'etapa1', 'linhas_de_base.json'))
    ap.add_argument('--i1', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1.json'))
    ap.add_argument('--i1-reps', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1_treino_reps.npz'))
    ap.add_argument('--brs-val', default=os.path.join(I.EXP03, 'resultados', 'etapa4', 'brs_val.json'))
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()

    am, div = I.ler_amostra(), I.ler_divisao()
    z = np.load(D._usar(args.features))
    ids = [str(i) for i in z['ids']]
    pos = {i: k for k, i in enumerate(ids)}
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    treino = [i for i in div if div[i] == 'treino']
    i_tr = np.array([pos[i] for i in treino])
    e1 = json.load(open(D._usar(args.etapa1), encoding='utf-8'))
    i1 = json.load(open(D._usar(args.i1), encoding='utf-8'))
    zr = np.load(D._usar(args.i1_reps))
    if [str(i) for i in zr['ids_treino']] != treino:
        D.erro('ids_treino fora de ordem')
    ref = json.load(open(D._usar(args.brs_val), encoding='utf-8'))['segundo_teste']['metricas']

    rot = D.ler_tsv(D._usar(os.path.join(args.lista_val, 'rotulos.tsv')))
    mapa = {r['id']: r['id_referencia'] for r in D.ler_tsv(D._usar(os.path.join(args.lista_val, 'mapa_falsos_ordenado.tsv')))}
    zv = np.load(D._usar(args.features_val))
    ids_v = [str(i) for i in zv['ids']]
    posv = {i: k for k, i in enumerate(ids_v)}
    rot = sorted(rot, key=lambda r: posv[r['id']])
    yv = np.array([r['rotulo'] == 'bonafide' for r in rot], int)
    loc_v = np.array([(r['id'] if r['rotulo'] == 'bonafide' else mapa[r['id']].split('/', 1)[1]).split('/')[2] for r in rot])

    cam, camv = P.memmap_npz(args.features, 'camadas'), P.memmap_npz(args.features_val, 'camadas')
    res = {}
    for r in I.REPRESENTACOES:
        if r == 'aasist':
            X, Xv = z['aasist'].astype(np.float32), zv['aasist'].astype(np.float32)
        else:
            c = int(r.split('_')[1]) - 1
            X = np.ascontiguousarray(cam[:, c, :], dtype=np.float32)
            Xv = np.ascontiguousarray(camv[:, c, :], dtype=np.float32)
        Xo = X[i_tr]
        s_b2 = P.classificador(e1['selecao_b2']['por_representacao'][r]['C']).fit(Xo, y[i_tr]).decision_function(Xv)
        Xod = np.concatenate([Xo, zr[f'd__{r}'].astype(np.float32)])
        s_d = P.classificador(i1['selecao']['por_metodo_e_representacao']['d'][r]['C']).fit(
            Xod, np.concatenate([y[i_tr], y[i_tr]])).decision_function(Xv)
        base = D.Cenario('brs_val', ids_v, s_b2, yv, {'locutor': loc_v})
        d = I.diferenca_pareada(base, s_d, B=args.B, semente=args.semente)
        e_b2, e_d = D.eer(s_b2[yv == 1], s_b2[yv == 0])[0], D.eer(s_d[yv == 1], s_d[yv == 0])[0]
        res[r] = {'eer_B2': e_b2, 'eer_I1_d': e_d, 'I1_d-B2': d,
                  'conferencia_dif_vs_brs_val': {'B2': abs(e_b2 - ref[f'B2@{r}']['eer']['valor']),
                                                 'I1_d': abs(e_d - ref[f'I1_d@{r}']['eer']['valor'])}}
        e = d['eer']
        print(f"{r:10s} B2 {D.pct(e_b2)}  I1_d {D.pct(e_d)}  Δ {100 * e['valor']:+.2f} pp [{100 * e['ic95'][0]:+.2f}, "
              f"{100 * e['ic95'][1]:+.2f}]  minDCF Δ {d['mindcf']['valor']:+.3f}  conferência {res[r]['conferencia_dif_vs_brs_val']}",
              flush=True)
    out = I.pasta_saida('etapa4')
    I.salvar_json(os.path.join(out, 'pareado.json'), res, __file__, {'B': args.B, 'semente': args.semente})


if __name__ == '__main__':
    main()
