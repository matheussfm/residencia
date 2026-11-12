"""Fase 3, controles complementares do probing no BRSpeech-DF.

1. TTS DEIXADO DE FORA (controle do "1,2%" do probing no próprio domínio): para cada um dos 5
   TTS, o probe é treinado com os genuínos e os falsos dos outros 4 TTS e testado nos genuínos e
   nos falsos do TTS deixado de fora, com locutores também disjuntos (5 partes por locutor). Se a
   EER continua baixa, o probe usa algo comum à síntese, e não traços de um gerador específico.
   Não controla o que é comum a TODOS os falsos do dataset (o mesmo processo de geração) nem
   o que é comum a todos os genuínos (o mesmo corpus de origem).

2. LOCUTORES "RUINS" × "BONS": no teste do BRSpeech-DF, 3 dos 10 locutores (5739, 13063, 12626)
   concentram 87% dos genuínos rejeitados pelo detector (Fase 1, seção 4.4). Para cada grupo:
   EER do detector completo e EER do probe no próprio domínio, treinado nos locutores do OUTRO
   grupo (transferência entre grupos) e em validação cruzada dentro do grupo. Responde se a voz
   real desses locutores é separável pelas representações ou se fica misturada com os falsos.

Camadas: 3, 5, 12, 24 e a saída final.

Saída: resultados/fase3/complementos.json.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import f3_probing as P

RUINS = {'5739', '13063', '12626'}


def eer_de(s, y):
    return D.eer(s[y == 1], s[y == 0])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True)
    ap.add_argument('--amostra', required=True)
    ap.add_argument('--camadas', type=int, nargs='+', default=[3, 5, 12, 24])
    ap.add_argument('--C', type=float, default=0.1)
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold

    z = np.load(D._usar(args.features))
    am = {r['id']: r for r in D.ler_tsv(D._usar(args.amostra))}
    ids = [str(i) for i in z['ids']]
    m_brs = np.array([am[i]['cenario'] == 'brs' for i in ids])
    idx = np.flatnonzero(m_brs)
    y = np.array([am[ids[i]]['rotulo'] == 'bonafide' for i in idx], int)
    loc = np.array([am[ids[i]]['locutor'] for i in idx])
    grupo = np.array([am[ids[i]]['grupo'] for i in idx])  # TTS ou 'bonafide'
    pont = z['pontuacao'][idx].astype(float)
    cam = P.memmap_npz(args.features, 'camadas')
    reps = [(f'camada_{c}', lambda c=c: np.ascontiguousarray(cam[idx, c - 1, :], dtype=np.float32))
            for c in args.camadas]
    reps.append(('final', lambda: z['final'][idx].astype(np.float32)))
    tts_lista = sorted(set(grupo) - {'bonafide'})
    ruim = np.isin(loc, list(RUINS))

    res = {'tts_deixado_de_fora': {}, 'locutores': {
        'ruins': sorted(RUINS), 'n_genuinos_ruins': int((ruim & (y == 1)).sum()),
        'n_genuinos_bons': int((~ruim & (y == 1)).sum()),
        'eer_modelo': {'ruins': eer_de(pont[ruim], y[ruim]), 'bons': eer_de(pont[~ruim], y[~ruim])},
        'camadas': {}}}
    print(f"detector completo: EER locutores ruins {D.pct(res['locutores']['eer_modelo']['ruins'])}, "
          f"bons {D.pct(res['locutores']['eer_modelo']['bons'])}")

    for nome, carregar in reps:
        X = carregar()
        # 1) TTS deixado de fora, com locutores disjuntos
        d_tts = {}
        for tts in tts_lista:
            s = np.full(y.size, np.nan)
            for tr, te in GroupKFold(n_splits=5).split(X, y, loc):
                tr = tr[grupo[tr] != tts]                              # treino sem o TTS
                te = te[(grupo[te] == tts) | (grupo[te] == 'bonafide')]  # teste só nele (+ genuínos)
                s[te] = P.classificador(args.C).fit(X[tr], y[tr]).decision_function(X[te])
            m = ~np.isnan(s)
            d_tts[tts] = eer_de(s[m], y[m])
        # referência: mesmo esquema, mas com o TTS presente no treino
        s = np.zeros(y.size)
        for tr, te in GroupKFold(n_splits=5).split(X, y, loc):
            s[te] = P.classificador(args.C).fit(X[tr], y[tr]).decision_function(X[te])
        d_tts['_referencia_todos_os_tts_no_treino'] = eer_de(s, y)
        res['tts_deixado_de_fora'][nome] = d_tts

        # 2) locutores ruins x bons
        d_loc = {}
        for alvo, m_alvo in (('ruins', ruim), ('bons', ~ruim)):
            clf = P.classificador(args.C).fit(X[~m_alvo], y[~m_alvo])   # treinado no outro grupo
            d_loc[f'{alvo}_treino_no_outro_grupo'] = eer_de(clf.decision_function(X[m_alvo]), y[m_alvo])
            Xa, ya, la = X[m_alvo], y[m_alvo], loc[m_alvo]
            sa = np.zeros(ya.size)
            for tr, te in GroupKFold(n_splits=np.unique(la).size).split(Xa, ya, la):  # deixa 1 locutor de fora
                sa[te] = P.classificador(args.C).fit(Xa[tr], ya[tr]).decision_function(Xa[te])
            d_loc[f'{alvo}_cv_dentro_do_grupo'] = eer_de(sa, ya)
        res['locutores']['camadas'][nome] = d_loc
        print(f"{nome:9s} TTS fora: " + ' '.join(f"{t}={D.pct(v, 1)}" for t, v in d_tts.items() if not t.startswith('_'))
              + f" (ref. {D.pct(d_tts['_referencia_todos_os_tts_no_treino'], 1)})"
              + f" | ruins: treino nos bons {D.pct(d_loc['ruins_treino_no_outro_grupo'], 1)},"
                f" CV {D.pct(d_loc['ruins_cv_dentro_do_grupo'], 1)}"
              + f" | bons: treino nos ruins {D.pct(d_loc['bons_treino_no_outro_grupo'], 1)},"
                f" CV {D.pct(d_loc['bons_cv_dentro_do_grupo'], 1)}", flush=True)
        del X

    D.salvar_json(os.path.join(D.pasta_saida('fase3'), 'complementos.json'), res, __file__,
                  {'camadas': args.camadas, 'C': args.C, 'locutores_ruins': sorted(RUINS)})


if __name__ == '__main__':
    main()
