"""Etapa 2 (I1): probe treinado com aumento de dados nas duas classes, sem nenhum dado do alvo.

Para cada representação do pré-registro (camadas 5, 12, 24 e AASIST) e cada método:
  limpo      só a partição de treino do df21_eval (é o B2 da Etapa 1);
  controle   treino limpo + a cópia "controle" (mesmo caminho de E/S das cópias aumentadas, sem aumento);
  a, b, c, d treino limpo + a cópia aumentada da variante (i1_aumentar.py),
o C é escolhido na validação LIMPA da origem (grade do pré-registro), e a representação de cada
método também (README, seção 4.3). Os probes são avaliados nos conjuntos da Etapa 1.

Variante I1 + B1: os probes limpo e (d) aplicados às representações dos áudios de teste
normalizados (volume_padronizar) e da condição controle da Fase 2c, extraídas com o f3_extrair.py
a partir das listas do e2_listas_b1.py. A diferença pareada da normalização é medida contra o
mesmo probe na condição controle.

Comparações (diferença pareada de EER e minDCF, IC por bootstrap de locutores):
  I1_v − B2, com a representação escolhida de cada um e em cada representação fixa;
  I1_v − I1_controle (efeito do aumento além de duplicar o treino); I1_v − B0;
  (B2 + B1) − (B2 + controle); (I1d + B1) − (I1d + controle); (I1d + B1) − (B2 + controle);
  (I1d + B1) − B1 (detector normalizado).

Conferências: as pontuações do probe limpo reproduzem as do B2 da Etapa 1; as pontuações do detector
gravadas nos .npz da variante B1 reproduzem as da Fase 2c; a EER do detector nas cópias aumentadas
do treino (descritiva, só para confirmar que o aumento muda o que o detector vê; NÃO é teste de ganho).

Saídas: resultados/etapa2/i1.json (+ .meta.json), resultados/etapa2/pontuacoes_i1.tsv e
resultados/etapa2/i1_treino_reps.npz (representações das cópias aumentadas do treino, só as 4 do
pré-registro, para a I2; fora do git).

Roda no servidor, no ambiente isolado com scikit-learn (as features não cabem no PC). Exemplo, a
partir de experimentos/03-intervencao/scripts:
    python -B i1_probing.py --features ../../02-diagnostico/resultados/fase3/features.npz \
        --features-i1 ../resultados/etapa2/features_i1.npz \
        --features-b1 ../resultados/etapa2/features_volume_padronizar.npz \
        --features-controle ../resultados/etapa2/features_controle.npz
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _interv import D
import e1_linhas_de_base as E1
import f3_probing as P  # noqa: E402
from _comum import escrever_tsv  # noqa: E402

VARIANTES = ['controle', 'a', 'b', 'c', 'd']
METODOS = ['limpo'] + VARIANTES


def ids_npz(caminho):
    return [str(i) for i in np.load(D._usar(caminho))['ids']]


def matriz(caminho, rep, linhas):
    """Representação `rep` das linhas (índices do .npz), em float32."""
    if rep == 'aasist':
        X = np.load(caminho)['aasist'][linhas]
    else:
        X = P.memmap_npz(caminho, 'camadas')[linhas, int(rep.split('_')[1]) - 1, :]
    return np.ascontiguousarray(X, dtype=np.float32)


def alinhar(caminho, ids_desejados):
    pos = {i: k for k, i in enumerate(ids_npz(caminho))}
    faltam = [i for i in ids_desejados if i not in pos]
    if faltam:
        D.erro(f'{len(faltam)} ids sem representação em {caminho} (ex.: {faltam[:3]})')
    return np.array([pos[i] for i in ids_desejados])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True, help='features.npz da Fase 3 do Exp. 02')
    ap.add_argument('--features-i1', required=True)
    ap.add_argument('--features-b1', default=None, help='volume_padronizar (opcional)')
    ap.add_argument('--features-controle', default=None, help='controle da Fase 2c (obrigatório com --features-b1)')
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--saida', default=None, help='pasta de saída (padrão: resultados/etapa2)')
    args = ap.parse_args()
    if bool(args.features_b1) != bool(args.features_controle):
        D.erro('--features-b1 e --features-controle vão juntos')
    out = args.saida or I.pasta_saida('etapa2')
    D.garantir_fora_third_party(out)
    os.makedirs(out, exist_ok=True)

    am, div = I.ler_amostra(), I.ler_divisao()
    conj = E1.conjuntos(am, div)
    ids = ids_npz(args.features)
    if set(ids) != set(am):
        D.erro('ids do .npz e do amostra.tsv não coincidem')
    pos = {i: k for k, i in enumerate(ids)}
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    treino = [i for i in div if div[i] == 'treino']  # mesma ordem da Etapa 1
    i_tr = np.array([pos[i] for i in treino])
    i_va = np.array([pos[i] for i in div if div[i] == 'validacao'])
    y_tr = y[i_tr]
    trials = [i.split('::', 1)[1] for i in treino]
    lin_i1 = {v: alinhar(args.features_i1, [f'i1_{v}::{t}' for t in trials]) for v in VARIANTES}
    com_b1 = bool(args.features_b1)
    if com_b1:
        lin_b1, lin_ctl = alinhar(args.features_b1, ids), alinhar(args.features_controle, ids)

    # ---- conferências
    conf = {}
    z_i1 = np.load(args.features_i1)
    p_i1 = z_i1['pontuacao'].astype(float)
    conf['eer_detector_copias_treino'] = {'original': D.eer(np.array([float(am[i]['pontuacao_exp01']) for i in treino])[y_tr == 1],
                                                            np.array([float(am[i]['pontuacao_exp01']) for i in treino])[y_tr == 0])[0]}
    for v in VARIANTES:
        s = p_i1[lin_i1[v]]
        conf['eer_detector_copias_treino'][v] = D.eer(s[y_tr == 1], s[y_tr == 0])[0]
    print('EER do detector nas cópias do treino (descritivo):',
          {k: D.pct(v) for k, v in conf['eer_detector_copias_treino'].items()})
    if com_b1:
        for nome, cam, lin, cond in (('volume_padronizar', args.features_b1, lin_b1, 'volume_padronizar'),
                                     ('controle', args.features_controle, lin_ctl, 'controle')):
            p_npz = np.load(cam)['pontuacao'].astype(float)[lin]
            ref = E1.pontuacoes_fase2c(cond, am)
            p_ref = np.array([ref[i] for i in ids])
            conf[f'pontuacao_npz_vs_fase2c_{nome}'] = {'pearson': float(np.corrcoef(p_npz, p_ref)[0, 1]),
                                                        'dif_abs_max': float(np.abs(p_npz - p_ref).max())}
        print('conferência das listas B1:', {k: v for k, v in conf.items() if k.startswith('pontuacao')})

    # ---- probes
    grade, escolha, pont = [], {m: {} for m in METODOS}, {}
    reps_treino = {}
    for rep in I.REPRESENTACOES:
        Xb = matriz(args.features, rep, np.arange(len(ids)))
        Xi = {v: matriz(args.features_i1, rep, lin_i1[v]) for v in VARIANTES}
        reps_treino.update({f'{v}__{rep}': Xi[v].astype(np.float16) for v in VARIANTES})
        Xn = matriz(args.features_b1, rep, lin_b1) if com_b1 else None
        Xc = matriz(args.features_controle, rep, lin_ctl) if com_b1 else None
        for m in METODOS:
            Xtr = Xb[i_tr] if m == 'limpo' else np.concatenate([Xb[i_tr], Xi[m]])
            ytr = y_tr if m == 'limpo' else np.concatenate([y_tr, y_tr])
            linhas = []
            for C in I.GRADE_C:
                s = P.classificador(C).fit(Xtr, ytr).decision_function(Xb[i_va])
                linhas.append({'metodo': m, 'representacao': rep, 'C': C,
                               'eer_validacao': D.eer(s[y[i_va] == 1], s[y[i_va] == 0])[0],
                               'mindcf_validacao': I.min_dcf(s[y[i_va] == 1], s[y[i_va] == 0])[0]})
            grade += linhas
            melhor = E1.escolher(linhas)
            escolha[m][rep] = melhor
            clf = P.classificador(melhor['C']).fit(Xtr, ytr)
            s = clf.decision_function(Xb)
            pont[f'{m}@{rep}'] = {i: float(s[k]) for k, i in enumerate(ids)}
            if com_b1 and m in ('limpo', 'd'):
                sn, sc = clf.decision_function(Xn), clf.decision_function(Xc)
                pont[f'{m}+B1@{rep}'] = {i: float(sn[k]) for k, i in enumerate(ids)}
                pont[f'{m}+ctl@{rep}'] = {i: float(sc[k]) for k, i in enumerate(ids)}
            print(f"  {rep:10s} {m:9s} C={melhor['C']:<6g} validação EER {D.pct(melhor['eer_validacao'])}", flush=True)
        del Xb, Xi, Xn, Xc
    sel = {m: E1.escolher(list(escolha[m].values()))['representacao'] for m in METODOS}
    print('representação escolhida por método:', sel)

    # o probe limpo deve reproduzir o B2 da Etapa 1
    b2_tsv = os.path.join(I.EXP03, 'resultados', 'etapa1', 'pontuacoes_b2.tsv')
    if os.path.exists(b2_tsv):
        b2 = D.ler_tsv(D._usar(b2_tsv))
        conf['limpo_vs_b2_etapa1_dif_abs_max'] = {rep: float(max(abs(float(r[rep]) - pont[f'limpo@{rep}'][r['id']]) for r in b2))
                                                  for rep in I.REPRESENTACOES}
        print('probe limpo × B2 da Etapa 1 (diferença máxima):', conf['limpo_vs_b2_etapa1_dif_abs_max'])

    # ---- métodos e comparações
    metodos = {'B0': {i: float(r['pontuacao_exp01']) for i, r in am.items()},
               'B1': E1.pontuacoes_fase2c('volume_padronizar', am)}
    metodos['B2'] = pont[f"limpo@{sel['limpo']}"]
    for v in VARIANTES:
        metodos[f'I1_{v}'] = pont[f'{v}@{sel[v]}']
    if com_b1:
        metodos['B2+B1'] = pont[f"limpo+B1@{sel['limpo']}"]
        metodos['B2+ctl'] = pont[f"limpo+ctl@{sel['limpo']}"]
        metodos['I1_d+B1'] = pont[f"d+B1@{sel['d']}"]
        metodos['I1_d+ctl'] = pont[f"d+ctl@{sel['d']}"]
    comp = [(f'I1_{v}', 'B2') for v in VARIANTES] + [(f'I1_{v}', 'I1_controle') for v in 'abcd'] \
        + [(f'I1_{v}', 'B0') for v in VARIANTES]
    if com_b1:
        comp += [('B2+B1', 'B2+ctl'), ('I1_d+B1', 'I1_d+ctl'), ('I1_d+B1', 'B2+ctl'), ('I1_d+B1', 'B1')]

    res = {'selecao': {'grade': grade, 'por_metodo_e_representacao': escolha, 'selecionada': sel},
           'conferencias': conf, 'metricas': {}, 'diferencas': {}, 'por_representacao': {}}
    for k, idk in conj.items():
        res['metricas'][k], res['diferencas'][k], res['por_representacao'][k] = {}, {}, {}
        for m, p in metodos.items():
            c = E1.cenario(k, idk, p, am)
            d = I.metricas_com_ic(c, B=args.B, semente=args.semente)
            if k == 'brs':
                ruim = np.isin(c.meta['locutor'], list(E1.RUINS))
                for rot, mk in (('ruins', ruim), ('bons', ~ruim)):
                    d[f'eer_{rot}'] = D.eer(c.pont[mk & (c.rotulo == 1)], c.pont[mk & (c.rotulo == 0)])[0]
            res['metricas'][k][m] = d
        for a, b in comp:
            base = E1.cenario(k, idk, metodos[b], am)
            res['diferencas'][k][f'{a}-{b}'] = I.diferenca_pareada(base, [metodos[a][i] for i in idk], B=args.B, semente=args.semente)
        # cada representação fixa: o efeito do aumento sem o efeito da escolha
        for rep in I.REPRESENTACOES:
            base = E1.cenario(k, idk, pont[f'limpo@{rep}'], am)
            r = {'limpo': {'eer': D.eer(base.bona, base.spoof)[0], 'mindcf': I.min_dcf(base.bona, base.spoof)[0]}}
            for v in VARIANTES:
                c = E1.cenario(k, idk, pont[f'{v}@{rep}'], am)
                r[v] = {'eer': D.eer(c.bona, c.spoof)[0], 'mindcf': I.min_dcf(c.bona, c.spoof)[0],
                        'dif_vs_limpo': I.diferenca_pareada(base, c.pont, B=args.B, semente=args.semente)}
            res['por_representacao'][k][rep] = r
        print(f'\n== {k}')
        for m, d in res['metricas'][k].items():
            e = d['eer']
            extra = f"  ruins {D.pct(d['eer_ruins'])} bons {D.pct(d['eer_bons'])}" if k == 'brs' else ''
            print(f"  {m:12s} EER {D.pct(e['valor']):>7s} [{D.pct(e['ic95'][0])}, {D.pct(e['ic95'][1])}]  "
                  f"minDCF {d['mindcf']['valor']:.3f}{extra}")
        for nome, d in res['diferencas'][k].items():
            e = d['eer']
            print(f"  Δ {nome:20s} EER {100 * e['valor']:+6.2f} pp [{100 * e['ic95'][0]:+.2f}, {100 * e['ic95'][1]:+.2f}]")
        for rep, r in res['por_representacao'][k].items():
            print(f"  {rep:10s} limpo {D.pct(r['limpo']['eer'])} | " + ' '.join(
                f"{v} {D.pct(r[v]['eer'])} ({100 * r[v]['dif_vs_limpo']['eer']['valor']:+.1f} "
                f"[{100 * r[v]['dif_vs_limpo']['eer']['ic95'][0]:+.1f}, {100 * r[v]['dif_vs_limpo']['eer']['ic95'][1]:+.1f}])"
                for v in VARIANTES), flush=True)

    tsv = os.path.join(out, 'pontuacoes_i1.tsv')
    cols = sorted(pont)
    escrever_tsv(tsv, ['id'] + cols, [[i] + [f'{pont[c][i]:.6f}' for c in cols] for i in ids])
    npz = os.path.join(out, 'i1_treino_reps.npz')
    np.savez(npz, ids_treino=np.array(treino), **reps_treino)
    res['pontuacoes_i1_tsv_sha256'] = D.sha256(tsv)
    res['i1_treino_reps_npz_sha256'] = D.sha256(npz)
    I.salvar_json(os.path.join(out, 'i1.json'), res, __file__,
                  {'B': args.B, 'semente': args.semente, 'representacoes': I.REPRESENTACOES, 'grade_C': I.GRADE_C,
                   'metodos': METODOS, 'criterio_escolha': 'menor EER na validação limpa; desempate minDCF e menor C',
                   'com_b1': com_b1})


if __name__ == '__main__':
    main()
