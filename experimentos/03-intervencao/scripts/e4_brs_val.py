"""Etapa 4: a divisão de validação do BRSpeech-DF (10 locutores, disjuntos dos do teste).

A divisão de treino (147,5 GB no Hugging Face) não foi usada; a de validação tem 3,5 GB, 1.161
genuínos e 5.795 falsos, e locutores disjuntos dos do teste (conferido na metadata do AKCIT).

  (a) SEGUNDO TESTE: B0 (detector), B2 (probe limpo) e I1 (d) avaliados na validação, com EER e
      minDCF e IC por bootstrap de locutores, nas quatro representações do pré-registro. Mostra se o
      quadro do teste oficial (10 locutores, percentil 94 dos sorteios no Exp. 02) se repete com
      outros 10 locutores.
  (b) FONTE DE ADAPTAÇÃO DA I2: os k exemplos vêm da validação (locutores novos) e o teste é a amostra
      do teste oficial, completa nos locutores (os 10), com o rodízio de TTS da Etapa 3 (3 TTS na
      adaptação, os outros 2 no teste; 20 sorteios). Locutores disjuntos por construção.

Conferência do pareamento dos falsos: o parquet não traz o nome original dos falsos; o locutor vem do
mapa "ordenado" do preparar_brspeech_hf.py (validado no teste no Exp. 01). Aqui ele é conferido de
novo: as pontuações do detector (repetição) devem correlacionar bem com as do AKCIT (zeros) pelo mapa
"ordenado" e mal pelo mapa "csv" (no Exp. 01, no teste: 0,82 contra 0,11 nos falsos).

Entradas: features.npz da Fase 3 (Exp. 02), o .npz da validação (f3_extrair.py sobre a lista do
preparar_brspeech_hf.py --subset dev), a pasta da lista (rotulos.tsv e mapas), os JSON das Etapas 1
e 2 e o i1_treino_reps.npz.

Saída: resultados/etapa4/brs_val.json (+ .meta.json).

Roda no servidor, no ambiente isolado. Exemplo, a partir de experimentos/03-intervencao/scripts:
    python -B e4_brs_val.py --features-val ../resultados/etapa4/features_brs_val.npz \
        --lista-val ../listas/brs_val
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
import i2_poucos_exemplos as Q

RUINS_TESTE = {'5739', '13063', '12626'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', default=os.path.join(D.EXP02, 'resultados', 'fase3', 'features.npz'))
    ap.add_argument('--features-val', required=True)
    ap.add_argument('--lista-val', required=True, help='pasta com rotulos.tsv e mapa_falsos_*.tsv da validação')
    ap.add_argument('--etapa1', default=os.path.join(I.EXP03, 'resultados', 'etapa1', 'linhas_de_base.json'))
    ap.add_argument('--i1', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1.json'))
    ap.add_argument('--i1-reps', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1_treino_reps.npz'))
    ap.add_argument('--k', type=int, nargs='+', default=[10, 50, 200])
    ap.add_argument('--sorteios', type=int, default=20)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--saida', default=None, help='pasta de saída (padrão: resultados/etapa4)')
    args = ap.parse_args()
    out = args.saida or I.pasta_saida('etapa4')
    D.garantir_fora_third_party(out)
    os.makedirs(out, exist_ok=True)

    # ---- origem (amostra da Fase 3) e escolhas das Etapas 1 e 2
    am, div = I.ler_amostra(), I.ler_divisao()
    z = np.load(D._usar(args.features))
    ids = [str(i) for i in z['ids']]
    pos = {i: k for k, i in enumerate(ids)}
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    treino = [i for i in div if div[i] == 'treino']
    i_tr = np.array([pos[i] for i in treino])
    i_brs = np.flatnonzero(np.array([am[i]['cenario'] == 'brs' for i in ids]))
    e1 = json.load(open(D._usar(args.etapa1), encoding='utf-8'))
    i1 = json.load(open(D._usar(args.i1), encoding='utf-8'))
    C = {'limpo': {r: e1['selecao_b2']['por_representacao'][r]['C'] for r in I.REPRESENTACOES},
         'd': {r: i1['selecao']['por_metodo_e_representacao']['d'][r]['C'] for r in I.REPRESENTACOES}}
    sel = {'limpo': e1['selecao_b2']['selecionada'], 'd': i1['selecao']['selecionada']['d']}
    zr = np.load(D._usar(args.i1_reps))
    if [str(i) for i in zr['ids_treino']] != treino:
        D.erro('ids_treino do i1_treino_reps.npz fora de ordem')

    # ---- validação do BRSpeech-DF
    rot = D.ler_tsv(D._usar(os.path.join(args.lista_val, 'rotulos.tsv')))
    mapa = {r['id']: r['id_referencia'] for r in D.ler_tsv(D._usar(os.path.join(args.lista_val, 'mapa_falsos_ordenado.tsv')))}
    mapa_csv = {r['id']: r['id_referencia'] for r in D.ler_tsv(D._usar(os.path.join(args.lista_val, 'mapa_falsos_csv.tsv')))}
    zv = np.load(D._usar(args.features_val))
    ids_v = [str(i) for i in zv['ids']]
    posv = {i: k for k, i in enumerate(ids_v)}
    if set(ids_v) != {r['id'] for r in rot}:
        D.erro('ids do .npz da validação e do rotulos.tsv não coincidem')
    rot = sorted(rot, key=lambda r: posv[r['id']])
    yv = np.array([r['rotulo'] == 'bonafide' for r in rot], int)
    grp_v = np.array([r['grupo'] for r in rot])
    ref_v = [r['id'] if r['rotulo'] == 'bonafide' else mapa[r['id']].split('/', 1)[1] for r in rot]
    loc_v = np.array([p.split('/')[2] for p in ref_v])
    p_b0_v = zv['pontuacao'].astype(float)
    if set(loc_v) & {am[i]['locutor'] for i in ids if am[i]['cenario'] == 'brs'}:
        D.erro('há locutores da validação também no teste')

    # conferência do mapa dos falsos contra as pontuações do AKCIT (SSL, zeros)
    S = os.path.join(D.BRSPEECH_REPO, 'data', 'scores', 'ssl')
    ak_g = D.ler_pontuacoes(D._usar(os.path.join(S, 'ssl_brspeech_full_scores.pkl')))
    ak_s = D.ler_pontuacoes(D._usar(os.path.join(S, 'ssl_sintetico_full_scores.pkl')))
    conf = {}
    mg = yv == 1
    conf['pearson_genuinos_vs_akcit'] = float(np.corrcoef(p_b0_v[mg], [ak_g[r['id']] for r, g in zip(rot, mg) if g])[0, 1])
    for nome, mp in (('ordenado', mapa), ('csv', mapa_csv)):
        conf[f'pearson_falsos_vs_akcit_mapa_{nome}'] = float(np.corrcoef(
            p_b0_v[~mg], [ak_s[mp[r['id']]] for r, g in zip(rot, mg) if not g])[0, 1])
    print('conferência do pareamento (detector com repetição × AKCIT com zeros):', conf, flush=True)

    def cen(nome, pont):
        return D.Cenario(nome, ids_v, pont, yv, {'locutor': loc_v, 'grupo': grp_v})

    # ---- (a) segundo teste
    res_a = {'n': {'genuinos': int(yv.sum()), 'falsos': int((1 - yv).sum()), 'locutores': int(np.unique(loc_v).size)},
             'metricas': {}, 'diferencas': {}, 'por_tts': {}}
    pont = {'B0': p_b0_v}
    X, Xv = {}, {}
    cam, camv = P.memmap_npz(args.features, 'camadas'), P.memmap_npz(args.features_val, 'camadas')
    for r in I.REPRESENTACOES:
        if r == 'aasist':
            X[r], Xv[r] = z['aasist'].astype(np.float32), zv['aasist'].astype(np.float32)
        else:
            c = int(r.split('_')[1]) - 1
            X[r] = np.ascontiguousarray(cam[:, c, :], dtype=np.float32)
            Xv[r] = np.ascontiguousarray(camv[:, c, :], dtype=np.float32)
        Xo = X[r][i_tr]
        Xod, yod = np.concatenate([Xo, zr[f'd__{r}'].astype(np.float32)]), np.concatenate([y[i_tr], y[i_tr]])
        pont[f'B2@{r}'] = P.classificador(C['limpo'][r]).fit(Xo, y[i_tr]).decision_function(Xv[r])
        pont[f'I1_d@{r}'] = P.classificador(C['d'][r]).fit(Xod, yod).decision_function(Xv[r])
    del cam, camv
    for m, p in pont.items():
        c = cen('brs_val', p)
        res_a['metricas'][m] = I.metricas_com_ic(c, B=args.B, semente=args.semente)
        res_a['por_tts'][m] = {t: D.eer(c.bona, c.pont[grp_v == t])[0] for t in Q.TTS}
        por_loc = {}
        for l in np.unique(loc_v):
            ml = loc_v == l
            por_loc[l] = D.eer(c.pont[ml & (yv == 1)], c.pont[ml & (yv == 0)])[0]
        res_a['metricas'][m]['eer_por_locutor'] = por_loc
    base = cen('brs_val', p_b0_v)
    for m in pont:
        if m != 'B0':
            res_a['diferencas'][f'{m}-B0'] = I.diferenca_pareada(base, pont[m], B=args.B, semente=args.semente)
    print('\n(a) validação do BRSpeech-DF como segundo teste:')
    for m, d in res_a['metricas'].items():
        e = d['eer']
        print(f"  {m:16s} EER {D.pct(e['valor'])} [{D.pct(e['ic95'][0])}, {D.pct(e['ic95'][1])}]  minDCF {d['mindcf']['valor']:.3f}")
    for n, d in res_a['diferencas'].items():
        e = d['eer']
        print(f"  Δ {n:16s} {100 * e['valor']:+.2f} pp [{100 * e['ic95'][0]:+.2f}, {100 * e['ic95'][1]:+.2f}]", flush=True)

    # ---- (b) I2 com a validação como fonte de adaptação
    yt, grp_t = y[i_brs], np.array([am[ids[k]]['grupo'] for k in i_brs])
    p_b0_t = np.array([float(am[ids[k]]['pontuacao_exp01']) for k in i_brs])
    ruim_t = np.isin(np.array([am[ids[k]]['locutor'] for k in i_brs]), list(RUINS_TESTE))

    def medidas(s, mt):
        yy, g = yt[mt], grp_t[mt]
        r = {'eer': D.eer(s[yy == 1], s[yy == 0])[0], 'mindcf': I.min_dcf(s[yy == 1], s[yy == 0])[0],
             'por_tts': {t: D.eer(s[yy == 1], s[g == t])[0] for t in set(g) - {'bonafide'}}}
        mr = ruim_t[mt]
        r['eer_ruins'] = D.eer(s[mr & (yy == 1)], s[mr & (yy == 0)])[0]
        r['eer_bons'] = D.eer(s[~mr & (yy == 1)], s[~mr & (yy == 0)])[0]
        return r

    origem = {}
    refs_t = {'B0': p_b0_t}
    for r in I.REPRESENTACOES:
        Xo = X[r][i_tr]
        origem[('A_limpo', r)] = (Xo, y[i_tr])
        origem[('A_d', r)] = (np.concatenate([Xo, zr[f'd__{r}'].astype(np.float32)]), np.concatenate([y[i_tr], y[i_tr]]))
        refs_t[f'B2@{r}'] = P.classificador(C['limpo'][r]).fit(*origem[('A_limpo', r)]).decision_function(X[r][i_brs])
        refs_t[f'I1_d@{r}'] = P.classificador(C['d'][r]).fit(*origem[('A_d', r)]).decision_function(X[r][i_brs])
    Cm = {'A_limpo': C['limpo'], 'A_d': C['d']}

    por_sorteio, pulados = [], {k: 0 for k in args.k}
    for j in range(args.sorteios):
        adapt = set(Q.COMBOS[j % len(Q.COMBOS)])
        mt = (yt == 1) | ~np.isin(grp_t, list(adapt))  # teste: todos os locutores; falsos só dos 2 TTS fora
        pool_b = np.flatnonzero(yv == 1)
        pool_s = np.flatnonzero((yv == 0) & np.isin(grp_v, list(adapt)))
        linha = {'j': j, 'tts_teste': sorted(set(Q.TTS) - adapt),
                 'referencias': {n: medidas(p[mt], mt) for n, p in refs_t.items()}, 'k': {}}
        for k in args.k:
            if pool_b.size < k or pool_s.size < k:
                pulados[k] += 1
                continue
            rng = np.random.default_rng([args.semente, 4, j, k])
            alvo = np.concatenate([rng.choice(pool_b, k, replace=False), rng.choice(pool_s, k, replace=False)])
            res_k = {}
            for r in I.REPRESENTACOES:
                Xa, ya = Xv[r][alvo], yv[alvo]
                Xt = X[r][i_brs][mt]
                res_k[f'B_alvo@{r}'] = medidas(P.classificador(Q.C_ALVO).fit(Xa, ya).decision_function(Xt), mt)
                for m in ('A_limpo', 'A_d'):
                    Xo, yo = origem[(m, r)]
                    w = Q.pesos(np.concatenate([np.zeros(yo.size), np.ones(ya.size)]), np.concatenate([yo, ya]))
                    clf = Q.probe_ponderado(Cm[m][r]).fit(np.concatenate([Xo, Xa]), np.concatenate([yo, ya]),
                                                          logisticregression__sample_weight=w)
                    res_k[f'{m}@{r}'] = medidas(clf.decision_function(Xt), mt)
            linha['k'][k] = res_k
        por_sorteio.append(linha)
        print(f"sorteio {j:2d} teste TTS {linha['tts_teste']}: " + '  '.join(
            f"k={k}: " + ' '.join(f"{m}={D.pct(linha['k'][k][m + '@' + r0]['eer'], 1)}"
                                  for m, r0 in (('A_limpo', sel['limpo']), ('A_d', sel['d']), ('B_alvo', sel['limpo'])))
            for k in args.k if k in linha['k']) + f"  | B0={D.pct(linha['referencias']['B0']['eer'], 1)}", flush=True)

    def vals(chave, k, med='eer'):
        return [l['k'][k][chave][med] if k in l['k'] else None for l in por_sorteio]

    resumo_k = {}
    for k in args.k:
        rk = {'metodos': {}, 'diferencas': {}}
        for r in I.REPRESENTACOES:
            for m in ('A_limpo', 'A_d', 'B_alvo'):
                rk['metodos'][f'{m}@{r}'] = {med: Q.resumo(vals(f'{m}@{r}', k, med)) for med in ('eer', 'mindcf', 'eer_ruins', 'eer_bons')}
            for a, b in (('A_limpo', 'B_alvo'), ('A_d', 'A_limpo'), ('A_d', 'B_alvo')):
                rk['diferencas'][f'{a}-{b}@{r}'] = Q.resumo_dif([x - w if x is not None and w is not None else None
                                                               for x, w in zip(vals(f'{a}@{r}', k), vals(f'{b}@{r}', k))])
            for m in ('A_limpo', 'A_d', 'B_alvo'):
                v0 = [l['referencias']['B0']['eer'] for l in por_sorteio]
                rk['diferencas'][f'{m}-B0@{r}'] = Q.resumo_dif([x - w if x is not None else None for x, w in zip(vals(f'{m}@{r}', k), v0)])
        resumo_k[k] = rk
    ref_res = {n: {med: Q.resumo([l['referencias'][n][med] for l in por_sorteio]) for med in ('eer', 'mindcf', 'eer_ruins', 'eer_bons')}
               for n in refs_t}

    print('\n(b) I2 com adaptação na validação e teste no teste oficial (EER mediana [p10, p90]):')
    for n in ('B0', f"B2@{sel['limpo']}", 'B2@camada_5', f"I1_d@{sel['d']}"):
        e = ref_res[n]['eer']
        print(f"  k=0   {n:16s} {D.pct(e['mediana'])} [{D.pct(e['p10'])}, {D.pct(e['p90'])}]")
    for k in args.k:
        for r in (sel['limpo'], 'camada_5'):
            print(f'  k={k:<3d} {r:10s} ' + '  '.join(
                f"{m} {D.pct(resumo_k[k]['metodos'][f'{m}@{r}']['eer'].get('mediana'))}" for m in ('A_limpo', 'A_d', 'B_alvo')))

    res = {'desenho': {'tts': Q.TTS, 'sorteios': args.sorteios, 'k': args.k, 'C': C, 'representacao_principal': sel,
                       'C_forma_B': Q.C_ALVO, 'adaptacao': 'validação do BRSpeech-DF (10 locutores, disjuntos do teste)',
                       'teste': 'amostra do teste oficial da Fase 3, todos os locutores; falsos só dos 2 TTS fora da adaptação'},
           'conferencia_pareamento': conf, 'segundo_teste': res_a,
           'i2_validacao': {'pulados': pulados, 'referencias_k0': ref_res, 'resumo': resumo_k, 'por_sorteio': por_sorteio}}
    I.salvar_json(os.path.join(out, 'brs_val.json'), res, __file__,
                  {'k': args.k, 'sorteios': args.sorteios, 'B': args.B, 'semente': args.semente})


if __name__ == '__main__':
    main()
