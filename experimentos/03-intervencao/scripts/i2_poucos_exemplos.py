"""Etapa 3 (I2): poucos exemplos em português, com locutores e TTS fora da adaptação.

Só o BRSpeech-DF (amostra da Fase 3: 1.320 genuínos e 1.320 falsos, 264 por TTS, 10 locutores).

Sorteios (os mesmos para todos os métodos e todos os k, de modo que as comparações são pareadas):
  - o sorteio j usa a combinação j mod 10 de 3 dos 5 TTS para a adaptação (rodízio: cada uma das
    10 combinações aparece 2 vezes em 20 sorteios); os outros 2 TTS ficam para o teste;
  - os 10 locutores são divididos ao acaso em 5 de adaptação e 5 de teste;
  - adaptação: k genuínos e k falsos sorteados entre os locutores de adaptação, os falsos só dos 3
    TTS de adaptação; teste: todos os genuínos dos locutores de teste e os falsos desses locutores
    feitos pelos 2 TTS de teste. Nenhum locutor nem TTS aparece nos dois lados.

Métodos (README, seção 4.4):
  A_limpo  probe treinado com a partição de treino da origem (limpa) + os k exemplos do alvo;
  A_d      o mesmo, com a origem do I1 (d): treino limpo + as cópias aumentadas (d);
  B_alvo   probe treinado só com os k exemplos do alvo (C = 0,1, como no Exp. 02).
  Referências nos mesmos subconjuntos de teste: o detector completo (B0) e os probes de origem sem
  nenhum exemplo do alvo (k = 0: B2 e I1_d).
Na forma A, os pesos dos exemplos fazem cada célula (domínio × classe) somar 1/4 do peso total, ou
seja, o alvo pesa, junto, o mesmo que a origem, e as classes pesam igual nos dois domínios. O peso
total é igual ao número de exemplos, como no class_weight='balanced' do probe da Fase 3, para que o C
tenha a mesma escala. O C da forma A é o escolhido na validação da origem para aquele probe e aquela
representação (Etapas 1 e 2). Todas as representações do pré-registro são avaliadas; a principal de
cada método é a escolhida na validação da origem.

Medidas por sorteio: EER e minDCF no subconjunto de teste, e a EER por TTS de teste. Resumo por
(método, representação, k): mediana, p10 e p90 sobre os sorteios; diferenças pareadas por sorteio
(A_limpo − B_alvo, A_d − A_limpo, A_d − B_alvo, cada método − B0), com mediana, p10, p90 e a fração de
sorteios em que a diferença é negativa. A incerteza aqui vem dos sorteios de locutores e TTS, não de
um bootstrap.

Entradas: features.npz da Fase 3 (Exp. 02), resultados/etapa1/linhas_de_base.json e, para A_d,
resultados/etapa2/i1.json e resultados/etapa2/i1_treino_reps.npz. Sem os da Etapa 2, roda só
A_limpo e B_alvo (--sem-d).

Saída: resultados/etapa3/i2.json (+ .meta.json).
"""
import argparse
import itertools
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _interv import D
import f3_probing as P  # noqa: E402

TTS = ['f5tts', 'fish-speech', 'toucantts', 'xtts', 'yourtts']
COMBOS = list(itertools.combinations(TTS, 3))  # 10 combinações de adaptação
C_ALVO = 0.1


def probe_ponderado(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def pesos(dominio, y):
    """Cada célula (domínio × classe) soma 1/4 do total; total = número de exemplos."""
    w = np.zeros(y.size)
    for d in np.unique(dominio):
        for c in (0, 1):
            m = (dominio == d) & (y == c)
            if not m.any():
                D.erro('célula domínio × classe vazia nos pesos da forma A')
            w[m] = 1.0 / (4 * m.sum())
    return w * y.size


def resumo(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {'n': 0}
    return {'mediana': float(np.median(v)), 'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90)),
            'n': int(v.size)}


def resumo_dif(v):
    r = resumo(v)
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size:
        r['fracao_negativa'] = float(np.mean(v < 0))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', default=os.path.join(D.EXP02, 'resultados', 'fase3', 'features.npz'))
    ap.add_argument('--etapa1', default=os.path.join(I.EXP03, 'resultados', 'etapa1', 'linhas_de_base.json'))
    ap.add_argument('--i1', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1.json'))
    ap.add_argument('--i1-reps', default=os.path.join(I.EXP03, 'resultados', 'etapa2', 'i1_treino_reps.npz'))
    ap.add_argument('--sem-d', action='store_true', help='roda sem A_d (antes da Etapa 2)')
    ap.add_argument('--k', type=int, nargs='+', default=[10, 50, 200])
    ap.add_argument('--sorteios', type=int, default=20)
    ap.add_argument('--representacoes', nargs='+', default=I.REPRESENTACOES, choices=I.REPRESENTACOES)
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--saida', default=None, help='pasta de saída (padrão: resultados/etapa3)')
    args = ap.parse_args()
    out = args.saida or I.pasta_saida('etapa3')
    D.garantir_fora_third_party(out)
    os.makedirs(out, exist_ok=True)

    am, div = I.ler_amostra(), I.ler_divisao()
    z = np.load(D._usar(args.features))
    ids = [str(i) for i in z['ids']]
    if set(ids) != set(am):
        D.erro('ids do .npz e do amostra.tsv não coincidem')
    pos = {i: k for k, i in enumerate(ids)}
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    loc = np.array([am[i]['locutor'] for i in ids])
    grp = np.array([am[i]['grupo'] for i in ids])
    p_b0 = np.array([float(am[i]['pontuacao_exp01']) for i in ids])
    treino = [i for i in div if div[i] == 'treino']  # mesma ordem das Etapas 1 e 2
    i_tr = np.array([pos[i] for i in treino])
    i_brs = np.flatnonzero(np.array([am[i]['cenario'] == 'brs' for i in ids]))
    if set(grp[i_brs]) != set(TTS) | {'bonafide'}:
        D.erro(f'grupos inesperados no BRSpeech-DF: {sorted(set(grp[i_brs]))}')
    locs = np.unique(loc[i_brs])

    e1 = json.load(open(D._usar(args.etapa1), encoding='utf-8'))
    escolha = {'A_limpo': {r: e1['selecao_b2']['por_representacao'][r]['C'] for r in I.REPRESENTACOES}}
    sel = {'A_limpo': e1['selecao_b2']['selecionada']}
    metodos = ['A_limpo', 'B_alvo']
    reps_d = {}
    if not args.sem_d:
        i1 = json.load(open(D._usar(args.i1), encoding='utf-8'))
        escolha['A_d'] = {r: i1['selecao']['por_metodo_e_representacao']['d'][r]['C'] for r in I.REPRESENTACOES}
        sel['A_d'] = i1['selecao']['selecionada']['d']
        zr = np.load(D._usar(args.i1_reps))
        if [str(i) for i in zr['ids_treino']] != treino:
            D.erro('ids_treino do i1_treino_reps.npz não seguem a ordem da partição de treino')
        reps_d = {r: zr[f'd__{r}'].astype(np.float32) for r in args.representacoes}
        metodos.append('A_d')
    principal = {m: sel.get(m, sel['A_limpo']) for m in metodos}  # B_alvo: a mesma do probe limpo
    if any(principal[m] not in args.representacoes for m in metodos):
        D.erro(f'a representação principal {principal} precisa estar em --representacoes')
    print('C da forma A por representação:', escolha, '| representação principal:', principal)

    camadas = P.memmap_npz(args.features, 'camadas')
    X = {}
    for r in args.representacoes:
        X[r] = np.ascontiguousarray(z['aasist'] if r == 'aasist' else camadas[:, int(r.split('_')[1]) - 1, :], dtype=np.float32)
    del camadas

    # sorteios fixos: divisão de TTS e de locutores, depois os k exemplos (aninhados não; um sorteio por k)
    sorteios = []
    for j in range(args.sorteios):
        rng = np.random.default_rng([args.semente, j])
        adapt_tts = set(COMBOS[j % len(COMBOS)])
        loc_a = set(rng.choice(locs, locs.size // 2, replace=False))
        em_a = np.isin(loc[i_brs], list(loc_a))
        pool_b = i_brs[em_a & (y[i_brs] == 1)]
        pool_s = i_brs[em_a & (y[i_brs] == 0) & np.isin(grp[i_brs], list(adapt_tts))]
        teste = i_brs[~em_a & ((y[i_brs] == 1) | ((y[i_brs] == 0) & ~np.isin(grp[i_brs], list(adapt_tts))))]
        sorteios.append({'j': j, 'tts_adaptacao': sorted(adapt_tts), 'tts_teste': sorted(set(TTS) - adapt_tts),
                         'locutores_adaptacao': sorted(loc_a), 'pool_b': pool_b, 'pool_s': pool_s, 'teste': teste})

    def medidas(s, idx):
        yy = y[idx]
        r = {'eer': D.eer(s[yy == 1], s[yy == 0])[0], 'mindcf': I.min_dcf(s[yy == 1], s[yy == 0])[0], 'por_tts': {}}
        for t in set(grp[idx]) - {'bonafide'}:
            m = grp[idx] == t
            r['por_tts'][t] = D.eer(s[yy == 1], s[m])[0]
        return r

    # referências sem exemplos do alvo, nos mesmos subconjuntos de teste
    refs = {'B0': p_b0}
    origem = {}
    for r in args.representacoes:
        Xo = X[r][i_tr]
        origem[('A_limpo', r)] = (Xo, y[i_tr])
        refs[f'B2@{r}'] = P.classificador(escolha['A_limpo'][r]).fit(Xo, y[i_tr]).decision_function(X[r])
        if 'A_d' in metodos:
            Xod = np.concatenate([Xo, reps_d[r]])
            yod = np.concatenate([y[i_tr], y[i_tr]])
            origem[('A_d', r)] = (Xod, yod)
            refs[f'I1_d@{r}'] = P.classificador(escolha['A_d'][r]).fit(Xod, yod).decision_function(X[r])

    por_sorteio = []
    pulados = {k: 0 for k in args.k}
    for s in sorteios:
        linha = {'j': s['j'], 'tts_teste': s['tts_teste'], 'locutores_adaptacao': s['locutores_adaptacao'],
                 'n_teste': [int((y[s['teste']] == 1).sum()), int((y[s['teste']] == 0).sum())],
                 'referencias': {n: medidas(p[s['teste']], s['teste']) for n, p in refs.items()}, 'k': {}}
        for k in args.k:
            if s['pool_b'].size < k or s['pool_s'].size < k:
                pulados[k] += 1
                continue
            rng = np.random.default_rng([args.semente, s['j'], k])
            alvo = np.concatenate([rng.choice(s['pool_b'], k, replace=False), rng.choice(s['pool_s'], k, replace=False)])
            res_k = {}
            for r in args.representacoes:
                Xa, ya = X[r][alvo], y[alvo]
                res_k[f'B_alvo@{r}'] = medidas(P.classificador(C_ALVO).fit(Xa, ya).decision_function(X[r][s['teste']]), s['teste'])
                for m in [x for x in metodos if x.startswith('A_')]:
                    Xo, yo = origem[(m, r)]
                    Xt, yt = np.concatenate([Xo, Xa]), np.concatenate([yo, ya])
                    w = pesos(np.concatenate([np.zeros(yo.size), np.ones(ya.size)]), yt)
                    clf = probe_ponderado(escolha[m][r]).fit(Xt, yt, logisticregression__sample_weight=w)
                    res_k[f'{m}@{r}'] = medidas(clf.decision_function(X[r][s['teste']]), s['teste'])
            linha['k'][k] = res_k
        por_sorteio.append(linha)
        print(f"sorteio {s['j']:2d} teste TTS {s['tts_teste']}: " + '  '.join(
            f"k={k}: " + ' '.join(f"{m}={D.pct(linha['k'][k][m + '@' + principal[m]]['eer'], 1)}" for m in metodos)
            for k in args.k if k in linha['k']) + f"  | B0={D.pct(linha['referencias']['B0']['eer'], 1)}", flush=True)

    # resumo
    def valores(chave, k, medida='eer'):
        return [l['k'][k][chave][medida] if k in l['k'] else None for l in por_sorteio]

    def valores_ref(chave, medida='eer'):
        return [l['referencias'][chave][medida] for l in por_sorteio]

    resumo_k = {}
    for k in args.k:
        rk = {'metodos': {}, 'diferencas': {}}
        for r in args.representacoes:
            for m in metodos:
                rk['metodos'][f'{m}@{r}'] = {med: resumo(valores(f'{m}@{r}', k, med)) for med in ('eer', 'mindcf')}
            pares = [('A_limpo', 'B_alvo')] + ([('A_d', 'A_limpo'), ('A_d', 'B_alvo')] if 'A_d' in metodos else [])
            for a, b in pares:
                va, vb = valores(f'{a}@{r}', k), valores(f'{b}@{r}', k)
                rk['diferencas'][f'{a}-{b}@{r}'] = resumo_dif([x - y_ if x is not None and y_ is not None else None
                                                              for x, y_ in zip(va, vb)])
            for m in metodos:
                va, v0 = valores(f'{m}@{r}', k), valores_ref('B0')
                rk['diferencas'][f'{m}-B0@{r}'] = resumo_dif([x - y_ if x is not None else None for x, y_ in zip(va, v0)])
            # EER por TTS de teste (em quantos sorteios cada TTS ficou fora da adaptação)
            for m in metodos:
                por_tts = {t: [] for t in TTS}
                for l in por_sorteio:
                    if k in l['k']:
                        for t, e in l['k'][k][f'{m}@{r}']['por_tts'].items():
                            por_tts[t].append(e)
                rk['metodos'][f'{m}@{r}']['eer_por_tts_teste'] = {t: resumo(v) for t, v in por_tts.items()}
        resumo_k[k] = rk
    ref_res = {n: {med: resumo(valores_ref(n, med)) for med in ('eer', 'mindcf')} for n in refs}

    res = {'desenho': {'tts': TTS, 'combinacoes_adaptacao': [list(c) for c in COMBOS], 'sorteios': args.sorteios,
                       'k': args.k, 'C_forma_A': escolha, 'C_forma_B': C_ALVO, 'representacao_principal': sel,
                       'pesos_forma_A': 'cada célula domínio × classe soma 1/4; total = n'},
           'pulados_por_falta_de_exemplos': pulados, 'referencias_k0': ref_res, 'resumo': resumo_k,
           'por_sorteio': por_sorteio}
    I.salvar_json(os.path.join(out, 'i2.json'), res, __file__,
                  {'k': args.k, 'sorteios': args.sorteios, 'semente': args.semente, 'representacoes': args.representacoes,
                   'metodos': metodos, 'C_forma_B': C_ALVO})

    print('\nResumo (EER mediana [p10, p90] sobre os sorteios; representação principal de cada método):')
    for n in ['B0', f"B2@{sel['A_limpo']}"] + ([f"I1_d@{sel['A_d']}"] if 'A_d' in metodos else []):
        e = ref_res[n]['eer']
        print(f"  k=0  {n:18s} {D.pct(e['mediana'])} [{D.pct(e['p10'])}, {D.pct(e['p90'])}]")
    for k in args.k:
        for m in metodos:
            r = principal[m]
            e = resumo_k[k]['metodos'][f'{m}@{r}']['eer']
            if e.get('n'):
                print(f"  k={k:<3d} {m + '@' + r:18s} {D.pct(e['mediana'])} [{D.pct(e['p10'])}, {D.pct(e['p90'])}] (n={e['n']})")
        for nome, d in resumo_k[k]['diferencas'].items():
            if nome.endswith('@' + sel['A_limpo']) and d.get('n'):
                print(f"         Δ {nome:26s} mediana {100 * d['mediana']:+.1f} pp [{100 * d['p10']:+.1f}, {100 * d['p90']:+.1f}]"
                      f"  negativa em {100 * d['fracao_negativa']:.0f}% dos sorteios")
    print('pulados por falta de exemplos:', pulados)


if __name__ == '__main__':
    main()
