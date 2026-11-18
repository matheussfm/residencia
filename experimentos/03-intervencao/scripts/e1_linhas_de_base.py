"""Etapa 1: linhas de base com o protocolo do Exp. 03 (sem GPU).

  B0  detector completo: pontuações do Exp. 01 (coluna pontuacao_exp01 do amostra.tsv da Fase 3);
  B1  detector + volume_padronizar: pontuações da Fase 2c do Exp. 02; a diferença pareada é medida
      contra a condição controle da Fase 2c (mesmo caminho de leitura e gravação);
  B2  probe limpo: o probe da Fase 3 (f3_probing.classificador), treinado só na partição de treino
      do df21_eval, com representação e C escolhidos pela validação da origem (README, seção 4.3).

Conjuntos de teste (ids da amostra da Fase 3):
  itw, brs, df21_hidden (completa), df21_hidden_teste (só os locutores da partição de teste) e
  df21_teste (partição de teste do df21_eval).

Para cada conjunto e método: EER e minDCF com IC de 95% por bootstrap de locutores, e diferença
pareada (B1 − controle, B1 − B0, B2 − B0). No BRSpeech-DF, também a EER nos 3 locutores "ruins" e
nos 7 "bons" (Exp. 02, seção 4.4).

Saídas: resultados/etapa1/linhas_de_base.json (+ .meta.json) e resultados/etapa1/pontuacoes_b2.tsv
(pontuação de cada probe B2 em todos os ids da amostra; reaproveitada nas etapas seguintes).
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _interv import D
import f3_probing as P  # noqa: E402  (Exp. 02; o mesmo probe da Fase 3)
from _comum import escrever_tsv  # noqa: E402

RUINS = {'5739', '13063', '12626'}
NORMALIZAR = os.path.join(D.EXP02, 'resultados', 'normalizar')


def conjuntos(am, div):
    """dict nome -> lista de ids da amostra (na ordem do amostra.tsv)."""
    ids = list(am)
    teste_locs = {am[i]['locutor'] for i in div if div[i] == 'teste'}
    return {
        'itw': [i for i in ids if am[i]['cenario'] == 'itw'],
        'brs': [i for i in ids if am[i]['cenario'] == 'brs'],
        'df21_hidden': [i for i in ids if am[i]['cenario'] == 'df21_hidden'],
        'df21_hidden_teste': [i for i in ids if am[i]['cenario'] == 'df21_hidden' and am[i]['locutor'] in teste_locs],
        'df21_teste': [i for i in ids if div.get(i) == 'teste'],
    }


def cenario(nome, ids, pont, am):
    return D.Cenario(nome, ids, [pont[i] for i in ids], [am[i]['rotulo'] == 'bonafide' for i in ids],
                     {'locutor': [am[i]['locutor'] for i in ids], 'grupo': [am[i]['grupo'] for i in ids]})


def pontuacoes_fase2c(condicao, am):
    """id da amostra -> pontuação da Fase 2c (no 2021 DF o id já tem o prefixo; nos outros, id_original)."""
    arquivo = {'itw': 'itw', 'brs': 'brs', 'df21_eval': 'df21', 'df21_hidden': 'df21'}
    lidos = {a: D.ler_pontuacoes(D._usar(os.path.join(NORMALIZAR, f'pontuacoes_{condicao}_{a}.txt')))
             for a in ('itw', 'brs', 'df21')}
    out = {}
    for i, r in am.items():
        p = lidos[arquivo[r['cenario']]]
        chave = i if r['cenario'].startswith('df21') else r['id_original']
        if chave in p:
            out[i] = p[chave]
    faltam = [i for i in am if i not in out]
    if faltam:
        D.erro(f'{len(faltam)} ids da amostra sem pontuação na condição {condicao} (ex.: {faltam[:3]})')
    return out


def representacoes(features, ids_npz):
    """Gera (nome, X float32) para as representações do pré-registro, uma por vez."""
    z = np.load(D._usar(features))
    camadas = P.memmap_npz(features, 'camadas')
    for nome in I.REPRESENTACOES:
        if nome == 'aasist':
            X = z['aasist']
        else:
            X = camadas[:, int(nome.split('_')[1]) - 1, :]
        yield nome, np.ascontiguousarray(X, dtype=np.float32)


def escolher(linhas):
    """Menor EER na validação; desempate pelo minDCF e depois pelo menor C."""
    return min(linhas, key=lambda r: (r['eer_validacao'], r['mindcf_validacao'], r['C']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', default=os.path.join(D.EXP02, 'resultados', 'fase3', 'features.npz'))
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--saida', default=None, help='pasta de saída (padrão: resultados/etapa1)')
    args = ap.parse_args()
    out = args.saida or I.pasta_saida('etapa1')
    D.garantir_fora_third_party(out)
    os.makedirs(out, exist_ok=True)

    am = I.ler_amostra()
    div = I.ler_divisao()
    conj = conjuntos(am, div)
    for k, v in conj.items():
        print(f'{k}: {len(v)} áudios, {len({am[i]["locutor"] for i in v})} locutores')

    # ---- B2: escolha na validação da origem
    z = np.load(D._usar(args.features))
    ids_npz = [str(i) for i in z['ids']]
    if set(ids_npz) != set(am):
        D.erro('ids do .npz e do amostra.tsv não coincidem')
    pos = {i: k for k, i in enumerate(ids_npz)}
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids_npz], int)
    i_tr = np.array([pos[i] for i in div if div[i] == 'treino'])
    i_va = np.array([pos[i] for i in div if div[i] == 'validacao'])

    selecao, pont_b2, escolhidos = [], {}, {}
    for nome, X in representacoes(args.features, ids_npz):
        linhas = []
        for C in I.GRADE_C:
            clf = P.classificador(C).fit(X[i_tr], y[i_tr])
            s = clf.decision_function(X[i_va])
            r = {'representacao': nome, 'C': C, 'eer_validacao': D.eer(s[y[i_va] == 1], s[y[i_va] == 0])[0],
                 'mindcf_validacao': I.min_dcf(s[y[i_va] == 1], s[y[i_va] == 0])[0]}
            linhas.append(r)
            print(f"  {nome:10s} C={C:<6g} validação: EER {D.pct(r['eer_validacao'])}  minDCF {r['mindcf_validacao']:.3f}", flush=True)
        melhor = escolher(linhas)
        escolhidos[nome] = melhor
        selecao += linhas
        clf = P.classificador(melhor['C']).fit(X[i_tr], y[i_tr])
        s = clf.decision_function(X)
        pont_b2[nome] = {i: float(s[pos[i]]) for i in ids_npz}
    selecionada = escolher(list(escolhidos.values()))['representacao']
    print(f'representação escolhida na validação: {selecionada} (C = {escolhidos[selecionada]["C"]})')

    tsv = os.path.join(out, 'pontuacoes_b2.tsv')
    escrever_tsv(tsv, ['id'] + I.REPRESENTACOES,
                 [[i] + [f'{pont_b2[n][i]:.6f}' for n in I.REPRESENTACOES] for i in ids_npz])

    # ---- B0, controle, B1
    metodos = {'B0': {i: float(r['pontuacao_exp01']) for i, r in am.items()},
               'controle_2c': pontuacoes_fase2c('controle', am),
               'B1': pontuacoes_fase2c('volume_padronizar', am)}
    metodos.update({f'B2_{n}': pont_b2[n] for n in I.REPRESENTACOES})
    comparacoes = [('B1', 'controle_2c'), ('B1', 'B0')] + [(f'B2_{n}', 'B0') for n in I.REPRESENTACOES]

    res = {'selecao_b2': {'grade': selecao, 'por_representacao': escolhidos, 'selecionada': selecionada},
           'conjuntos': {k: {'n_genuinos': sum(am[i]['rotulo'] == 'bonafide' for i in v),
                             'n_falsos': sum(am[i]['rotulo'] != 'bonafide' for i in v),
                             'locutores': len({am[i]['locutor'] for i in v})} for k, v in conj.items()},
           'metricas': {}, 'diferencas': {}, 'pontuacoes_b2_tsv_sha256': D.sha256(tsv)}
    for k, ids in conj.items():
        res['metricas'][k], res['diferencas'][k] = {}, {}
        for m, pont in metodos.items():
            c = cenario(k, ids, pont, am)
            d = I.metricas_com_ic(c, B=args.B, semente=args.semente)
            if k == 'brs':
                ruim = np.isin(c.meta['locutor'], list(RUINS))
                for rot, mk in (('ruins', ruim), ('bons', ~ruim)):
                    d[f'eer_{rot}'] = D.eer(c.pont[mk & (c.rotulo == 1)], c.pont[mk & (c.rotulo == 0)])[0]
            res['metricas'][k][m] = d
        for a, b in comparacoes:
            base = cenario(k, ids, metodos[b], am)
            res['diferencas'][k][f'{a}-{b}'] = I.diferenca_pareada(base, [metodos[a][i] for i in ids],
                                                                   B=args.B, semente=args.semente)
        print(f'\n== {k}')
        for m, d in res['metricas'][k].items():
            e, f = d['eer'], d['mindcf']
            extra = f"  ruins {D.pct(d['eer_ruins'])} bons {D.pct(d['eer_bons'])}" if k == 'brs' else ''
            print(f"  {m:16s} EER {D.pct(e['valor']):>7s} [{D.pct(e['ic95'][0])}, {D.pct(e['ic95'][1])}]  "
                  f"minDCF {f['valor']:.3f} [{f['ic95'][0]:.3f}, {f['ic95'][1]:.3f}]{extra}")
        for nome, d in res['diferencas'][k].items():
            e = d['eer']
            print(f"  Δ {nome:18s} EER {100 * e['valor']:+6.2f} pp [{100 * e['ic95'][0]:+.2f}, {100 * e['ic95'][1]:+.2f}]"
                  f"  minDCF {d['mindcf']['valor']:+.3f} [{d['mindcf']['ic95'][0]:+.3f}, {d['mindcf']['ic95'][1]:+.3f}]", flush=True)

    I.salvar_json(os.path.join(out, 'linhas_de_base.json'), res, __file__,
                  {'B': args.B, 'semente': args.semente, 'representacoes': I.REPRESENTACOES, 'grade_C': I.GRADE_C,
                   'criterio_escolha': 'menor EER na validação; desempate minDCF e menor C'})


if __name__ == '__main__':
    main()
