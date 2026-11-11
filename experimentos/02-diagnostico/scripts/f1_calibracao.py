"""Fase 1, análise B: o detector ainda ordena bem e só o limiar desloca (calibração)?

Separa duas perdas, para cada cenário:
  - perda de ORDENAÇÃO: medida pela EER e pelo minCllr (o melhor que qualquer limiar ou
    transformação monotônica consegue no próprio domínio);
  - perda de CALIBRAÇÃO: quanto se perde por usar o limiar, ou a calibração, aprendidos no
    domínio de treino (2021 DF).

Medidas:
  1. erro balanceado (média de genuínos rejeitados e falsos aceitos) no limiar da EER do 2021 DF,
     contra a EER do próprio domínio (o erro balanceado no melhor limiar é ~ a EER);
  2. Cllr real: calibração afim (regressão logística) ajustada no 2021 DF e aplicada ao cenário;
     minCllr: PAV no próprio cenário. Cllr − minCllr = perda de calibração (em bits);
     Cllr = 1 bit equivale a um detector que não informa nada;
  3. recalibração com poucos exemplos: ajusta a calibração afim com k genuínos e k falsos do
     cenário-alvo e mede o erro balanceado no limiar resultante (LLR = 0) nos demais áudios.
     Repete 200 vezes por k. Responde se "um pouco de dado do domínio" corrige o limiar.

Detectores: o SSL (XLS-R + AASIST) com as nossas pontuações em todos os cenários e, como
controle, SSL, SLS e AASIST com as pontuações publicadas pelo AKCIT (2021 DF e BRSpeech-DF; o
AKCIT não publica In-the-Wild).

Saída: resultados/fase1/calibracao.json.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D


def recalibracao_poucos(c, k, reps, rng):
    ib, is_ = np.flatnonzero(c.rotulo == 1), np.flatnonzero(c.rotulo == 0)
    if ib.size <= k or is_.size <= k:
        return None
    erros, degeneradas = [], 0
    for _ in range(reps):
        sb, ss = rng.choice(ib, k, replace=False), rng.choice(is_, k, replace=False)
        a, b = D.ajustar_calibracao(c.pont[sb], c.pont[ss])
        if not a > 0:  # amostra em que a calibração não ordena genuíno acima de falso
            degeneradas += 1
            continue
        limiar = -b / a
        resto = np.ones(len(c), bool)
        resto[sb] = resto[ss] = False
        pb, ps = c.pont[resto & (c.rotulo == 1)], c.pont[resto & (c.rotulo == 0)]
        erros.append(D.taxas_no_limiar(pb, ps, limiar)['erro_balanceado'])
    erros = np.array(erros)
    return {'k_por_classe': k, 'reps': reps, 'degeneradas': degeneradas, 'mediana': float(np.median(erros)),
            'p10': float(np.percentile(erros, 10)), 'p90': float(np.percentile(erros, 90))}


def avaliar(treino, alvos, ks, reps, rng):
    a, b = D.ajustar_calibracao(treino.bona, treino.spoof)
    _, t_treino = D.eer(treino.bona, treino.spoof)
    saida = {'calibracao_treino': {'a': a, 'b': b, 'limiar_llr0': -b / a}, 'limiar_eer_treino': t_treino, 'cenarios': {}}
    for nome, c in alvos.items():
        e, t = D.eer(c.bona, c.spoof)
        cl = D.cllr(a * c.bona + b, a * c.spoof + b)
        mcl = D.min_cllr(c.bona, c.spoof)
        no_treino = D.taxas_no_limiar(c.bona, c.spoof, t_treino)
        d = {
            'eer': e, 'limiar_eer': t, 'deslocamento_limiar': t - t_treino,
            'no_limiar_do_treino': no_treino,
            'perda_por_limiar_pp': 100 * (no_treino['erro_balanceado'] - e),
            'cllr_calibracao_do_treino': cl, 'min_cllr': mcl, 'perda_calibracao_bits': cl - mcl,
            'recalibracao_poucos_exemplos': [r for r in (recalibracao_poucos(c, k, reps, rng) for k in ks) if r],
        }
        saida['cenarios'][nome] = d
        rec = ' '.join(f"k={r['k_por_classe']}:{D.pct(r['mediana'], 1)}" for r in d['recalibracao_poucos_exemplos'])
        print(f"  {nome:22s} EER {D.pct(e)} | erro bal. no limiar do treino {D.pct(no_treino['erro_balanceado'])}"
              f" (FRR {D.pct(no_treino['genuinos_rejeitados'], 1)}, FAR {D.pct(no_treino['falsos_aceitos'], 1)})"
              f" | Cllr {cl:.3f} minCllr {mcl:.3f} | recal. {rec}")
    return saida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.semente)
    ks = [10, 50, 200]
    res = {}

    print('SSL (XLS-R + AASIST), nossas pontuações (repetição, corte de 4 s)')
    df21 = D.carregar_df21()
    res['ssl_nossas'] = avaliar(df21, {'df21': df21, 'itw': D.carregar_itw(), 'brs_teste': D.carregar_brs()},
                                ks, args.reps, rng)
    for det in ('ssl', 'sls', 'aasist'):
        print(f'{det}, pontuações publicadas pelo AKCIT')
        t = D.carregar_df21_akcit(det)
        res[f'{det}_akcit'] = avaliar(t, {'df21': t, 'brs_teste': D.carregar_brs_akcit(det, 'test'),
                                          'brs_todos': D.carregar_brs_akcit(det, 'todos')}, ks, args.reps, rng)

    D.salvar_json(f"{D.pasta_saida('fase1')}/calibracao.json", res, __file__,
                  {'reps': args.reps, 'semente': args.semente, 'k': ks})


if __name__ == '__main__':
    main()
