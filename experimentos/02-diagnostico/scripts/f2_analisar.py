"""Fase 2: análise pareada das pontuações dos áudios perturbados (saída do f2_perturbar.py).

Entradas: as pontuações do inferir.py (Experimento 01) sobre a lista.tsv do f2_perturbar.py e o
rotulos.tsv da mesma pasta (grupo = condição; id = "<condicao>__<trial>").

Para cada condição, comparando cada áudio com ele mesmo na condição "original":
  - deslocamento da pontuação (mediana e quantis 10/90), separado por classe;
  - fração de genuínos abaixo do limiar (rejeitados) e de falsos acima (aceitos), no limiar da
    EER do 2021 DF completo (Experimento 01) e no limiar da EER da condição original da amostra;
  - EER dos genuínos perturbados contra os falsos ORIGINAIS (só o genuíno mudou) e EER com as
    duas classes perturbadas;
  - IC de 95% do deslocamento mediano por bootstrap dos áudios (pareado).

Exemplo:
    python f2_analisar.py --pontuacoes ../resultados/fase2/pontuacoes.txt \
        --rotulos /dados/exp02_perturbacoes/rotulos.tsv --saida ../resultados/fase2/perturbacoes.json
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D

LIMIAR_DF21 = -3.5328  # limiar da EER do 2021 DF completo com as nossas pontuações (Experimento 01)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pontuacoes', required=True)
    ap.add_argument('--rotulos', required=True)
    ap.add_argument('--saida', required=True)
    ap.add_argument('--B', type=int, default=1000)
    args = ap.parse_args()
    D.garantir_fora_third_party(args.saida)
    rng = np.random.default_rng(0)

    pont = D.ler_pontuacoes(D._usar(args.pontuacoes))
    rot = D.ler_tsv(D._usar(args.rotulos))
    por = {}
    for r in rot:
        cond, trial = r['id'].split('__', 1)
        if r['id'] not in pont:
            D.erro(f"sem pontuação: {r['id']}")
        por.setdefault(cond, {})[trial] = (pont[r['id']], r['rotulo'] == 'bonafide')
    if 'original' not in por:
        D.erro('falta a condição original')
    orig = por['original']
    trials = sorted(orig)
    o = np.array([orig[t][0] for t in trials])
    g = np.array([orig[t][1] for t in trials])
    _, t_orig = D.eer(o[g], o[~g])

    res = {'limiar_df21': LIMIAR_DF21, 'limiar_eer_original': t_orig, 'condicoes': {}}
    for cond in sorted(por, key=lambda c: (c != 'original', c)):
        if set(por[cond]) != set(trials):
            D.erro(f'{cond}: conjunto de áudios diferente do original')
        p = np.array([por[cond][t][0] for t in trials])
        d = {}
        for nome, m in (('bonafide', g), ('spoof', ~g)):
            delta = p[m] - o[m]
            boot = [np.median(delta[rng.integers(0, delta.size, delta.size)]) for _ in range(args.B)]
            d[f'delta_{nome}'] = {'mediana': float(np.median(delta)), 'ic95_mediana': list(np.percentile(boot, [2.5, 97.5])),
                                  'p10': float(np.percentile(delta, 10)), 'p90': float(np.percentile(delta, 90))}
        for nome_t, t in (('limiar_df21', LIMIAR_DF21), ('limiar_original', t_orig)):
            d[nome_t] = {'genuinos_rejeitados': float((p[g] < t).mean()), 'falsos_aceitos': float((p[~g] >= t).mean())}
        d['eer_gen_perturbado_vs_fal_original'] = D.eer(p[g], o[~g])[0]
        d['eer_ambos_perturbados'] = D.eer(p[g], p[~g])[0]
        res['condicoes'][cond] = d
        print(f"{cond:20s} Δgen {d['delta_bonafide']['mediana']:+.2f}  Δfal {d['delta_spoof']['mediana']:+.2f}"
              f"  FRR {D.pct(d['limiar_df21']['genuinos_rejeitados'], 1)}  FAR {D.pct(d['limiar_df21']['falsos_aceitos'], 1)}"
              f"  EER(gen pert.) {D.pct(d['eer_gen_perturbado_vs_fal_original'], 1)}  EER(ambos) {D.pct(d['eer_ambos_perturbados'], 1)}")
    D.salvar_json(args.saida, res, __file__, {'B': args.B})


if __name__ == '__main__':
    main()
