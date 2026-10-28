"""EER e acurácia em limiar fixo a partir de pontuações e rótulos.

Convenção (a do Tak): pontuação maior => genuíno. No limiar t, prevê-se
genuíno se pontuação >= t e falso se pontuação < t (como no notebook do AKCIT).

Entradas:
  --pontuacoes  arquivo "id pontuacao" (ou .pkl do AKCIT)
  --rotulos     TSV com colunas id, rotulo (bonafide/spoof) e, opcionalmente, grupo
Saída: métricas no terminal e em JSON (--saida).

Com --por-grupo, calcula também a EER de todos os genuínos contra os falsos de
cada grupo (ex.: cada TTS do BRSpeech-DF) e a acurácia de cada grupo no limiar.

Exemplo:
    python metricas.py --pontuacoes resultados/brspeech_test/pontuacoes_repeat.txt \
        --rotulos listas/brspeech_test/rotulos.tsv --por-grupo \
        --saida resultados/brspeech_test/metricas_repeat.json
"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

import numpy as np

from _comum import ROTULOS_VALIDOS, erro, garantir_fora_third_party, ler_pontuacoes, ler_tsv, resumo

LIMIAR_AKCIT = -3.5327  # limiar da EER do modelo no ASVspoof 2021 DF (usado no artigo do BRSpeech-DF)


# ---------------------------------------------------------------------------
# Copiado de third_party/SSL_Anti-spoofing/eval_metrics_DF.py, sem alterações.
# MIT License, Copyright (c) 2022 Hemlata (https://github.com/TakHemlata/SSL_Anti-spoofing)
def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
# ---------------------------------------------------------------------------


def no_limiar(bona, spoof, t):
    """Taxas no limiar t (genuíno se pontuação >= t)."""
    acertos = (bona >= t).sum() + (spoof < t).sum()
    return {
        'limiar': t,
        'acuracia': float(acertos / (bona.size + spoof.size)),
        'falsos_aceitos': float((spoof >= t).mean()),     # falso classificado como genuíno
        'genuinos_rejeitados': float((bona < t).mean()),  # genuíno classificado como falso
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pontuacoes', required=True)
    ap.add_argument('--rotulos', required=True)
    ap.add_argument('--limiar', type=float, action='append',
                    help=f'limiar(es) para acurácia; padrão {LIMIAR_AKCIT}. Pode repetir.')
    ap.add_argument('--por-grupo', action='store_true', help='métricas por valor da coluna grupo')
    ap.add_argument('--permitir-extras', action='store_true',
                    help='aceita pontuações sem rótulo (ex.: fases progress/hidden do 2021 DF)')
    ap.add_argument('--saida', help='JSON com as métricas')
    args = ap.parse_args()
    limiares = args.limiar or [LIMIAR_AKCIT]
    if args.saida:
        garantir_fora_third_party(args.saida)

    pont = ler_pontuacoes(args.pontuacoes)
    rot = ler_tsv(args.rotulos)
    ids_rot = {r['id'] for r in rot}
    sem_pont = [r['id'] for r in rot if r['id'] not in pont]
    if sem_pont:
        erro(f'{len(sem_pont)} ids rotulados sem pontuação (ex.: {sem_pont[:3]})')
    extras = len(set(pont) - ids_rot)
    if extras and not args.permitir_extras:
        erro(f'{extras} pontuações sem rótulo (use --permitir-extras se for esperado)')
    invalidos = {r['rotulo'] for r in rot} - set(ROTULOS_VALIDOS)
    if invalidos:
        erro(f'rótulos inválidos: {invalidos}')

    bona = np.array([pont[r['id']] for r in rot if r['rotulo'] == 'bonafide'])
    spoof = np.array([pont[r['id']] for r in rot if r['rotulo'] == 'spoof'])
    eer, limiar_eer = compute_eer(bona, spoof)
    res = {
        'pontuacoes': args.pontuacoes,
        'rotulos': args.rotulos,
        'n_bonafide': int(bona.size),
        'n_spoof': int(spoof.size),
        'n_extras_ignorados': extras,
        'eer': float(eer),
        'limiar_eer': float(limiar_eer),
        'media_bonafide': float(bona.mean()),
        'media_spoof': float(spoof.mean()),
        'limiares': [no_limiar(bona, spoof, t) for t in limiares],
    }

    if args.por_grupo:
        grupos = {}
        for r in rot:
            grupos.setdefault(r.get('grupo') or '-', []).append(r)
        res['grupos'] = {}
        for g, rs in sorted(grupos.items()):
            gb = np.array([pont[r['id']] for r in rs if r['rotulo'] == 'bonafide'])
            gs = np.array([pont[r['id']] for r in rs if r['rotulo'] == 'spoof'])
            d = {'n_bonafide': int(gb.size), 'n_spoof': int(gs.size)}
            if gs.size:  # grupo de falsos: EER contra todos os genuínos
                d['eer_vs_todos_genuinos'] = float(compute_eer(bona, gs)[0])
            d['acuracia_no_limiar'] = {str(t): float(((gb >= t).sum() + (gs < t).sum()) / max(gb.size + gs.size, 1))
                                       for t in limiares}
            res['grupos'][g] = d

    if args.saida:
        with open(args.saida, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    itens = [
        ('genuínos / falsos', f"{res['n_bonafide']} / {res['n_spoof']}"),
        ('EER', f'{100 * eer:.2f}%  (limiar {limiar_eer:.4f})'),
        ('média genuínos / falsos', f"{res['media_bonafide']:.3f} / {res['media_spoof']:.3f}"),
    ]
    for m in res['limiares']:
        itens.append((f"acurácia em {m['limiar']}", f"{100 * m['acuracia']:.2f}%"))
    for g, d in res.get('grupos', {}).items():
        if 'eer_vs_todos_genuinos' in d:
            itens.append((f'EER genuínos vs {g} (n={d["n_spoof"]})', f"{100 * d['eer_vs_todos_genuinos']:.2f}%"))
    if extras:
        itens.append(('pontuações sem rótulo ignoradas', extras))
    resumo('metricas.py', itens)

    if eer > 0.5:
        print('\nAVISO: EER acima de 50%. A orientação das pontuações provavelmente está invertida '
              '(esperado: maior = genuíno). Confira antes de usar este resultado.')


if __name__ == '__main__':
    main()
