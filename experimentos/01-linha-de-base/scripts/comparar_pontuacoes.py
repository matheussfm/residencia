"""Compara, arquivo a arquivo, as nossas pontuações com as publicadas pelos autores.

Referências disponíveis nos submódulos (lidas, nunca alteradas):
  2021 DF:     third_party/SSL_Anti-spoofing/Scores/DF/Scores_DF.txt
  BRSpeech-DF: third_party/BRSpeech-DF-Dataset/data/scores/ssl/ssl_brspeech_full_scores.pkl
               third_party/BRSpeech-DF-Dataset/data/scores/ssl/ssl_sintetico_full_scores.pkl
  (--referencia pode ser repetido; os arquivos são unidos.)

Mede quanto as pontuações diferem e se a decisão no limiar muda. Serve para
confirmar que a nossa inferência reproduz a dos autores e, no BRSpeech-DF,
para descobrir se o AKCIT preencheu os áudios com repetição ou com zeros.

Exemplo:
    python comparar_pontuacoes.py --nossas resultados/asvspoof21df/pontuacoes_repeat.txt \
        --referencia ../../third_party/SSL_Anti-spoofing/Scores/DF/Scores_DF.txt
"""
import argparse
import json
import sys

sys.dont_write_bytecode = True

import numpy as np

from _comum import erro, garantir_fora_third_party, ler_pontuacoes, resumo

LIMIAR_AKCIT = -3.5327


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nossas', required=True)
    ap.add_argument('--referencia', required=True, action='append')
    ap.add_argument('--tolerancia', type=float, default=1e-3,
                    help='diferença absoluta considerada "igual" (padrão 1e-3)')
    ap.add_argument('--limiar', type=float, default=LIMIAR_AKCIT)
    ap.add_argument('--saida', help='JSON com o resultado')
    args = ap.parse_args()
    if args.saida:
        garantir_fora_third_party(args.saida)

    nossas = ler_pontuacoes(args.nossas)
    ref = {}
    for caminho in args.referencia:
        r = ler_pontuacoes(caminho)
        repetidos = set(r) & set(ref)
        if repetidos:
            erro(f'ids repetidos entre referências (ex.: {list(repetidos)[:3]})')
        ref.update(r)

    comuns = [k for k in nossas if k in ref]
    if not comuns:
        erro('nenhum id em comum: confira se os ids seguem o mesmo formato da referência')
    a = np.array([nossas[k] for k in comuns])
    b = np.array([ref[k] for k in comuns])
    dif = np.abs(a - b)
    decisao_igual = (a >= args.limiar) == (b >= args.limiar)
    res = {
        'nossas': args.nossas,
        'referencia': args.referencia,
        'n_comuns': len(comuns),
        'so_nossas': len(nossas) - len(comuns),
        'n_referencia': len(ref),
        'pearson': float(np.corrcoef(a, b)[0, 1]) if len(comuns) > 1 else None,
        'dif_abs_max': float(dif.max()),
        'dif_abs_media': float(dif.mean()),
        'tolerancia': args.tolerancia,
        'frac_acima_tolerancia': float((dif > args.tolerancia).mean()),
        'limiar': args.limiar,
        'frac_decisao_igual': float(decisao_igual.mean()),
        'piores': [{'id': comuns[i], 'nossa': float(a[i]), 'referencia': float(b[i])}
                   for i in np.argsort(-dif)[:5]],
    }
    if args.saida:
        with open(args.saida, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    if res['frac_acima_tolerancia'] == 0:
        veredito = 'IGUAIS dentro da tolerância'
    elif res['pearson'] is not None and res['pearson'] > 0.99:
        veredito = 'MUITO PRÓXIMAS, mas acima da tolerância (diferença numérica ou de pré-processamento?)'
    else:
        veredito = 'DIFERENTES'
    resumo('comparar_pontuacoes.py', [
        ('ids em comum', f"{res['n_comuns']} (só nossas: {res['so_nossas']}; referência tem {res['n_referencia']})"),
        ('correlação de Pearson', f"{res['pearson']:.6f}" if res['pearson'] is not None else '-'),
        ('diferença absoluta máx / média', f"{res['dif_abs_max']:.2e} / {res['dif_abs_media']:.2e}"),
        (f'acima da tolerância {args.tolerancia:g}', f"{100 * res['frac_acima_tolerancia']:.2f}%"),
        (f'mesma decisão no limiar {args.limiar}', f"{100 * res['frac_decisao_igual']:.2f}%"),
        ('veredito', veredito),
    ])


if __name__ == '__main__':
    main()
