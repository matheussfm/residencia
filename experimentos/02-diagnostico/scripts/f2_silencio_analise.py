"""Fase 2b: análise da normalização do silêncio em dados reais (saída do f2_silencio_real.py).

Compara, para In-the-Wild e BRSpeech-DF, as pontuações originais do Experimento 01 (repetição,
corte de 4 s) com as dos áudios sem silêncio nas pontas ("aparar") e com 0,5 s de silêncio
padronizado no início ("padronizar"):
  - EER com IC de 95% por bootstrap de locutores;
  - deslocamento mediano pareado (mesmo áudio) de genuínos e falsos;
  - BRSpeech-DF: EER nos 3 locutores "ruins" e nos 7 "bons".

Entradas: resultados/silencio/pontuacoes_<condicao>_<cenario>.txt (do inferir.py).
Saída: resultados/silencio/silencio_real.json.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D

RUINS = {'5739', '13063', '12626'}


def amostra_df21(cenario):
    """Trials do 2021 DF da amostra da Fase 3 (ids com prefixo), com as pontuações do Exp. 01."""
    am = [r for r in D.ler_tsv(D._usar(os.path.join(D.EXP02, 'listas', 'fase3', 'amostra.tsv'))) if r['cenario'] == cenario]
    return D.Cenario(cenario, [r['id'] for r in am], [float(r['pontuacao_exp01']) for r in am],
                     [r['rotulo'] == 'bonafide' for r in am], {'locutor': [r['locutor'] for r in am]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pasta', default=os.path.join(D.EXP02, 'resultados', 'silencio'))
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--condicoes', nargs='+', default=['aparar', 'padronizar'])
    ap.add_argument('--base', default='exp01', help="'exp01' (pontuações do Exp. 01) ou o nome de uma condição de controle")
    ap.add_argument('--saida', default='silencio_real.json')
    ap.add_argument('--cenarios', nargs='+', default=['itw', 'brs'], choices=['itw', 'brs', 'df21_eval', 'df21_hidden'])
    args = ap.parse_args()
    res = {}
    carregadores = {'itw': (D.carregar_itw, 'itw'), 'brs': (D.carregar_brs, 'brs'),
                    'df21_eval': (lambda: amostra_df21('df21_eval'), 'df21'),
                    'df21_hidden': (lambda: amostra_df21('df21_hidden'), 'df21')}
    for cen in args.cenarios:
        carregar, arq = carregadores[cen]
        base = carregar()
        if args.base != 'exp01':  # controle regravado: mesmos ids, pontuações da condição de controle
            pb = D.ler_pontuacoes(D._usar(os.path.join(args.pasta, f'pontuacoes_{args.base}_{arq}.txt')))
            exp01_eer = D.eer(base.bona, base.spoof)[0]
            base = D.Cenario(base.nome, base.ids, [pb[i] for i in base.ids], base.rotulo, base.meta, base.duracao)
            print(f'{cen}: base = {args.base} (EER do Exp. 01 nos mesmos áudios: {D.pct(exp01_eer)})')
        r = {'original' if args.base == 'exp01' else args.base: {'eer': D.com_ic(base, grupos='locutor', B=args.B),
                          'media_bonafide': float(base.bona.mean()), 'media_spoof': float(base.spoof.mean())}}
        for cond in [c for c in args.condicoes if c != args.base]:
            cam = os.path.join(args.pasta, f'pontuacoes_{cond}_{arq}.txt')
            if not os.path.exists(cam):
                print(f'(sem {cam})')
                continue
            pont = D.ler_pontuacoes(D._usar(cam))
            c = D.Cenario(base.nome, base.ids, [pont[i] for i in base.ids], base.rotulo, base.meta, base.duracao)
            delta = c.pont - base.pont
            # diferença de EER pareada: os mesmos locutores sorteados nas duas condições
            par = D.Cenario(base.nome, base.ids, c.pont, base.rotulo, dict(base.meta, p0=base.pont), base.duracao)
            dif = D.com_ic(par, estatistica=lambda k: D.eer(k.bona, k.spoof)[0] - D.eer(
                k.meta['p0'][k.rotulo == 1].astype(float), k.meta['p0'][k.rotulo == 0].astype(float))[0],
                grupos='locutor', B=args.B)
            d = {'eer': D.com_ic(c, grupos='locutor', B=args.B),
                 'media_bonafide': float(c.bona.mean()), 'media_spoof': float(c.spoof.mean()),
                 'delta_mediano_bonafide': float(np.median(delta[c.rotulo == 1])),
                 'delta_mediano_spoof': float(np.median(delta[c.rotulo == 0])),
                 'auc': D.auc(c.bona, c.spoof),
                 'diferenca_eer_pareada': dif}
            if cen == 'brs':
                ruim = np.isin(c.meta['locutor'], list(RUINS))
                for nome, m in (('ruins', ruim), ('bons', ~ruim)):
                    d[f'eer_{nome}'] = D.eer(c.pont[m & (c.rotulo == 1)], c.pont[m & (c.rotulo == 0)])[0]
                    r['original' if args.base == 'exp01' else args.base][f'eer_{nome}'] = D.eer(base.pont[m & (base.rotulo == 1)], base.pont[m & (base.rotulo == 0)])[0]
            r[cond] = d
        res[cen] = r
        print(f'\n== {cen}')
        for cond, d in r.items():
            ic = d['eer']['ic95']
            extra = (f"  ruins {D.pct(d.get('eer_ruins'))} bons {D.pct(d.get('eer_bons'))}" if cen == 'brs' else '')
            dl = (f"  Δgen {d['delta_mediano_bonafide']:+.2f} Δfal {d['delta_mediano_spoof']:+.2f}" if 'delta_mediano_bonafide' in d else '')
            dp = d.get('diferenca_eer_pareada')
            dps = (f"  Δ EER pareada {100 * dp['valor']:+.2f} pp [{100 * dp['ic95'][0]:+.2f}, {100 * dp['ic95'][1]:+.2f}]" if dp else '')
            print(f"  {cond:11s} EER {D.pct(d['eer']['valor'])} [{D.pct(ic[0])}, {D.pct(ic[1])}]{dps}  médias gen {d['media_bonafide']:+.2f}"
                  f" fal {d['media_spoof']:+.2f}{dl}{extra}")
    D.salvar_json(os.path.join(args.pasta, args.saida), res, __file__, {'B': args.B, 'condicoes': args.condicoes, 'base': args.base})


if __name__ == '__main__':
    main()
