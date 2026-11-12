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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pasta', default=os.path.join(D.EXP02, 'resultados', 'silencio'))
    ap.add_argument('--B', type=int, default=1000)
    args = ap.parse_args()
    res = {}
    for cen, carregar in (('itw', D.carregar_itw), ('brs', D.carregar_brs)):
        base = carregar()
        r = {'original': {'eer': D.com_ic(base, grupos='locutor', B=args.B),
                          'media_bonafide': float(base.bona.mean()), 'media_spoof': float(base.spoof.mean())}}
        for cond in ('aparar', 'padronizar'):
            cam = os.path.join(args.pasta, f'pontuacoes_{cond}_{cen}.txt')
            if not os.path.exists(cam):
                print(f'(sem {cam})')
                continue
            pont = D.ler_pontuacoes(D._usar(cam))
            c = D.Cenario(base.nome, base.ids, [pont[i] for i in base.ids], base.rotulo, base.meta, base.duracao)
            delta = c.pont - base.pont
            d = {'eer': D.com_ic(c, grupos='locutor', B=args.B),
                 'media_bonafide': float(c.bona.mean()), 'media_spoof': float(c.spoof.mean()),
                 'delta_mediano_bonafide': float(np.median(delta[c.rotulo == 1])),
                 'delta_mediano_spoof': float(np.median(delta[c.rotulo == 0])),
                 'auc': D.auc(c.bona, c.spoof)}
            if cen == 'brs':
                ruim = np.isin(c.meta['locutor'], list(RUINS))
                for nome, m in (('ruins', ruim), ('bons', ~ruim)):
                    d[f'eer_{nome}'] = D.eer(c.pont[m & (c.rotulo == 1)], c.pont[m & (c.rotulo == 0)])[0]
                    r['original'][f'eer_{nome}'] = D.eer(base.pont[m & (base.rotulo == 1)], base.pont[m & (base.rotulo == 0)])[0]
            r[cond] = d
        res[cen] = r
        print(f'\n== {cen}')
        for cond, d in r.items():
            ic = d['eer']['ic95']
            extra = (f"  ruins {D.pct(d.get('eer_ruins'))} bons {D.pct(d.get('eer_bons'))}" if cen == 'brs' else '')
            dl = (f"  Δgen {d['delta_mediano_bonafide']:+.2f} Δfal {d['delta_mediano_spoof']:+.2f}" if 'delta_mediano_bonafide' in d else '')
            print(f"  {cond:11s} EER {D.pct(d['eer']['valor'])} [{D.pct(ic[0])}, {D.pct(ic[1])}]  médias gen {d['media_bonafide']:+.2f}"
                  f" fal {d['media_spoof']:+.2f}{dl}{extra}")
    D.salvar_json(os.path.join(args.pasta, 'silencio_real.json'), res, __file__, {'B': args.B})


if __name__ == '__main__':
    main()
