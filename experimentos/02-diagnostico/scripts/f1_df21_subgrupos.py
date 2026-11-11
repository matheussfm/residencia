"""Fase 1, análise C: ASVspoof 2021 DF por subgrupo, e o efeito do silêncio (hidden track).

Usa a chave completa do 2021 DF distribuída pelo AKCIT (ver _diag.CHAVE_DF21).

1. Subgrupos da fase eval, com IC de 95% por bootstrap de locutores:
   - atributos das duas classes (codec, origem: asvspoof = VCTK/2019 LA, vcc2018, vcc2020):
     EER dentro do subgrupo, e taxa de genuínos rejeitados no limiar da EER global;
   - atributos só dos falsos (tipo de vocoder, ataque): EER de todos os genuínos contra os
     falsos do subgrupo, e taxa de falsos aceitos no limiar global.
2. Hidden track: a fase "hidden" tem os trials com o silêncio do início e do fim removido
   (coluna trim = trim). Compara com a fase eval (notrim) em três níveis:
   - geral;
   - por origem e por codec, dentro da mesma célula (controle de composição);
   - genuínos e falsos separados no limiar da EER do eval: quem se desloca sem o silêncio?
   Atenção: os trials da hidden track não são os mesmos áudios do eval com o silêncio cortado
   (não há como parear), então a comparação é entre conjuntos, não por arquivo.

Saídas: resultados/fase1/df21_subgrupos.json e figuras/df21_subgrupos.png.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F


def por_atributo(c, attr, t_global, ambos, B):
    out = {}
    for v in sorted(np.unique(c.meta[attr])):
        if v in ('-', 'bonafide'):
            continue
        m = c.meta[attr] == v
        if ambos:
            sub = c.subconjunto(m)
            d = {'n_bonafide': int(sub.rotulo.sum()), 'n_spoof': int((sub.rotulo == 0).sum())}
            d['eer'] = D.com_ic(sub, grupos='locutor', B=B)
            d['no_limiar_global'] = D.taxas_no_limiar(sub.bona, sub.spoof, t_global)
            d['media_bonafide'] = float(sub.bona.mean()) if sub.bona.size else None
        else:
            sub = c.subconjunto(m | (c.rotulo == 1))  # todos os genuínos + falsos do grupo
            d = {'n_bonafide': int(sub.rotulo.sum()), 'n_spoof': int((sub.rotulo == 0).sum())}
            d['eer'] = D.com_ic(sub, grupos='locutor', B=B)
            d['falsos_aceitos_no_limiar_global'] = float((sub.spoof >= t_global).mean())
        d['media_spoof'] = float(sub.spoof.mean())
        out[v] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--B', type=int, default=200)
    args = ap.parse_args()
    ev, hid = D.carregar_df21(fase='eval'), D.carregar_df21(fase='hidden')
    e_glob, t_glob = D.eer(ev.bona, ev.spoof)
    res = {'eer_eval': e_glob, 'limiar_eer_eval': t_glob, 'subgrupos': {}, 'hidden_track': {}}

    for attr, ambos in (('codec', True), ('origem', True), ('vocoder', False), ('ataque', False)):
        res['subgrupos'][attr] = por_atributo(ev, attr, t_glob, ambos, args.B)
        print(f'\n== {attr}')
        for v, d in res['subgrupos'][attr].items():
            extra = (f"FRR no limiar global {D.pct(d['no_limiar_global']['genuinos_rejeitados'])}" if ambos
                     else f"FAR no limiar global {D.pct(d['falsos_aceitos_no_limiar_global'])}")
            print(f"  {v:34s} n={d['n_bonafide']}/{d['n_spoof']}  EER {D.pct(d['eer']['valor'])}"
                  f" [{D.pct(d['eer']['ic95'][0])}, {D.pct(d['eer']['ic95'][1])}]  {extra}")

    # Hidden track
    h = res['hidden_track']
    h['composicao'] = {a: {'eval': dict(zip(*np.unique(ev.meta[a], return_counts=True))),
                           'hidden': dict(zip(*np.unique(hid.meta[a], return_counts=True)))}
                       for a in ('origem', 'codec', 'vocoder', 'trim')}
    h['geral'] = {'eval': D.com_ic(ev, grupos='locutor', B=args.B), 'hidden': D.com_ic(hid, grupos='locutor', B=args.B)}
    h['geral']['eval_no_limiar_eval'] = D.taxas_no_limiar(ev.bona, ev.spoof, t_glob)
    h['geral']['hidden_no_limiar_eval'] = D.taxas_no_limiar(hid.bona, hid.spoof, t_glob)
    h['media'] = {'eval': {'bonafide': float(ev.bona.mean()), 'spoof': float(ev.spoof.mean())},
                  'hidden': {'bonafide': float(hid.bona.mean()), 'spoof': float(hid.spoof.mean())}}
    h['cruzada'] = {'gen_hidden_vs_fal_eval': D.eer(hid.bona, ev.spoof)[0],
                    'gen_eval_vs_fal_hidden': D.eer(ev.bona, hid.spoof)[0]}
    print(f"\n== hidden track: EER eval {D.pct(h['geral']['eval']['valor'])} | hidden {D.pct(h['geral']['hidden']['valor'])}"
          f" {[D.pct(x) for x in h['geral']['hidden']['ic95']]}")
    print(f"   no limiar do eval: hidden FRR {D.pct(h['geral']['hidden_no_limiar_eval']['genuinos_rejeitados'])}"
          f" FAR {D.pct(h['geral']['hidden_no_limiar_eval']['falsos_aceitos'])}")
    print(f"   cruzada: gen hidden x fal eval {D.pct(h['cruzada']['gen_hidden_vs_fal_eval'])};"
          f" gen eval x fal hidden {D.pct(h['cruzada']['gen_eval_vs_fal_hidden'])}")
    print(f"   médias: {h['media']}")
    h['por_celula'] = {}
    for attr in ('origem', 'codec'):
        h['por_celula'][attr] = {}
        for v in sorted(set(np.unique(hid.meta[attr])) & set(np.unique(ev.meta[attr]))):
            se, sh = ev.subconjunto(ev.meta[attr] == v), hid.subconjunto(hid.meta[attr] == v)
            if not (sh.bona.size and sh.spoof.size):
                continue
            h['por_celula'][attr][v] = {
                'n_hidden': [int(sh.rotulo.sum()), int((sh.rotulo == 0).sum())],
                'eer_eval': D.eer(se.bona, se.spoof)[0], 'eer_hidden': D.eer(sh.bona, sh.spoof)[0],
                'frr_hidden_no_limiar_eval': float((sh.bona < t_glob).mean()),
                'far_hidden_no_limiar_eval': float((sh.spoof >= t_glob).mean()),
                'frr_eval_no_limiar_eval': float((se.bona < t_glob).mean()),
                'far_eval_no_limiar_eval': float((se.spoof >= t_glob).mean()),
            }
            d = h['por_celula'][attr][v]
            print(f"   {attr}={v:12s} EER eval {D.pct(d['eer_eval'])} -> hidden {D.pct(d['eer_hidden'])}"
                  f" | FRR {D.pct(d['frr_eval_no_limiar_eval'])} -> {D.pct(d['frr_hidden_no_limiar_eval'])}"
                  f" | FAR {D.pct(d['far_eval_no_limiar_eval'])} -> {D.pct(d['far_hidden_no_limiar_eval'])}")

    # Figura: EER por codec e por tipo de vocoder, com IC (pontos + barras de erro)
    fig, axs = F.plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, attr, tit in ((axs[0], 'codec', 'por codec'),
                          (axs[1], 'vocoder', 'por vocoder (contra todos os genuínos)')):
        it = sorted(res['subgrupos'][attr].items(), key=lambda kv: kv[1]['eer']['valor'])
        y = np.arange(len(it))
        v = np.array([d['eer']['valor'] for _, d in it]) * 100
        lo = np.array([d['eer']['ic95'][0] for _, d in it]) * 100
        hi = np.array([d['eer']['ic95'][1] for _, d in it]) * 100
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt='o', color=F.AZUL, ecolor=F.EIXO, markersize=6, capsize=0)
        ax.axvline(100 * e_glob, color=F.TINTA2, linewidth=1, linestyle='--')
        ax.set_yticks(y, [k.replace('_', ' ') for k, _ in it])
        ax.set_title(f'EER {tit}', loc='left')
        ax.set_xlabel('EER (%), IC de 95% por locutores')
    axs[0].annotate('EER global', xy=(100 * e_glob, len(res['subgrupos']['codec']) - 0.6), xytext=(4, 0),
                    textcoords='offset points', color=F.TINTA2, fontsize=8)
    F.salvar(fig, f"{D.pasta_saida('fase1', 'figuras')}/df21_subgrupos.png")

    D.salvar_json(f"{D.pasta_saida('fase1')}/df21_subgrupos.json", res, __file__, {'B': args.B})


if __name__ == '__main__':
    main()
