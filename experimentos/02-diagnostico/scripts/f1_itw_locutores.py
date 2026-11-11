"""Fase 1, análise D: In-the-Wild por locutor (reproduz a análise citada no Termo da Semana 8).

Para cada um dos 54 locutores, no limiar da EER global do In-the-Wild:
  - genuínos rejeitados (FRR) e falsos aceitos (FAR), com IC de 95% por bootstrap dos áudios
    do locutor (Wilson seria parecido; o bootstrap mantém o mesmo método das outras análises);
  - EER do próprio locutor (seus genuínos contra seus falsos), que não depende do limiar;
  - pontuação média dos genuínos.
Concentração: que fração dos genuínos rejeitados vem dos k locutores com mais erros.
Sensibilidade: EER global sem os k piores locutores (k = 1, 3, 5). É um teste de sensibilidade,
não uma correção; retirar os piores locutores é escolher o resultado.

Também repete a concentração no limiar da EER do 2021 DF, para ver se a lista muda.

Saídas: resultados/fase1/itw_locutores.json, itw_locutores.tsv e figuras/itw_locutores.png.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F
from _comum import escrever_tsv


def ic_taxa(x, B, rng):
    if x.size == 0:
        return [float('nan')] * 2
    m = rng.integers(0, x.size, (B, x.size))
    v = x[m].mean(axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def concentracao(erros_por_loc):
    total = sum(erros_por_loc.values())
    ordem = sorted(erros_por_loc.items(), key=lambda kv: -kv[1])
    acum, out = 0, []
    for i, (loc, n) in enumerate(ordem, 1):
        acum += n
        out.append({'k': i, 'locutor': loc, 'erros': int(n), 'fracao_acumulada': acum / total if total else 0})
    k_metade = next((o['k'] for o in out if o['fracao_acumulada'] >= 0.5), None)
    return {'total_erros': int(total), 'k_para_metade': k_metade, 'ordem': out[:15]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--B', type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    c = D.carregar_itw()
    e_glob, t = D.eer(c.bona, c.spoof)

    linhas, por_loc = [], {}
    for loc in sorted(np.unique(c.meta['locutor'])):
        s = c.subconjunto(c.meta['locutor'] == loc)
        rej, ace = (s.bona < t).astype(float), (s.spoof >= t).astype(float)
        d = {'n_bonafide': int(s.bona.size), 'n_spoof': int(s.spoof.size),
             'frr': float(rej.mean()), 'frr_ic95': ic_taxa(rej, args.B, rng),
             'far': float(ace.mean()), 'far_ic95': ic_taxa(ace, args.B, rng),
             'genuinos_rejeitados': int(rej.sum()), 'falsos_aceitos': int(ace.sum()),
             'eer_proprio': D.eer(s.bona, s.spoof)[0],
             'media_bonafide': float(s.bona.mean()), 'media_spoof': float(s.spoof.mean()),
             'duracao_mediana_bonafide': float(np.median(s.duracao[s.rotulo == 1]))}
        por_loc[loc] = d
        linhas.append([loc, d['n_bonafide'], d['n_spoof'], f"{d['frr']:.4f}", f"{d['frr_ic95'][0]:.4f}",
                       f"{d['frr_ic95'][1]:.4f}", f"{d['far']:.4f}", f"{d['eer_proprio']:.4f}",
                       f"{d['media_bonafide']:.3f}", f"{d['duracao_mediana_bonafide']:.2f}"])

    conc_rej = concentracao({k: v['genuinos_rejeitados'] for k, v in por_loc.items()})
    conc_ace = concentracao({k: v['falsos_aceitos'] for k, v in por_loc.items()})
    sens = {}
    for k in (1, 3, 5):
        piores = [o['locutor'] for o in conc_rej['ordem'][:k]]
        s = c.subconjunto(~np.isin(c.meta['locutor'], piores))
        sens[k] = {'sem': piores, 'eer': D.eer(s.bona, s.spoof)[0]}

    # Relação entre a pontuação média dos genuínos de um locutor e a duração mediana deles
    from scipy.stats import spearmanr
    mb = np.array([v['media_bonafide'] for v in por_loc.values()])
    du = np.array([v['duracao_mediana_bonafide'] for v in por_loc.values()])
    rho = spearmanr(mb, du)

    res = {'eer_global': e_glob, 'limiar': t, 'locutores': por_loc,
           'concentracao_genuinos_rejeitados': conc_rej, 'concentracao_falsos_aceitos': conc_ace,
           'sensibilidade_sem_piores': sens,
           'spearman_media_genuinos_vs_duracao_mediana': {'rho': float(rho.correlation), 'p': float(rho.pvalue)}}
    out = D.pasta_saida('fase1')
    escrever_tsv(f'{out}/itw_locutores.tsv', ['locutor', 'n_bonafide', 'n_spoof', 'frr', 'frr_ic95_inf', 'frr_ic95_sup',
                                              'far', 'eer_proprio', 'media_bonafide', 'duracao_mediana_bonafide'],
                 sorted(linhas, key=lambda l: -float(l[3])))
    D.salvar_json(f'{out}/itw_locutores.json', res, __file__, {'B': args.B, 'limiar': 'EER global do In-the-Wild'})

    print(f'EER global {D.pct(e_glob)}, limiar {t:.4f}')
    print(f"genuínos rejeitados: {conc_rej['total_erros']}; metade vem de {conc_rej['k_para_metade']} locutores")
    for o in conc_rej['ordem'][:8]:
        d = por_loc[o['locutor']]
        print(f"  {o['locutor']:22s} {o['erros']:5d} erros ({D.pct(o['fracao_acumulada'], 1)} acum.)  "
              f"FRR {D.pct(d['frr'], 1)} de {d['n_bonafide']}  EER próprio {D.pct(d['eer_proprio'], 1)}")
    print(f"falsos aceitos: {conc_ace['total_erros']}; metade vem de {conc_ace['k_para_metade']} locutores")
    print('sem os piores:', {k: D.pct(v['eer']) for k, v in sens.items()})
    print('spearman média genuínos x duração:', res['spearman_media_genuinos_vs_duracao_mediana'])
    top = sorted(por_loc.items(), key=lambda kv: -kv[1]['frr'])
    print('maiores FRR (n>=30):', [(k, D.pct(v['frr'], 0), v['n_bonafide']) for k, v in top if v['n_bonafide'] >= 30][:10])

    # Figura: FRR por locutor com IC, ordenado; tamanho do ponto ~ n de genuínos
    it = sorted(por_loc.items(), key=lambda kv: kv[1]['frr'])
    y = np.arange(len(it))
    v = np.array([d['frr'] for _, d in it]) * 100
    lo = np.array([d['frr_ic95'][0] for _, d in it]) * 100
    hi = np.array([d['frr_ic95'][1] for _, d in it]) * 100
    fig, ax = F.plt.subplots(figsize=(6.5, 9))
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt='none', ecolor=F.EIXO, linewidth=1.5)
    ax.scatter(v, y, s=np.clip([d['n_bonafide'] / 8 for _, d in it], 10, 160), color=F.COR_GEN, zorder=3,
               edgecolor=F.FUNDO, linewidth=1)
    ax.axvline(100 * e_glob, color=F.TINTA2, linewidth=1, linestyle='--')
    ax.set_yticks(y, [f"{k} ({d['n_bonafide']})" for k, d in it], fontsize=7)
    ax.set_xlabel('genuínos rejeitados no limiar da EER global (%), IC de 95%')
    ax.set_title('In-the-Wild: genuínos rejeitados por locutor\n(entre parênteses, nº de genuínos; '
                 'área do ponto ~ nº de genuínos)', loc='left')
    F.salvar(fig, f"{D.pasta_saida('fase1', 'figuras')}/itw_locutores.png")


if __name__ == '__main__':
    main()
