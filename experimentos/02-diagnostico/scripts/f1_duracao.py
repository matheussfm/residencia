"""Fase 1, análise E: duração, janela de 4 s e preenchimento.

1. EER, genuínos rejeitados e falsos aceitos por faixa de duração, nos três cenários, com IC de
   95% por bootstrap de locutores. As taxas usam o limiar da EER global de cada cenário. Reproduz
   a análise do Termo da Semana 8 (BRSpeech-DF: abaixo de ~4 s, perto do acaso).
2. Atalho de duração: EER e AUC usando SÓ a duração como pontuação (maior = genuíno e o
   inverso; reporta o melhor sentido). Se a duração sozinha separa as classes, qualquer detector
   pode aprender esse atalho, e parte da EER pode refletir duração, não síntese.
3. EER dentro das faixas, com pesos que igualam a distribuição de duração das duas classes
   (média das EER das faixas ponderada pelo número de genuínos), como controle da diferença de
   duração entre classes no In-the-Wild.
4. Correlação de Spearman entre pontuação e duração, dentro de cada classe.
5. Preenchimento: nos áudios curtos (< 4,04 s), que são os únicos afetados, EER com repetição e
   com zeros; nos longos, as duas pontuações devem ser iguais (controle).
6. Corte: nos áudios longos, EER com e sem o corte de 4 s (o Experimento 01 só reportou o geral).

Saídas: resultados/fase1/duracao.json e figuras/duracao.png.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import spearmanr

import _diag as D
import _figuras as F

FAIXAS = [0, 2, 3, D.CORTE_S, 6, 10, 1e9]
ROTULOS = ['<2 s', '2–3 s', '3–4,04 s', '4,04–6 s', '6–10 s', '>10 s']


def por_faixa(c, t, B):
    out = []
    for i in range(len(FAIXAS) - 1):
        m = (c.duracao >= FAIXAS[i]) & (c.duracao < FAIXAS[i + 1])
        s = c.subconjunto(m)
        d = {'faixa': ROTULOS[i], 'n_bonafide': int(s.rotulo.sum()), 'n_spoof': int((s.rotulo == 0).sum())}
        if d['n_bonafide'] >= 10 and d['n_spoof'] >= 10:
            d['eer'] = D.com_ic(s, grupos='locutor', B=B)
            d.update({k: v for k, v in D.taxas_no_limiar(s.bona, s.spoof, t).items() if k != 'limiar'})
        out.append(d)
    return out


def atalho_duracao(c):
    db, ds = c.duracao[c.rotulo == 1], c.duracao[c.rotulo == 0]
    e1, e2 = D.eer(db, ds)[0], D.eer(-db, -ds)[0]
    return {'eer': min(e1, e2), 'sentido': 'genuínos mais longos' if e1 <= e2 else 'genuínos mais curtos',
            'auc_genuino_mais_longo': D.auc(db, ds)}


def eer_estratificada(c):
    eers, pesos = [], []
    for i in range(len(FAIXAS) - 1):
        m = (c.duracao >= FAIXAS[i]) & (c.duracao < FAIXAS[i + 1])
        s = c.subconjunto(m)
        if s.bona.size >= 10 and s.spoof.size >= 10:
            eers.append(D.eer(s.bona, s.spoof)[0])
            pesos.append(s.bona.size)
    return float(np.average(eers, weights=pesos)) if eers else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--B', type=int, default=500)
    args = ap.parse_args()
    cens = {'df21': D.carregar_df21(), 'itw': D.carregar_itw(), 'brs': D.carregar_brs()}
    res = {}
    for k, c in cens.items():
        B = 100 if k == 'df21' else args.B
        e, t = D.eer(c.bona, c.spoof)
        r = {'eer_global': e, 'limiar': t, 'faixas': por_faixa(c, t, B), 'atalho_duracao': atalho_duracao(c),
             'eer_estratificada_por_faixa': eer_estratificada(c)}
        for nome, rot in (('bonafide', 1), ('spoof', 0)):
            rho = spearmanr(c.pont[c.rotulo == rot], c.duracao[c.rotulo == rot])
            r[f'spearman_pontuacao_duracao_{nome}'] = float(rho.correlation)
        curto = c.duracao < D.CORTE_S
        if k in ('itw', 'brs'):
            for var in ('zero', 'repeat_semcorte'):
                cv = D.carregar_itw(var) if k == 'itw' else D.carregar_brs(var)
                if not np.array_equal(cv.ids, c.ids):
                    D.erro('ids fora de ordem entre variantes')
                alvo = curto if var == 'zero' else ~curto
                outro = ~alvo
                r[f'{var}'] = {
                    'subconjunto': 'curtos (< 4,04 s)' if var == 'zero' else 'longos (> 4,04 s)',
                    'eer_repeat': eer_de_mask(c, alvo), f'eer_{var}': eer_de_mask(cv, alvo),
                    'controle_dif_abs_max_no_resto': float(np.abs(cv.pont[outro] - c.pont[outro]).max()),
                }
        res[k] = r
        print(f"\n== {k}: EER {D.pct(e)} | só duração: EER {D.pct(r['atalho_duracao']['eer'])} ({r['atalho_duracao']['sentido']})"
              f" | estratificada {D.pct(r['eer_estratificada_por_faixa'])}"
              f" | rho gen {r['spearman_pontuacao_duracao_bonafide']:.3f} fal {r['spearman_pontuacao_duracao_spoof']:.3f}")
        for d in r['faixas']:
            if 'eer' in d:
                print(f"   {d['faixa']:9s} n={d['n_bonafide']:6d}/{d['n_spoof']:6d} EER {D.pct(d['eer']['valor'])}"
                      f" [{D.pct(d['eer']['ic95'][0])}, {D.pct(d['eer']['ic95'][1])}] FRR {D.pct(d['genuinos_rejeitados'])}"
                      f" FAR {D.pct(d['falsos_aceitos'])}")
            else:
                print(f"   {d['faixa']:9s} n={d['n_bonafide']}/{d['n_spoof']} (pequeno demais)")
        for var in ('zero', 'repeat_semcorte'):
            if var in r:
                x = r[var]
                print(f"   {var}: {x['subconjunto']} EER repeat {D.pct(x['eer_repeat'])} -> {D.pct(x['eer_' + var])}"
                      f" (controle: dif. máx. no resto {x['controle_dif_abs_max_no_resto']:.3g})")

    # Figura: EER por faixa, um painel por cenário, IC por locutores
    nomes = {'df21': 'ASVspoof 2021 DF', 'itw': 'In-the-Wild', 'brs': 'BRSpeech-DF'}
    fig, axs = F.plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (k, r) in zip(axs, res.items()):
        fx = [d for d in r['faixas'] if 'eer' in d]
        x = np.arange(len(fx))
        v = np.array([d['eer']['valor'] for d in fx]) * 100
        lo = np.array([d['eer']['ic95'][0] for d in fx]) * 100
        hi = np.array([d['eer']['ic95'][1] for d in fx]) * 100
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt='o-', color=F.AZUL, ecolor=F.EIXO, markersize=6, capsize=0)
        ax.axhline(100 * r['eer_global'], color=F.TINTA2, linewidth=1, linestyle='--')
        ax.axhline(50, color=F.MUDO, linewidth=1, linestyle=':')
        ax.set_xticks(x, [d['faixa'] for d in fx], rotation=35, ha='right')
        ax.set_title(f"{nomes[k]} (global {D.pct(r['eer_global'], 1)})", loc='left')
        ax.set_xlabel('duração do áudio')
    axs[0].set_ylabel('EER (%), IC de 95% por locutores')
    axs[-1].annotate('acaso', xy=(0, 50), xytext=(0, 3), textcoords='offset points', color=F.MUDO, fontsize=8)
    F.salvar(fig, f"{D.pasta_saida('fase1', 'figuras')}/duracao.png")
    D.salvar_json(f"{D.pasta_saida('fase1')}/duracao.json", res, __file__, {'B': args.B, 'B_df21': 100, 'faixas_s': FAIXAS})


def eer_de_mask(c, m):
    return D.eer(c.pont[m & (c.rotulo == 1)], c.pont[m & (c.rotulo == 0)])[0]


if __name__ == '__main__':
    main()
