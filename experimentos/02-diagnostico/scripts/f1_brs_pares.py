"""Fase 1, análise F: BRSpeech-DF pareado e variação entre locutores.

O BRSpeech-DF é paralelo: cada falso refaz a frase de um genuíno, do mesmo locutor. Isso
permite comparar cada falso com o seu genuíno, mantendo fixos o locutor, a frase e, em boa
parte, a duração. Só a síntese muda.

1. Taxa de acerto pareada: P(pontuação do genuíno > pontuação do seu falso), por TTS, contra a
   AUC não pareada (todos os genuínos contra todos os falsos do TTS). Se a pareada for bem
   maior, o detector ainda percebe a síntese, mas o nível da pontuação varia muito de um
   genuíno para outro (locutor, gravação, frase), e é essa variação que gera a EER alta.
2. Correlação entre a pontuação de cada genuíno e a do seu falso. Correlação alta indica que a
   pontuação é dominada pelo que o par compartilha (locutor, texto, condição do áudio de
   referência), e não pela síntese.
3. Por locutor: genuínos rejeitados no limiar global, EER do próprio locutor e média dos
   genuínos. EER agrupada contra a média das EER por locutor: se a agrupada é bem maior, uma
   parte grande do erro vem de deslocamentos entre locutores (calibração por locutor).
4. Representatividade da divisão de teste: com as pontuações do AKCIT para o SSL em todas as
   divisões (62 locutores; nenhum detector foi treinado no BRSpeech-DF), sorteia 10 locutores
   1000 vezes e compara a EER sorteada com a do teste oficial. Mesmo pré-processamento do AKCIT
   (zeros) nos dois lados, para a comparação ser justa.

Saídas: resultados/fase1/brs_pares.json e figuras/brs_pares.png, figuras/brs_locutores.png.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import pearsonr, spearmanr

import _diag as D
import _figuras as F


def pares(c):
    idx_gen = {i: k for k, i in enumerate(c.ids) if c.rotulo[k] == 1}
    out = {}
    for tts in sorted(set(c.meta['grupo']) - {'bonafide'}):
        m = np.flatnonzero(c.meta['grupo'] == tts)
        g = np.array([c.pont[idx_gen[p]] for p in c.meta['par'][m]])
        s = c.pont[m]
        out[tts] = {
            'n_pares': int(m.size),
            'acerto_pareado': float(np.mean(g > s) + 0.5 * np.mean(g == s)),
            'auc_nao_pareada': D.auc(c.bona, s),
            'eer_vs_todos_genuinos': D.eer(c.bona, s)[0],
            'pearson_gen_fal': float(pearsonr(g, s)[0]),
            'spearman_gen_fal': float(spearmanr(g, s).correlation),
            'mediana_diferenca_gen_menos_fal': float(np.median(g - s)),
        }
    m = c.rotulo == 0
    g = np.array([c.pont[idx_gen[p]] for p in c.meta['par'][m]])
    out['todos'] = {'n_pares': int(m.sum()), 'acerto_pareado': float(np.mean(g > c.pont[m])),
                    'auc_nao_pareada': D.auc(c.bona, c.spoof), 'eer_vs_todos_genuinos': D.eer(c.bona, c.spoof)[0],
                    'pearson_gen_fal': float(pearsonr(g, c.pont[m])[0]),
                    'spearman_gen_fal': float(spearmanr(g, c.pont[m]).correlation)}
    return out, g, c.pont[m]


def por_locutor(c, t):
    out = {}
    for loc in sorted(np.unique(c.meta['locutor'])):
        s = c.subconjunto(c.meta['locutor'] == loc)
        out[loc] = {'n_bonafide': int(s.bona.size), 'n_spoof': int(s.spoof.size),
                    'frr_no_limiar_global': float((s.bona < t).mean()),
                    'far_no_limiar_global': float((s.spoof >= t).mean()),
                    'eer_proprio': D.eer(s.bona, s.spoof)[0],
                    'media_bonafide': float(s.bona.mean()), 'media_spoof': float(s.spoof.mean())}
    eers = np.array([v['eer_proprio'] for v in out.values()])
    pesos = np.array([v['n_bonafide'] for v in out.values()])
    return out, {'eer_agrupada': D.eer(c.bona, c.spoof)[0], 'media_eer_por_locutor': float(eers.mean()),
                 'media_ponderada_eer_por_locutor': float(np.average(eers, weights=pesos)),
                 'desvio_padrao_media_genuinos_entre_locutores': float(np.std([v['media_bonafide'] for v in out.values()])),
                 'desvio_padrao_media_falsos_entre_locutores': float(np.std([v['media_spoof'] for v in out.values()]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sorteios', type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {}

    c = D.carregar_brs()
    e, t = D.eer(c.bona, c.spoof)
    res['teste_nossas_repeat'] = {'pares': None, 'locutores': None}
    p, g_all, s_all = pares(c)
    loc, resumo_loc = por_locutor(c, t)
    res['teste_nossas_repeat'] = {'eer': e, 'limiar': t, 'pares': p, 'locutores': loc, 'resumo_locutores': resumo_loc}
    print(f'BRSpeech-DF teste (nossas, repetição): EER {D.pct(e)}')
    for k, v in p.items():
        print(f"  {k:12s} acerto pareado {D.pct(v['acerto_pareado'], 1)} | AUC não pareada {v['auc_nao_pareada']:.3f}"
              f" | EER {D.pct(v['eer_vs_todos_genuinos'], 1)} | r(gen, fal) {v['pearson_gen_fal']:.2f}")
    print(f"  agrupada {D.pct(resumo_loc['eer_agrupada'])} vs média das EER por locutor {D.pct(resumo_loc['media_eer_por_locutor'])}")
    for k, v in loc.items():
        print(f"   loc {k:6s} n={v['n_bonafide']:4d}/{v['n_spoof']:4d} FRR {D.pct(v['frr_no_limiar_global'], 1)}"
              f" EER próprio {D.pct(v['eer_proprio'], 1)} média gen {v['media_bonafide']:.2f} fal {v['media_spoof']:.2f}")

    # Todas as divisões, pontuações do AKCIT (SSL, zeros)
    todos = D.carregar_brs_akcit('ssl', 'todos')
    teste = D.carregar_brs_akcit('ssl', 'test')
    e_t, t_t = D.eer(todos.bona, todos.spoof)
    pt, _, _ = pares(todos)
    loc_t, resumo_t = por_locutor(todos, t_t)
    locs = np.unique(todos.meta['locutor'])
    locs_teste = set(np.unique(teste.meta['locutor']))
    # índice por locutor para sortear rápido
    blocos = {l: np.flatnonzero(todos.meta['locutor'] == l) for l in locs}
    eers = []
    for _ in range(args.sorteios):
        esc = rng.choice(locs, len(locs_teste), replace=False)
        idx = np.concatenate([blocos[l] for l in esc])
        eers.append(D.eer(todos.pont[idx][todos.rotulo[idx] == 1], todos.pont[idx][todos.rotulo[idx] == 0])[0])
    eers = np.array(eers)
    e_teste = D.eer(teste.bona, teste.spoof)[0]
    res['todas_divisoes_akcit_ssl'] = {
        'n_bonafide': int(todos.rotulo.sum()), 'n_spoof': int((todos.rotulo == 0).sum()), 'n_locutores': int(locs.size),
        'eer': e_t, 'pares': pt, 'locutores': loc_t, 'resumo_locutores': resumo_t,
        'sorteio_10_locutores': {'sorteios': args.sorteios, 'eer_teste_oficial': e_teste,
                                 'percentil_do_teste': float((eers < e_teste).mean()),
                                 'quantis': dict(zip(['5', '25', '50', '75', '95'], np.percentile(eers, [5, 25, 50, 75, 95])))},
        'locutores_teste': sorted(locs_teste),
    }
    print(f'\nTodas as divisões (AKCIT, SSL, zeros): {locs.size} locutores, EER {D.pct(e_t)}; teste oficial {D.pct(e_teste)}')
    q = res['todas_divisoes_akcit_ssl']['sorteio_10_locutores']
    print(f"  10 locutores sorteados: mediana {D.pct(q['quantis']['50'])}, 5–95% [{D.pct(q['quantis']['5'])}, {D.pct(q['quantis']['95'])}];"
          f" o teste oficial fica no percentil {100 * q['percentil_do_teste']:.0f}")
    print(f"  agrupada {D.pct(resumo_t['eer_agrupada'])} vs média por locutor {D.pct(resumo_t['media_eer_por_locutor'])}")
    for k, v in pt.items():
        print(f"  {k:12s} acerto pareado {D.pct(v['acerto_pareado'], 1)} | AUC não pareada {v['auc_nao_pareada']:.3f} | r {v['pearson_gen_fal']:.2f}")

    # Figura 1: dispersão genuíno x seu falso (teste, nossas pontuações)
    fig, axs = F.plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axs[0]
    ax.scatter(g_all, s_all, s=6, color=F.AZUL, alpha=0.25, linewidths=0)
    lim = [min(g_all.min(), s_all.min()), max(g_all.max(), s_all.max())]
    ax.plot(lim, lim, color=F.TINTA2, linewidth=1, linestyle='--')
    ax.set_xlabel('pontuação do genuíno')
    ax.set_ylabel('pontuação do falso com a mesma frase')
    ax.set_title(f"Pares do teste (r = {p['todos']['pearson_gen_fal']:.2f}); acima da diagonal,\n"
                 'o falso parece mais genuíno que o próprio par', loc='left')
    ax = axs[1]
    tts = [k for k in p if k != 'todos']
    x = np.arange(len(tts))
    ax.plot(x, [100 * p[k]['acerto_pareado'] for k in tts], 'o-', color=F.AZUL, label='acerto pareado')
    ax.plot(x, [100 * p[k]['auc_nao_pareada'] for k in tts], 'o-', color=F.LARANJA, label='AUC não pareada')
    ax.axhline(50, color=F.MUDO, linewidth=1, linestyle=':')
    ax.set_xticks(x, tts)
    ax.set_ylim(40, 100)
    ax.set_ylabel('%')
    ax.set_title('Ordenação com e sem pareamento, por TTS', loc='left')
    ax.legend(loc='lower right')
    F.salvar(fig, f"{D.pasta_saida('fase1', 'figuras')}/brs_pares.png")

    # Figura 2: distribuição das EER com 10 locutores sorteados
    fig, ax = F.plt.subplots(figsize=(6, 3.2))
    ax.hist(100 * eers, bins=40, color=F.AZUL, edgecolor=F.FUNDO, linewidth=0.5)
    ax.axvline(100 * e_teste, color=F.LARANJA, linewidth=2)
    ax.annotate(f'teste oficial: {D.pct(e_teste, 1)}', xy=(100 * e_teste, ax.get_ylim()[1] * 0.9), xytext=(5, 0),
                textcoords='offset points', color=F.TINTA)
    ax.set_xlabel('EER (%) com 10 locutores sorteados entre os 62 (SSL, pontuações do AKCIT)')
    ax.set_ylabel('sorteios')
    ax.set_title('Quanto a EER do BRSpeech-DF depende de quais locutores entram', loc='left')
    F.salvar(fig, f"{D.pasta_saida('fase1', 'figuras')}/brs_locutores.png")

    D.salvar_json(f"{D.pasta_saida('fase1')}/brs_pares.json", res, __file__, {'sorteios': args.sorteios})


if __name__ == '__main__':
    main()
