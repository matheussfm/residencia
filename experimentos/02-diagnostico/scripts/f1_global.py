"""Fase 1, análise A: quem se desloca fora do domínio, o genuíno ou o falso?

Para cada cenário (2021 DF eval, In-the-Wild, BRSpeech-DF), com as pontuações do Experimento 01
(preenchimento por repetição, corte de 4 s):
  - EER com IC de 95% por bootstrap de locutores, e AUC;
  - quantis da pontuação de cada classe;
  - taxas no limiar da EER do 2021 DF (o "limiar do domínio de treino").

Matriz cruzada: EER e AUC dos genuínos do cenário X contra os falsos do cenário Y. Todas as
pontuações saem do mesmo modelo com o mesmo pré-processamento, então são comparáveis entre
cenários. Se trocar só os genuínos (linha) já leva a EER para perto do valor fora do domínio,
a perda vem dos genuínos; se trocar só os falsos (coluna) não muda nada, os falsos continuam
detectáveis.

Distância de Wasserstein entre as distribuições de cada classe em cada cenário e no 2021 DF,
para medir o tamanho do deslocamento, em unidades de pontuação.

Saídas: resultados/fase1/global.json e figuras/distribuicoes.png, figuras/matriz_cruzada.png.
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import wasserstein_distance

import _diag as D
import _figuras as F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--B', type=int, default=1000, help='réplicas de bootstrap (o 2021 DF usa B/5)')
    args = ap.parse_args()
    out = D.pasta_saida('fase1')
    fig_dir = D.pasta_saida('fase1', 'figuras')

    cens = {'df21': D.carregar_df21(), 'itw': D.carregar_itw(), 'brs': D.carregar_brs()}
    nomes = {'df21': 'ASVspoof 2021 DF', 'itw': 'In-the-Wild', 'brs': 'BRSpeech-DF'}
    _, t_df21 = D.eer(cens['df21'].bona, cens['df21'].spoof)

    res = {'limiar_eer_df21': t_df21, 'cenarios': {}, 'matriz_cruzada': {}, 'wasserstein_vs_df21': {}}
    for k, c in cens.items():
        B = max(args.B // 5, 100) if k == 'df21' else args.B
        e, t = D.eer(c.bona, c.spoof)
        q = [1, 5, 25, 50, 75, 95, 99]
        res['cenarios'][k] = {
            'descricao': c.descricao,
            'n_bonafide': int(c.rotulo.sum()), 'n_spoof': int((c.rotulo == 0).sum()),
            'n_locutores': int(np.unique(c.meta['locutor']).size),
            'eer': D.com_ic(c, grupos='locutor', B=B),
            'eer_ic_por_audio': D.com_ic(c, grupos=None, B=B)['ic95'],
            'limiar_eer': t,
            'auc': D.auc(c.bona, c.spoof),
            'quantis_bonafide': dict(zip(map(str, q), np.percentile(c.bona, q))),
            'quantis_spoof': dict(zip(map(str, q), np.percentile(c.spoof, q))),
            'media_bonafide': float(c.bona.mean()), 'media_spoof': float(c.spoof.mean()),
            'no_limiar_df21': D.taxas_no_limiar(c.bona, c.spoof, t_df21),
            'no_limiar_akcit': D.taxas_no_limiar(c.bona, c.spoof, D.LIMIAR_AKCIT),
        }
        print(f"{k:5s} EER {D.pct(e)} IC95 locutores {[D.pct(x) for x in res['cenarios'][k]['eer']['ic95']]}"
              f" AUC {res['cenarios'][k]['auc']:.4f}")

    for kg, cg in cens.items():
        res['matriz_cruzada'][kg] = {}
        for ks, cs in cens.items():
            e, _ = D.eer(cg.bona, cs.spoof)
            res['matriz_cruzada'][kg][ks] = {'eer': e, 'auc': D.auc(cg.bona, cs.spoof)}
    print('\nMatriz cruzada (linhas: genuínos de; colunas: falsos de) EER')
    print('       ' + ''.join(f'{k:>9s}' for k in cens))
    for kg in cens:
        print(f'{kg:6s} ' + ''.join(f"{D.pct(res['matriz_cruzada'][kg][ks]['eer']):>9s}" for ks in cens))

    ref = cens['df21']
    for k, c in cens.items():
        res['wasserstein_vs_df21'][k] = {'bonafide': float(wasserstein_distance(c.bona, ref.bona)),
                                         'spoof': float(wasserstein_distance(c.spoof, ref.spoof))}

    # Figura 1: distribuições (pequenos múltiplos, um painel por cenário, eixo x comum)
    fig, axs = F.plt.subplots(3, 1, figsize=(7, 6.4), sharex=True)
    bins = np.linspace(-6, 6, 121)
    for ax, (k, c) in zip(axs, cens.items()):
        for v, cor, rot in ((c.bona, F.COR_GEN, 'genuínos'), (c.spoof, F.COR_FAL, 'falsos')):
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, density=True, histtype='step', color=cor,
                    linewidth=2, label=f'{rot} (n={v.size:,})'.replace(',', '.'))
        ax.axvline(t_df21, color=F.TINTA2, linewidth=1, linestyle='--')
        e = res['cenarios'][k]['eer']['valor']
        ax.set_title(f"{nomes[k]}: EER {D.pct(e)}", loc='left')
        ax.set_ylabel('densidade')
        ax.legend(loc='upper right')
    axs[0].annotate('limiar da EER no 2021 DF', xy=(t_df21, axs[0].get_ylim()[1] * 0.8), xytext=(6, 0),
                    textcoords='offset points', color=F.TINTA2, fontsize=8)
    axs[-1].set_xlabel('pontuação (logit da classe genuíno; maior = genuíno; cortada em ±6)')
    F.salvar(fig, f'{fig_dir}/distribuicoes.png')

    # Figura 2: matriz cruzada (sequencial azul, um tom)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('azul', ['#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])
    ks = list(cens)
    M = np.array([[res['matriz_cruzada'][a][b]['eer'] for b in ks] for a in ks]) * 100
    fig, ax = F.plt.subplots(figsize=(5.2, 4.2))
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=50)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{M[i, j]:.1f}%', ha='center', va='center', color='#ffffff' if M[i, j] > 20 else F.TINTA)
    ax.set_xticks(range(3), [nomes[k] for k in ks])
    ax.set_yticks(range(3), [nomes[k] for k in ks])
    ax.set_xlabel('falsos de')
    ax.set_ylabel('genuínos de')
    ax.set_title('EER cruzada: genuínos de um cenário contra falsos de outro', loc='left')
    fig.colorbar(im, ax=ax, label='EER (%)', shrink=0.8)
    F.salvar(fig, f'{fig_dir}/matriz_cruzada.png')

    D.salvar_json(f'{out}/global.json', res, __file__, {'B': args.B, 'B_df21': max(args.B // 5, 100)})
    print(f'\nWasserstein vs 2021 DF: {res["wasserstein_vs_df21"]}')


if __name__ == '__main__':
    main()
