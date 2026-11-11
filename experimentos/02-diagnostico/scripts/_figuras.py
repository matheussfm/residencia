"""Estilo comum das figuras (PNG estáticas, fundo claro).

Paleta categórica com separação verificada para daltonismo: azul, laranja, verde-água etc.
Convenção fixa em todas as figuras: genuíno = azul, falso = laranja.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

AZUL, LARANJA, AQUA, AMARELO, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
CORES = [AZUL, LARANJA, AQUA, AMARELO, MAGENTA, '#008300', '#6250d6', '#e34948']
COR_GEN, COR_FAL = AZUL, LARANJA
TINTA, TINTA2, MUDO, GRADE, EIXO, FUNDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': FUNDO, 'axes.facecolor': FUNDO, 'savefig.facecolor': FUNDO,
    'axes.edgecolor': EIXO, 'axes.labelcolor': TINTA2, 'axes.titlecolor': TINTA,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.6,
    'xtick.color': MUDO, 'ytick.color': MUDO, 'text.color': TINTA,
    'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False,
    'lines.linewidth': 2,
})


def salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
