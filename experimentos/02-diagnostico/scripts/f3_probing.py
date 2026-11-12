"""Fase 3: onde a informação se perde? Probing linear por camada.

Entradas: o .npz do f3_extrair.py e o amostra.tsv do f3_selecionar.py.

Para cada representação (24 camadas do XLS-R, a saída final do XLS-R e o embedding do AASIST),
um classificador linear (padronização + regressão logística) é avaliado de duas formas:

  1. TRANSFERÊNCIA: treinado no 2021 DF eval (metade dos locutores) e testado nos outros
     cenários e na outra metade dos locutores do 2021 DF. Imita o detector: aprende no domínio
     de treino e é usado fora dele.
  2. NO PRÓPRIO DOMÍNIO: validação cruzada em 5 partes por locutor, dentro de cada cenário. Mede
     se a informação para separar as classes EXISTE naquela representação, naquele domínio.

Leitura:
  - no próprio domínio baixo e transferência alta: a informação está lá, mas o que se aprende no
    2021 DF não transfere: o gargalo é o que o classificador aprende (back-end/treino);
  - no próprio domínio alto em todas as camadas: o front-end não separa esse domínio;
  - a EER do modelo completo, no mesmo conjunto, serve de referência.

Saídas: resultados/fase3/probing.json e figuras/probing.png (em resultados/fase3/).
"""
import argparse
import sys

sys.dont_write_bytecode = True

import numpy as np

import _diag as D
import _figuras as F

CENARIOS = ['df21_eval', 'df21_hidden', 'itw', 'brs']
NOMES = {'df21_eval': '2021 DF eval', 'df21_hidden': '2021 DF hidden (sem silêncio)',
         'itw': 'In-the-Wild', 'brs': 'BRSpeech-DF'}


def classificador(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight='balanced'))


def eer_xy(s, y):
    return D.eer(s[y == 1], s[y == 0])[0]


def no_dominio(X, y, g, C, folds=5):
    from sklearn.model_selection import GroupKFold
    s = np.zeros(y.size)
    for tr, te in GroupKFold(n_splits=min(folds, np.unique(g).size)).split(X, y, g):
        if np.unique(y[tr]).size < 2:
            return float('nan')
        s[te] = classificador(C).fit(X[tr], y[tr]).decision_function(X[te])
    return eer_xy(s, y)


def memmap_npz(caminho, nome):
    """Memory-map de um array guardado sem compressão dentro de um .npz (np.savez)."""
    import struct
    import zipfile
    with zipfile.ZipFile(caminho) as zf:
        info = zf.getinfo(nome + '.npy')
    if info.compress_type != zipfile.ZIP_STORED:
        D.erro(f'{nome}.npy está comprimido no .npz; use np.savez (sem compressão)')
    with open(caminho, 'rb') as f:
        f.seek(info.header_offset)
        cab = f.read(30)
        n_nome, n_extra = struct.unpack('<HH', cab[26:30])
        f.seek(info.header_offset + 30 + n_nome + n_extra)
        versao = np.lib.format.read_magic(f)
        ler = np.lib.format.read_array_header_1_0 if versao == (1, 0) else np.lib.format.read_array_header_2_0
        forma, fortran, dtype = ler(f)
        inicio = f.tell()
    return np.memmap(caminho, dtype=dtype, mode='r', offset=inicio, shape=forma, order='F' if fortran else 'C')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', required=True)
    ap.add_argument('--amostra', required=True)
    ap.add_argument('--saida', default=None, help='pasta de saída (padrão: resultados/fase3)')
    ap.add_argument('--C', type=float, default=0.1)
    ap.add_argument('--semente', type=int, default=0)
    args = ap.parse_args()
    import os
    out = args.saida or D.pasta_saida('fase3')
    D.garantir_fora_third_party(out)
    os.makedirs(os.path.join(out, 'figuras'), exist_ok=True)

    z = np.load(D._usar(args.features))
    am = {r['id']: r for r in D.ler_tsv(D._usar(args.amostra))}
    ids = [str(i) for i in z['ids']]
    if set(ids) != set(am):
        D.erro('ids do .npz e do amostra.tsv não coincidem')
    cen = np.array([am[i]['cenario'] for i in ids])
    y = np.array([am[i]['rotulo'] == 'bonafide' for i in ids], int)
    loc = np.array([am[i]['locutor'] for i in ids])
    p_npz = z['pontuacao'].astype(float)
    p_exp01 = np.array([float(am[i]['pontuacao_exp01']) for i in ids])
    conf = {'pearson_pontuacao_npz_vs_exp01': float(np.corrcoef(p_npz, p_exp01)[0, 1]),
            'dif_abs_max': float(np.abs(p_npz - p_exp01).max())}
    print('conferência das pontuações (extração × Exp. 01):', conf)

    # As camadas ocupam ~500 MB. Para não depender de memória nem de disco livres, o array é lido
    # por memory-map direto de dentro do .npz (um zip sem compressão), uma camada por vez.
    camadas = memmap_npz(args.features, 'camadas')
    reps = [(f'camada_{i + 1}', camadas[:, i, :]) for i in range(camadas.shape[1])]
    reps += [('final', z['final']), ('aasist', z['aasist'])]

    # Divisão por locutores do 2021 DF eval: metade treina a transferência, metade testa
    rng = np.random.default_rng(args.semente)
    m_df = cen == 'df21_eval'
    locs_df = np.unique(loc[m_df])
    treino_locs = set(rng.choice(locs_df, locs_df.size // 2, replace=False))
    m_tr = m_df & np.isin(loc, list(treino_locs))
    m_te_df = m_df & ~m_tr

    res = {'conferencia': conf, 'C': args.C, 'n': {c: [int(((cen == c) & (y == 1)).sum()), int(((cen == c) & (y == 0)).sum())]
                                                     for c in CENARIOS},
           'eer_modelo': {}, 'representacoes': {}}
    for c in CENARIOS:
        m = (cen == c) if c != 'df21_eval' else m_te_df
        res['eer_modelo'][c] = eer_xy(p_npz[m], y[m])
    print('EER do modelo completo na amostra:', {c: D.pct(v) for c, v in res['eer_modelo'].items()})

    for nome, X in reps:
        X = np.ascontiguousarray(X, dtype=np.float32)
        clf = classificador(args.C).fit(X[m_tr], y[m_tr])
        d = {'transferencia': {}, 'no_dominio': {}}
        for c in CENARIOS:
            m = (cen == c) if c != 'df21_eval' else m_te_df
            d['transferencia'][c] = eer_xy(clf.decision_function(X[m]), y[m])
            mc = cen == c
            d['no_dominio'][c] = no_dominio(X[mc], y[mc], loc[mc], args.C)
        res['representacoes'][nome] = d
        print(f"{nome:10s} transf.: " + ' '.join(f"{c}={D.pct(d['transferencia'][c], 1)}" for c in CENARIOS)
              + ' | no domínio: ' + ' '.join(f"{c}={D.pct(d['no_dominio'][c], 1)}" for c in CENARIOS), flush=True)

    D.salvar_json(os.path.join(out, 'probing.json'), res, __file__, {'C': args.C, 'semente': args.semente})
    del camadas, reps

    nomes = list(res['representacoes'])
    x = np.arange(len(nomes))
    fig, axs = F.plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, c in zip(axs.ravel(), CENARIOS):
        t = [100 * res['representacoes'][n]['transferencia'][c] for n in nomes]
        dm = [100 * res['representacoes'][n]['no_dominio'][c] for n in nomes]
        ax.plot(x, t, '-', color=F.AZUL, label='treinado no 2021 DF (transferência)')
        ax.plot(x, dm, '-', color=F.AQUA, label='treinado no próprio domínio (CV por locutor)')
        ax.axhline(100 * res['eer_modelo'][c], color=F.TINTA2, linestyle='--', linewidth=1)
        ax.annotate('modelo completo', xy=(0, 100 * res['eer_modelo'][c]), xytext=(0, 3), textcoords='offset points',
                    color=F.TINTA2, fontsize=8)
        ax.set_title(NOMES[c], loc='left')
        ax.set_ylim(0, 55)
    for ax in axs[-1]:
        nc = len(nomes) - 2
        ax.set_xticks(list(range(0, nc, 2)) + [nc, nc + 1],
                      [str(i + 1) for i in range(0, nc, 2)] + ['final', 'AASIST'], fontsize=7)
        ax.set_xlabel('camada do XLS-R (1–24), saída final e embedding do AASIST')
    for ax in axs[:, 0]:
        ax.set_ylabel('EER do probe linear (%)')
    axs[0, 0].legend(loc='upper right', fontsize=8)
    F.salvar(fig, os.path.join(out, 'figuras', 'probing.png'))


if __name__ == '__main__':
    main()
