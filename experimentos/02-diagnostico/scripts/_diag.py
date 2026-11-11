"""Funções compartilhadas pelas análises do Experimento 02 (diagnóstico da perda de generalização).

Reaproveita o Experimento 01 em vez de duplicar código:
  - _comum.py   (leitura de pontuações e TSV, trava contra escrita em third_party/, SHA-256);
  - metricas.py (compute_eer copiado do Tak, limiar −3,5327).

Também carrega os três cenários num formato único (`Cenario`) e implementa o bootstrap por
grupos (locutor) e as métricas de calibração (Cllr e minCllr).

Compatível com Python 3.8 (imagem Docker do Experimento 01) e com versões mais novas.
"""
import json
import os
import platform
import sys

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
EXP01 = os.path.abspath(os.path.join(AQUI, '..', '..', '01-linha-de-base'))
sys.path.insert(0, os.path.join(EXP01, 'scripts'))

import numpy as np

from _comum import (BRSPEECH_REPO, RAIZ_REPO, erro, garantir_fora_third_party,  # noqa: E402
                    ler_pontuacoes, ler_tsv, sha256)
from metricas import LIMIAR_AKCIT, compute_eer  # noqa: E402

EXP02 = os.path.abspath(os.path.join(AQUI, '..'))
RES01 = os.path.join(EXP01, 'resultados')
LIS01 = os.path.join(EXP01, 'listas')
CORTE_S = 64600 / 16000

# Chave completa do ASVspoof 2021 DF (codec, origem, ataque, vocoder...), distribuída pelo AKCIT.
# O conjunto eval e os rótulos foram conferidos contra DF-keys-stage-1 do Tak (idênticos);
# só muda o nome da fase oculta (hidden_track no Tak, hidden aqui). Ver PROCESSO.md.
CHAVE_DF21 = os.path.join(BRSPEECH_REPO, 'data', 'metadata', 'trial_metadata.txt')
COLUNAS_DF21 = ['locutor', 'id', 'codec', 'origem', 'ataque', 'rotulo', 'trim', 'fase',
                'vocoder', 'tarefa', 'equipe', 'generos', 'col13']

ENTRADAS_USADAS = set()  # caminhos lidos, para o .meta.json


def _usar(caminho):
    ENTRADAS_USADAS.add(os.path.abspath(caminho))
    return caminho


class Cenario:
    """Pontuações e metadados de um cenário, alinhados por posição.

    rotulo: 1 = genuíno (bonafide), 0 = falso (spoof). Pontuação maior => genuíno.
    meta: dict nome -> np.array de str (ex.: 'locutor', 'grupo', 'codec').
    duracao: np.array de float (s) ou None.
    """

    def __init__(self, nome, ids, pont, rotulo, meta, duracao=None, descricao=''):
        self.nome = nome
        self.ids = np.asarray(ids)
        self.pont = np.asarray(pont, dtype=float)
        self.rotulo = np.asarray(rotulo, dtype=int)
        self.meta = {k: np.asarray(v) for k, v in meta.items()}
        self.duracao = None if duracao is None else np.asarray(duracao, dtype=float)
        self.descricao = descricao

    def __len__(self):
        return self.ids.size

    def subconjunto(self, mascara, nome=None):
        return Cenario(nome or self.nome, self.ids[mascara], self.pont[mascara], self.rotulo[mascara],
                       {k: v[mascara] for k, v in self.meta.items()},
                       None if self.duracao is None else self.duracao[mascara], self.descricao)

    @property
    def bona(self):
        return self.pont[self.rotulo == 1]

    @property
    def spoof(self):
        return self.pont[self.rotulo == 0]


def _duracoes(caminho):
    return {r['id']: float(r['segundos']) for r in ler_tsv(_usar(caminho))}


def _alinhar(ids, pont_dict, origem):
    faltam = [i for i in ids if i not in pont_dict]
    if faltam:
        erro(f'{len(faltam)} ids sem pontuação em {origem} (ex.: {faltam[:3]})')
    return [pont_dict[i] for i in ids]


def carregar_df21(variante='repeat', fase='eval', pontuacoes=None):
    """ASVspoof 2021 DF. fase: 'eval' (EER oficial), 'hidden' (trim, sem silêncio) ou 'progress'."""
    caminho = pontuacoes or os.path.join(RES01, 'asvspoof21df', f'pontuacoes_{variante}.txt')
    pont = ler_pontuacoes(_usar(caminho))
    linhas = []
    with open(_usar(CHAVE_DF21), encoding='utf-8') as f:
        for linha in f:
            col = linha.split()
            if len(col) != len(COLUNAS_DF21):
                erro(f'linha da chave com {len(col)} colunas: {linha.strip()!r}')
            if col[7] == fase:
                linhas.append(col)
    if not linhas:
        erro(f'nenhum trial na fase {fase!r}')
    ids = [c[1] for c in linhas]
    meta = {nome: [c[i] for c in linhas] for i, nome in enumerate(COLUNAS_DF21) if nome not in ('id', 'rotulo', 'fase')}
    dur = None
    if fase == 'eval':
        d = _duracoes(os.path.join(RES01, 'asvspoof21df', 'duracoes_por_arquivo.tsv'))
        dur = [d.get(i, np.nan) for i in ids]
    return Cenario(f'df21_{fase}', ids, _alinhar(ids, pont, caminho), [c[5] == 'bonafide' for c in linhas], meta, dur,
                   f'ASVspoof 2021 DF, fase {fase}, pontuações {os.path.basename(caminho)}')


def carregar_itw(variante='repeat'):
    """In-the-Wild. meta['locutor'] vem da coluna speaker do meta.csv."""
    rot = ler_tsv(_usar(os.path.join(LIS01, 'in_the_wild', 'rotulos.tsv')))
    caminho = os.path.join(RES01, 'in_the_wild', f'pontuacoes_{variante}.txt')
    pont = ler_pontuacoes(_usar(caminho))
    d = _duracoes(os.path.join(RES01, 'in_the_wild', 'duracoes_por_arquivo.tsv'))
    ids = [r['id'] for r in rot]
    return Cenario('itw', ids, _alinhar(ids, pont, caminho), [r['rotulo'] == 'bonafide' for r in rot],
                   {'locutor': [r['grupo'] for r in rot]}, [d[i] for i in ids],
                   f'In-the-Wild, pontuações {os.path.basename(caminho)}')


def carregar_brs(variante='repeat'):
    """BRSpeech-DF (teste, versão Hugging Face).

    meta['grupo']: TTS ou 'bonafide'. meta['locutor']: 3º nível do caminho original do AKCIT.
    meta['par']: id do genuíno com a mesma frase (para os genuínos, o próprio id). O par dos
    falsos vem do mapa 'ordenado', validado no Experimento 01 (correlação de 1,0000 com as
    pontuações do AKCIT).
    """
    L = os.path.join(LIS01, 'brspeech_test_hf')
    rot = ler_tsv(_usar(os.path.join(L, 'rotulos.tsv')))
    mapa = {r['id']: r['id_referencia'] for r in ler_tsv(_usar(os.path.join(L, 'mapa_falsos_ordenado.tsv')))}
    caminho = os.path.join(RES01, 'brspeech_test_hf', f'pontuacoes_{variante}.txt')
    pont = ler_pontuacoes(_usar(caminho))
    d = _duracoes(os.path.join(RES01, 'brspeech_test_hf', 'duracoes_por_arquivo.tsv'))
    ids, par = [r['id'] for r in rot], []
    for r in rot:
        par.append(r['id'] if r['rotulo'] == 'bonafide' else mapa[r['id']].split('/', 1)[1])
    gen = {r['id'] for r in rot if r['rotulo'] == 'bonafide'}
    if any(p not in gen for p in par):
        erro('há falsos sem genuíno correspondente')
    return Cenario('brs', ids, _alinhar(ids, pont, caminho), [r['rotulo'] == 'bonafide' for r in rot],
                   {'grupo': [r['grupo'] for r in rot], 'locutor': [p.split('/')[2] for p in par], 'par': par},
                   [d[i] for i in ids], f'BRSpeech-DF teste (HF), pontuações {os.path.basename(caminho)}')


def carregar_brs_akcit(detector, subset='test'):
    """BRSpeech-DF com as pontuações publicadas pelo AKCIT (detector: ssl, aasist ou sls).

    Usa os ids originais do AKCIT (relative_path) e os metadados do submódulo. subset: 'test',
    'train', 'dev' ou 'todos' (nenhum dos detectores foi treinado no BRSpeech-DF).
    O pré-processamento é o do AKCIT (no SSL, preenchimento por zeros), não o do Tak.
    """
    S = os.path.join(BRSPEECH_REPO, 'data', 'scores', detector)
    pg = ler_pontuacoes(_usar(os.path.join(S, f'{detector}_brspeech_full_scores.pkl')))
    ps = ler_pontuacoes(_usar(os.path.join(S, f'{detector}_sintetico_full_scores.pkl')))
    M = os.path.join(BRSPEECH_REPO, 'data', 'metadata', 'brspeech_df')
    import csv
    ids, pont, rot, grupo, loc, par = [], [], [], [], [], []
    for nome, fonte, eh_gen in (('brspeech_meta.csv', pg, True), ('synthetic_meta.csv', ps, False)):
        with open(_usar(os.path.join(M, nome)), newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='|'):
                if subset != 'todos' and r['subset'] != subset:
                    continue
                rp = r['relative_path']
                if rp not in fonte:
                    continue  # contado abaixo
                ids.append(rp)
                pont.append(fonte[rp])
                rot.append(eh_gen)
                grupo.append('bonafide' if eh_gen else r['tts'])
                p = rp if eh_gen else rp.split('/', 1)[1]
                par.append(p)
                loc.append(p.split('/')[2])
    return Cenario(f'brs_akcit_{detector}_{subset}', ids, pont, rot,
                   {'grupo': grupo, 'locutor': loc, 'par': par}, None,
                   f'BRSpeech-DF ({subset}), pontuações publicadas pelo AKCIT para {detector}')


def carregar_df21_akcit(detector):
    """ASVspoof 2021 DF eval com as pontuações publicadas pelo AKCIT (ssl, sls ou aasist)."""
    S = os.path.join(BRSPEECH_REPO, 'data', 'scores', detector)
    if detector == 'aasist':
        bruto = ler_pontuacoes(_usar(os.path.join(S, 'aasist_asvspoof2021DFeval_scores.pkl')))
        pont = {os.path.splitext(os.path.basename(k))[0]: v for k, v in bruto.items()}
        tmp = os.path.join(EXP02, 'resultados', '.aasist_df21_tmp.txt')
        garantir_fora_third_party(tmp)
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            f.writelines(f'{k} {v}\n' for k, v in pont.items())
        try:
            c = carregar_df21(pontuacoes=tmp)
        finally:
            os.remove(tmp)
            ENTRADAS_USADAS.discard(os.path.abspath(tmp))
        c.descricao = 'ASVspoof 2021 DF eval, pontuações publicadas pelo AKCIT para aasist'
        return c
    c = carregar_df21(pontuacoes=os.path.join(S, f'{detector}_asvspoof2021DF_scores.txt'))
    c.descricao = f'ASVspoof 2021 DF eval, pontuações publicadas pelo AKCIT para {detector}'
    return c


# ---------------------------------------------------------------------------------------------
# Métricas

def eer(bona, spoof):
    """EER (fração) e limiar, pela função do Tak."""
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    if bona.size == 0 or spoof.size == 0:
        return float('nan'), float('nan')
    e, t = compute_eer(bona, spoof)
    return float(e), float(t)


def auc(bona, spoof):
    """P(pontuação de um genuíno > pontuação de um falso), empates contam 1/2 (Mann-Whitney)."""
    from scipy.stats import rankdata
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    if bona.size == 0 or spoof.size == 0:
        return float('nan')
    r = rankdata(np.concatenate([bona, spoof]))
    return float((r[:bona.size].sum() - bona.size * (bona.size + 1) / 2) / (bona.size * spoof.size))


def taxas_no_limiar(bona, spoof, t):
    """Genuíno se pontuação >= t (convenção do metricas.py)."""
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    frr = float((bona < t).mean()) if bona.size else float('nan')
    far = float((spoof >= t).mean()) if spoof.size else float('nan')
    return {'limiar': float(t), 'genuinos_rejeitados': frr, 'falsos_aceitos': far,
            'erro_balanceado': float(np.nanmean([frr, far]))}


def cllr(llr_bona, llr_spoof):
    """Cllr (bits). Supõe que as entradas são log-razões de verossimilhança naturais."""
    lb, ls = np.asarray(llr_bona, float), np.asarray(llr_spoof, float)
    return float((np.mean(np.logaddexp(0, -lb)) + np.mean(np.logaddexp(0, ls))) / (2 * np.log(2)))


def min_cllr(bona, spoof):
    """minCllr: Cllr após a melhor transformação monotônica (PAV), isto é, só a perda de ordenação."""
    from sklearn.isotonic import IsotonicRegression
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    x = np.concatenate([bona, spoof])
    y = np.concatenate([np.ones(bona.size), np.zeros(spoof.size)])
    # Pesos que equalizam as classes: a posterior do PAV vira uma LLR ao tirar log-odds 0.
    w = np.concatenate([np.full(bona.size, 0.5 / bona.size), np.full(spoof.size, 0.5 / spoof.size)])
    p = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit(x, y, sample_weight=w).predict(x)
    eps = 1e-12
    llr = np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps))
    return cllr(llr[:bona.size], llr[bona.size:])


def ajustar_calibracao(bona, spoof):
    """Calibração afim s -> a*s + b por regressão logística com classes de peso igual.

    Devolve (a, b). Com classes equilibradas, a*s + b é uma estimativa de LLR.
    """
    from sklearn.linear_model import LogisticRegression
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    x = np.concatenate([bona, spoof])[:, None]
    y = np.concatenate([np.ones(bona.size), np.zeros(spoof.size)])
    m = LogisticRegression(C=1e6, class_weight='balanced', max_iter=1000).fit(x, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


# ---------------------------------------------------------------------------------------------
# Bootstrap

def bootstrap(estatistica, cen, grupos=None, B=1000, semente=0, alfa=0.05):
    """Intervalo percentil de `estatistica(cenario)` por reamostragem.

    grupos: nome de uma coluna de cen.meta (ex.: 'locutor'). Se dado, reamostra os grupos com
    reposição e leva todos os áudios de cada um (bootstrap por grupos), o que respeita a
    dependência entre áudios do mesmo locutor. Senão, reamostra áudios dentro de cada classe.
    Devolve (ic_inf, ic_sup, n_validas).
    """
    rng = np.random.default_rng(semente)
    if grupos is not None:
        g = cen.meta[grupos]
        unicos, inv = np.unique(g, return_inverse=True)
        ordem = np.argsort(inv, kind='stable')
        limites = np.searchsorted(inv[ordem], np.arange(unicos.size + 1))
        blocos = [ordem[limites[k]:limites[k + 1]] for k in range(unicos.size)]
    else:
        ib, is_ = np.flatnonzero(cen.rotulo == 1), np.flatnonzero(cen.rotulo == 0)
    vals = []
    for _ in range(B):
        if grupos is not None:
            esc = rng.integers(0, len(blocos), len(blocos))
            idx = np.concatenate([blocos[k] for k in esc])
        else:
            idx = np.concatenate([rng.choice(ib, ib.size), rng.choice(is_, is_.size)])
        v = estatistica(cen.subconjunto(idx))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float('nan'), float('nan'), 0
    lo, hi = np.percentile(vals, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(lo), float(hi), len(vals)


def eer_de(c):
    return eer(c.bona, c.spoof)[0]


def com_ic(cen, estatistica=eer_de, grupos=None, B=1000, semente=0):
    """Valor pontual e IC de 95% como dict."""
    v = estatistica(cen)
    lo, hi, n = bootstrap(estatistica, cen, grupos, B, semente)
    return {'valor': v, 'ic95': [lo, hi], 'bootstrap': {'B': B, 'validas': n, 'grupos': grupos, 'semente': semente}}


# ---------------------------------------------------------------------------------------------
# Saída

def pasta_saida(*partes):
    p = os.path.join(EXP02, 'resultados', *partes)
    garantir_fora_third_party(p)
    os.makedirs(p, exist_ok=True)
    return p


def _jsonavel(o):
    if isinstance(o, dict):
        return {str(k): _jsonavel(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonavel(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def salvar_json(caminho, dados, script, parametros=None):
    """Grava o JSON e um .meta.json com os SHA-256 das entradas, as versões e os parâmetros."""
    garantir_fora_third_party(caminho)
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(_jsonavel(dados), f, indent=2, ensure_ascii=False)
    import scipy
    try:
        import sklearn  # não está na imagem do Experimento 01; só é usado pela calibração
        versao_sklearn = sklearn.__version__
    except ImportError:
        versao_sklearn = None
    meta = {
        'script': os.path.relpath(script, RAIZ_REPO).replace(os.sep, '/'),
        'parametros': parametros or {},
        'entradas': {os.path.relpath(p, RAIZ_REPO).replace(os.sep, '/'): sha256(p) for p in sorted(ENTRADAS_USADAS)},
        'saida_sha256': sha256(caminho),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'sklearn': versao_sklearn,
        'plataforma': platform.platform(),
    }
    with open(caminho + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def pct(x, casas=2):
    return '-' if x is None or not np.isfinite(x) else f'{100 * x:.{casas}f}%'
