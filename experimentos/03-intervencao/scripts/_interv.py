"""Funções compartilhadas pelo Experimento 03 (intervenção).

Reaproveita o Experimento 02 em vez de duplicar código:
  - _diag.py      (cenários, EER do Tak via metricas.py do Exp. 01, bootstrap por locutor, .meta.json);
  - f3_probing.py (o mesmo probe linear da Fase 3 e a leitura das camadas por memory-map).

Acrescenta:
  - minDCF com os parâmetros do ASVspoof 5 (Track 1);
  - EER e minDCF com IC por bootstrap de locutores e diferença pareada;
  - a divisão fixa da amostra df21_eval da Fase 3 em treino / validação / teste (e0_dividir.py);
  - as representações e a grade de C fixadas a priori (pré-registro, README seção 4).

Compatível com Python 3.8 e versões mais novas.
"""
import os
import sys

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
EXP02_SCRIPTS = os.path.abspath(os.path.join(AQUI, '..', '..', '02-diagnostico', 'scripts'))
sys.path.insert(0, EXP02_SCRIPTS)

import numpy as np  # noqa: E402

import _diag as D  # noqa: E402

EXP03 = os.path.abspath(os.path.join(AQUI, '..'))
AMOSTRA_F3 = os.path.join(D.EXP02, 'listas', 'fase3', 'amostra.tsv')
DIVISAO = os.path.join(EXP03, 'listas', 'divisao_df21_eval.tsv')

# ---------------------------------------------------------------------------------------------
# minDCF, ASVspoof 5, Track 1 (conferido em Wang et al., 2024, arXiv:2408.08739, seção 3.1):
#   DCF(t) = beta * P_miss(t) + P_fa(t)    (DCF normalizada)
#   beta   = (C_miss / C_fa) * (1 - pi_spf) / pi_spf
#   P_miss = taxa de genuínos rejeitados; P_fa = taxa de falsos aceitos.
#   minDCF = mínimo de DCF(t) sobre os limiares (limiar "oráculo").
# O artigo remete os detalhes ao plano de avaliação (fase 2), não consultado diretamente.
DCF_PARAMS = {'C_miss': 1.0, 'C_fa': 10.0, 'pi_spf': 0.05,
              'fonte': 'ASVspoof 5, Track 1: Wang et al. (2024), arXiv:2408.08739, seção 3.1, eqs. (1)-(2)'}
BETA = DCF_PARAMS['C_miss'] / DCF_PARAMS['C_fa'] * (1 - DCF_PARAMS['pi_spf']) / DCF_PARAMS['pi_spf']

# Pré-registro (README, seção 4): representações e grade de C fixadas antes de rodar.
REPRESENTACOES = ['camada_5', 'camada_12', 'camada_24', 'aasist']
GRADE_C = [0.001, 0.01, 0.1, 1.0]
PARTICOES = ('treino', 'validacao', 'teste')


def min_dcf(bona, spoof, beta=BETA):
    """minDCF normalizada e o limiar que a atinge. Genuíno se pontuação >= limiar.

    Varre todos os limiares relevantes (cada pontuação observada e +inf). Com limiar -inf, tudo é
    aceito: P_miss = 0, P_fa = 1, DCF = 1. Logo minDCF <= 1.
    """
    bona, spoof = np.sort(np.asarray(bona, float)), np.sort(np.asarray(spoof, float))
    if bona.size == 0 or spoof.size == 0:
        return float('nan'), float('nan')
    t = np.concatenate([np.unique(np.concatenate([bona, spoof])), [np.inf]])
    p_miss = np.searchsorted(bona, t, side='left') / bona.size        # bona < t
    p_fa = 1 - np.searchsorted(spoof, t, side='left') / spoof.size    # spoof >= t
    dcf = beta * p_miss + p_fa
    k = int(np.argmin(dcf))
    return float(dcf[k]), float(t[k])


def eer_de(c):
    return D.eer(c.bona, c.spoof)[0]


def mindcf_de(c):
    return min_dcf(c.bona, c.spoof)[0]


ESTATISTICAS = {'eer': eer_de, 'mindcf': mindcf_de}


def metricas_com_ic(cen, B=1000, semente=0):
    """EER e minDCF com IC de 95% por bootstrap de locutores (cen.meta['locutor'])."""
    return {nome: D.com_ic(cen, estatistica=f, grupos='locutor', B=B, semente=semente)
            for nome, f in ESTATISTICAS.items()}


def diferenca_pareada(cen, pont_nova, B=1000, semente=0):
    """Diferença (nova − base) de EER e de minDCF, pareada: os mesmos locutores sorteados nas duas.

    cen.pont é a base; pont_nova está alinhada a cen.ids.
    """
    par = D.Cenario(cen.nome, cen.ids, np.asarray(pont_nova, float), cen.rotulo,
                    dict(cen.meta, p0=cen.pont), cen.duracao)

    def dif(f):
        def g(k):
            p0 = k.meta['p0'].astype(float)
            return f(k.bona, k.spoof) - f(p0[k.rotulo == 1], p0[k.rotulo == 0])
        return g

    return {'eer': D.com_ic(par, estatistica=dif(lambda b, s: D.eer(b, s)[0]), grupos='locutor', B=B, semente=semente),
            'mindcf': D.com_ic(par, estatistica=dif(lambda b, s: min_dcf(b, s)[0]), grupos='locutor', B=B, semente=semente)}


# ---------------------------------------------------------------------------------------------
# Amostra da Fase 3 e divisão

def ler_amostra():
    """dict id -> linha do amostra.tsv da Fase 3 (id, cenario, id_original, rotulo, locutor, grupo, pontuacao_exp01)."""
    return {r['id']: r for r in D.ler_tsv(D._usar(AMOSTRA_F3))}


def ler_divisao():
    """dict id -> partição ('treino', 'validacao' ou 'teste') dos trials df21_eval da amostra."""
    if not os.path.exists(DIVISAO):
        D.erro(f'divisão não encontrada: {DIVISAO} (rode e0_dividir.py)')
    return {r['id']: r['particao'] for r in D.ler_tsv(D._usar(DIVISAO))}


def pasta_saida(*partes):
    p = os.path.join(EXP03, 'resultados', *partes)
    D.garantir_fora_third_party(p)
    os.makedirs(p, exist_ok=True)
    return p


def salvar_json(caminho, dados, script, parametros=None):
    """Como o _diag.salvar_json, com os parâmetros do minDCF sempre registrados."""
    p = dict(parametros or {})
    p.setdefault('min_dcf', dict(DCF_PARAMS, beta=BETA))
    D.salvar_json(caminho, dados, script, p)
