"""Etapa 0: testes das métricas novas antes do uso (sem dados reais).

  1. minDCF contra uma varredura por força bruta, em pontuações sintéticas (com empates);
  2. casos-limite: classes perfeitamente separadas (minDCF = 0), pontuações iguais (minDCF = 1),
     orientação invertida (minDCF = 1);
  3. diferença pareada: comparar uma pontuação com ela mesma dá diferença 0 e IC [0, 0].
"""
import sys

sys.dont_write_bytecode = True

import numpy as np

import _interv as I
from _interv import D


def forca_bruta(b, s, beta=I.BETA):
    melhor = 1.0  # aceitar tudo
    for t in np.concatenate([b, s]):
        melhor = min(melhor, beta * np.mean(b < t) + np.mean(s >= t))
    return min(melhor, beta * 1.0)  # rejeitar tudo


def main():
    rng = np.random.default_rng(0)
    for k in range(200):
        nb, ns = rng.integers(1, 60), rng.integers(1, 60)
        b = np.round(rng.normal(1, 1, nb), 1)  # arredondado: força empates
        s = np.round(rng.normal(0, 1, ns), 1)
        a, f = I.min_dcf(b, s)[0], forca_bruta(b, s)
        assert abs(a - f) < 1e-12, (k, a, f)
    print('1. minDCF = força bruta em 200 casos sintéticos: ok')

    assert I.min_dcf([2, 3], [0, 1])[0] == 0.0
    assert I.min_dcf([1, 1], [1, 1])[0] == 1.0
    assert I.min_dcf([0, 1], [2, 3])[0] == 1.0
    assert abs(I.BETA - 1.9) < 1e-12
    print(f'2. casos-limite: ok (beta = {I.BETA:.2f})')

    n = 400
    loc = np.repeat(np.arange(20).astype(str), n // 20)
    y = np.tile([1, 0], n // 2)
    p = rng.normal(y * 2.0, 1)
    c = D.Cenario('teste', np.arange(n).astype(str), p, y, {'locutor': loc})
    d = I.diferenca_pareada(c, p, B=50)
    for m in ('eer', 'mindcf'):
        assert d[m]['valor'] == 0 and d[m]['ic95'] == [0.0, 0.0], d[m]
    d2 = I.diferenca_pareada(c, p + rng.normal(0, 3, n), B=50)
    assert d2['eer']['valor'] > 0 and d2['mindcf']['valor'] > 0
    print('3. diferença pareada: 0 contra si mesma; positiva com ruído acrescentado: ok')
    m = I.metricas_com_ic(c, B=50)
    print(f"   exemplo: EER {D.pct(m['eer']['valor'])} {[D.pct(x) for x in m['eer']['ic95']]}, "
          f"minDCF {m['mindcf']['valor']:.3f} {[round(x, 3) for x in m['mindcf']['ic95']]}")


if __name__ == '__main__':
    main()
