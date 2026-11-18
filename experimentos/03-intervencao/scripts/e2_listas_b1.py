"""Etapa 2: listas para extrair as representações dos áudios de teste normalizados (variante I1 + B1).

Os áudios já existem no servidor: são os da Fase 2c do Exp. 02 (f2_normalizar_real.py), em
~/matheus/dados/exp02_normalizar/<condicao>_<cenario>.lista.tsv. Este script filtra essas listas
para os 10.640 ids da amostra da Fase 3 e troca o id pelo da amostra (prefixo "itw::", "brs::",
"df21_eval::" ou "df21_hidden::"), para que o .npz do f3_extrair.py fique alinhado ao das
features originais.

Condições:
  volume_padronizar  a normalização B1 (corta o silêncio, RMS a −18 dBFS, 0,5 s de silêncio no início);
  controle           só reamostra e regrava; é a base pareada da normalização, porque passa pelo mesmo
                     caminho de leitura e gravação (Exp. 02, PROCESSO, problema 14).

Saídas: listas/etapa2/<condicao>/lista.tsv (fora do git, como toda lista.tsv com caminhos do servidor)
e listas/etapa2/listas_b1.json (contagens e SHA-256 das listas de origem).

Exemplo (no servidor, fora ou dentro do container; só usa a biblioteca padrão):
    python3 -B e2_listas_b1.py --pasta ~/matheus/dados/exp02_normalizar
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))
from _comum import erro, escrever_tsv, ler_tsv, sha256  # noqa: E402

AMOSTRA = os.path.join(AQUI, '..', '..', '02-diagnostico', 'listas', 'fase3', 'amostra.tsv')
SAIDA = os.path.join(AQUI, '..', 'listas', 'etapa2')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pasta', required=True, help='pasta de saída do f2_normalizar_real.py no servidor')
    ap.add_argument('--condicoes', nargs='+', default=['volume_padronizar', 'controle'])
    args = ap.parse_args()
    pasta = os.path.expanduser(args.pasta)
    am = ler_tsv(AMOSTRA)
    resumo = {'amostra_sha256': sha256(AMOSTRA), 'condicoes': {}}
    for cond in args.condicoes:
        origem = {}
        for arq in ('itw', 'brs', 'df21'):
            cam = os.path.join(pasta, f'{cond}_{arq}.lista.tsv')
            if not os.path.exists(cam):
                erro(f'lista não encontrada: {cam}')
            origem[arq] = ({r['id']: r['caminho'] for r in ler_tsv(cam)}, sha256(cam))
        linhas = []
        for r in am:
            if r['cenario'] in ('itw', 'brs'):
                tabela, chave = origem[r['cenario']][0], r['id_original']
            else:
                tabela, chave = origem['df21'][0], r['id']
            if chave not in tabela:
                erro(f"{cond}: {r['id']} não está na lista da Fase 2c")
            linhas.append((r['id'], tabela[chave]))
        destino = os.path.join(SAIDA, cond, 'lista.tsv')
        escrever_tsv(destino, ['id', 'caminho'], linhas)
        resumo['condicoes'][cond] = {'n': len(linhas), 'listas_origem_sha256': {k: v[1] for k, v in origem.items()},
                                     'lista_sha256': sha256(destino)}
        print(f'{cond}: {len(linhas)} áudios -> {destino}')
    with open(os.path.join(SAIDA, 'listas_b1.json'), 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
