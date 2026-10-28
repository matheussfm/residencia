"""Gera lista e rótulos da divisão de teste do BRSpeech-DF para inferir.py e metricas.py.

Usa os metadados publicados pelo AKCIT no submódulo (somente leitura):
  data/metadata/brspeech_df/brspeech_meta.csv   (genuínos)
  data/metadata/brspeech_df/synthetic_meta.csv  (falsos, com a coluna tts)

Os ids são exatamente o relative_path dos metadados, que também é a chave dos .pkl de
pontuação do AKCIT; assim comparar_pontuacoes.py compara arquivo a arquivo.
O grupo é o TTS (falsos) ou "bonafide" (genuínos).

Genuínos e falsos ficam em pastas-raiz diferentes no dataset (o relative_path dos
falsos começa pelo nome do TTS). Se um arquivo não existir com a extensão dos metadados,
tenta a outra (.wav/.flac), como faz o loader do AKCIT.

Exemplo (dentro do container):
    python preparar_brspeech.py --raiz-genuinos /dados/brspeech_df/brspeech \
        --raiz-falsos /dados/brspeech_df/sintetico --saida listas/brspeech_test
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

from _comum import (BRSPEECH_REPO, checar_taxas, conferir_contagem, erro, escrever_tsv,
                    garantir_fora_third_party, resumo, validar_id)

META = os.path.join(BRSPEECH_REPO, 'data', 'metadata', 'brspeech_df')


def ler_meta(caminho, subset):
    with open(caminho, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f, delimiter='|') if r['subset'] == subset]


def localizar(raiz, rel):
    c = os.path.join(raiz, rel)
    if os.path.exists(c):
        return c
    base, ext = os.path.splitext(c)
    alt = base + ('.wav' if ext.lower() == '.flac' else '.flac')
    return alt if os.path.exists(alt) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz-genuinos', required=True, help='pasta a que o relative_path dos genuínos é relativo')
    ap.add_argument('--raiz-falsos', required=True, help='pasta a que o relative_path dos falsos é relativo')
    ap.add_argument('--saida', required=True, help='pasta onde gravar lista.tsv e rotulos.tsv')
    ap.add_argument('--subset', default='test')
    ap.add_argument('--sem-checar-audio', action='store_true', help='não confere se os arquivos existem')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)

    gen = ler_meta(os.path.join(META, 'brspeech_meta.csv'), args.subset)
    fal = ler_meta(os.path.join(META, 'synthetic_meta.csv'), args.subset)

    lista, rotulos, faltando = [], [], []
    for linhas, raiz in ((gen, args.raiz_genuinos), (fal, args.raiz_falsos)):
        for r in linhas:
            id_ = r['relative_path']
            validar_id(id_)
            caminho = os.path.join(raiz, id_) if args.sem_checar_audio else localizar(raiz, id_)
            if caminho is None:
                faltando.append(os.path.join(raiz, id_))
                continue
            lista.append((id_, caminho))
            rotulos.append((id_, r['df_class'], r.get('tts') or 'bonafide'))
    if faltando:
        erro(f'{len(faltando)} áudios não encontrados (ex.: {faltando[:3]}). Confira --raiz-genuinos/--raiz-falsos.')
    if len({i for i, _ in lista}) != len(lista):
        erro('ids repetidos entre genuínos e falsos')

    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'rotulos.tsv'), ['id', 'rotulo', 'grupo'], rotulos)

    por_grupo = {}
    for _, _, g in rotulos:
        por_grupo[g] = por_grupo.get(g, 0) + 1
    if args.subset == 'test':
        conferir_contagem('genuínos', len(gen), 1320)
        conferir_contagem('falsos', len(fal), 6587)
    itens = [
        ('subset', args.subset),
        ('genuínos / falsos', f'{len(gen)} / {len(fal)}'),
        ('por grupo', por_grupo),
        ('locutores (genuínos)', len({r['relative_path'].split('/')[2] for r in gen})),
    ]
    if not args.sem_checar_audio:
        itens += [
            ('taxas, genuínos (amostra de 200)', checar_taxas([c for _, c in lista[:len(gen)]])),
            ('taxas, falsos (amostra de 200)', checar_taxas([c for _, c in lista[len(gen):]])),
        ]
    itens.append(('saída', args.saida))
    resumo('preparar_brspeech.py', itens)


if __name__ == '__main__':
    main()
