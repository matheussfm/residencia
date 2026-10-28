"""Gera lista e rótulos do In-the-Wild (Müller et al., 2022) para inferir.py e metricas.py.

Espera a pasta release_in_the_wild/ com os .wav e o meta.csv (colunas file, speaker,
label; rótulos "bona-fide" e "spoof"). O id é o nome do arquivo e o grupo é o locutor.
O formato do meta.csv é conferido e o script para se for diferente do esperado.

Exemplo (dentro do container):
    python preparar_itw.py --raiz /dados/in_the_wild/release_in_the_wild --saida listas/in_the_wild
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

from _comum import (checar_taxas, conferir_contagem, erro, escrever_tsv,
                    garantir_fora_third_party, resumo, validar_id)

ROTULOS = {'bona-fide': 'bonafide', 'spoof': 'spoof'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz', required=True, help='pasta com os .wav e o meta.csv')
    ap.add_argument('--meta', default=None, help='caminho do meta.csv (padrão: <raiz>/meta.csv)')
    ap.add_argument('--saida', required=True, help='pasta onde gravar lista.tsv e rotulos.tsv')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)

    meta = args.meta or os.path.join(args.raiz, 'meta.csv')
    with open(meta, newline='', encoding='utf-8') as f:
        linhas = list(csv.DictReader(f))
    if not linhas or not {'file', 'speaker', 'label'} <= set(linhas[0]):
        erro(f'meta.csv com colunas inesperadas: {list(linhas[0]) if linhas else "vazio"}')
    desconhecidos = {r['label'] for r in linhas} - set(ROTULOS)
    if desconhecidos:
        erro(f'rótulos inesperados no meta.csv: {desconhecidos}')

    lista, rotulos, faltando = [], [], []
    for r in linhas:
        id_ = r['file']
        validar_id(id_)
        caminho = os.path.join(args.raiz, id_)
        if not os.path.exists(caminho):
            faltando.append(caminho)
        lista.append((id_, caminho))
        rotulos.append((id_, ROTULOS[r['label']], r['speaker']))
    if faltando:
        erro(f'{len(faltando)} áudios não encontrados (ex.: {faltando[:3]})')
    if len({i for i, _ in lista}) != len(lista):
        erro('ids repetidos no meta.csv')

    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'rotulos.tsv'), ['id', 'rotulo', 'grupo'], rotulos)

    n_bona = sum(r[1] == 'bonafide' for r in rotulos)
    conferir_contagem('arquivos', len(lista), 31779)
    resumo('preparar_itw.py', [
        ('arquivos', len(lista)),
        ('genuínos / falsos', f'{n_bona} / {len(lista) - n_bona}'),
        ('locutores', len({r[2] for r in rotulos})),
        ('taxas de amostragem (amostra de 200)', checar_taxas([c for _, c in lista])),
        ('saída', args.saida),
    ])


if __name__ == '__main__':
    main()
