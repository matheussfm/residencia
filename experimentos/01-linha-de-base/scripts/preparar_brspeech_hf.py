"""Gera lista, rótulos e mapas da divisão de teste do BRSpeech-DF (versão Hugging Face).

Lê o indice.tsv gravado por extrair_brspeech_hf.py e os metadados do AKCIT no submódulo
(somente leitura). Saídas em --saida:
- lista.tsv, rotulos.tsv: como nos outros preparadores; grupo = modelo TTS ou "bonafide".
  Os genuínos recebem como id o relative_path do AKCIT (casado pelo nome do arquivo), que é
  a chave dos .pkl de pontuação publicados. Os falsos recebem o caminho extraído.
- mapa_falsos_ordenado.tsv / mapa_falsos_csv.tsv: hipóteses de correspondência entre cada
  falso do parquet (que não traz o nome original) e o relative_path do AKCIT, supondo que as
  linhas de cada modelo estejam em ordem alfabética do nome ou na ordem do CSV do AKCIT.
  A comparação com as pontuações publicadas (comparar_pontuacoes.py --mapa) mostra qual
  hipótese é a correta: a certa dá correlação alta; a errada, perto de zero.

Exemplo (dentro do container):
    python preparar_brspeech_hf.py --raiz /dados/brspeech_df/hf_test --saida ../listas/brspeech_test_hf
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True

from _comum import (BRSPEECH_REPO, checar_taxas, conferir_contagem, erro, escrever_tsv,
                    garantir_fora_third_party, ler_tsv, resumo, validar_id)

META = os.path.join(BRSPEECH_REPO, 'data', 'metadata', 'brspeech_df')


def ler_meta(nome):
    with open(os.path.join(META, nome), newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f, delimiter='|') if r['subset'] == 'test']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raiz', required=True, help='pasta gravada por extrair_brspeech_hf.py')
    ap.add_argument('--saida', required=True)
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)

    indice = ler_tsv(os.path.join(args.raiz, 'indice.tsv'))
    gen_akcit = ler_meta('brspeech_meta.csv')
    fal_akcit = ler_meta('synthetic_meta.csv')
    por_nome = {os.path.splitext(os.path.basename(r['relative_path']))[0]: r['relative_path'] for r in gen_akcit}
    if len(por_nome) != len(gen_akcit):
        erro('nomes de arquivo repetidos entre os genuínos do AKCIT')

    lista, rotulos, faltando = [], [], []
    falsos_hf = {}
    for r in indice:
        caminho = os.path.join(args.raiz, r['caminho'])
        if not os.path.exists(caminho):
            faltando.append(caminho)
        if r['rotulo'] == 'bonafide':
            nome = os.path.splitext(os.path.basename(r['caminho']))[0]
            if nome not in por_nome:
                erro(f'genuíno do parquet sem correspondente nos metadados do AKCIT: {nome}')
            id_ = por_nome[nome]
        else:
            id_ = r['caminho']
            falsos_hf.setdefault(r['modelo'], []).append(id_)
        validar_id(id_)
        lista.append((id_, caminho))
        rotulos.append((id_, r['rotulo'], r['modelo']))
    if faltando:
        erro(f'{len(faltando)} áudios não encontrados (ex.: {faltando[:3]})')
    if len({i for i, _ in lista}) != len(lista):
        erro('ids repetidos')

    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], lista)
    escrever_tsv(os.path.join(args.saida, 'rotulos.tsv'), ['id', 'rotulo', 'grupo'], rotulos)

    for hipotese in ('ordenado', 'csv'):
        mapa = []
        for modelo, ids in falsos_hf.items():
            akcit = [r['relative_path'] for r in fal_akcit if r['tts'] == modelo]
            if len(akcit) != len(ids):
                erro(f'{modelo}: {len(ids)} falsos no parquet e {len(akcit)} nos metadados do AKCIT')
            if hipotese == 'ordenado':
                akcit = sorted(akcit, key=os.path.basename)
            mapa += list(zip(ids, akcit))
        escrever_tsv(os.path.join(args.saida, f'mapa_falsos_{hipotese}.tsv'), ['id', 'id_referencia'], mapa)

    n_gen = sum(r[1] == 'bonafide' for r in rotulos)
    conferir_contagem('genuínos', n_gen, 1320)
    conferir_contagem('falsos', len(rotulos) - n_gen, 6587)
    itens = [
        ('genuínos / falsos', f'{n_gen} / {len(rotulos) - n_gen}'),
        ('falsos por modelo', {m: len(v) for m, v in sorted(falsos_hf.items())}),
    ]
    for g in ['bonafide'] + sorted(falsos_hf):
        caminhos = [c for (_, c), r in zip(lista, rotulos) if r[2] == g]
        itens.append((f'taxas, {g} (amostra de 40)', checar_taxas(caminhos[::max(1, len(caminhos) // 40)], 40)))
    itens.append(('saída', args.saida))
    resumo('preparar_brspeech_hf.py', itens)


if __name__ == '__main__':
    main()
