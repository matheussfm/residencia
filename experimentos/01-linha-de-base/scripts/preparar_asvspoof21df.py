"""Gera lista e rótulos do ASVspoof 2021 DF (avaliação) para inferir.py e metricas.py.

- lista.tsv: todos os 611.829 trials do protocolo (o evaluate_2021_DF.py exige o total).
- rotulos.tsv: só a fase "eval" das chaves DF-keys-stage-1 (533.928 trials), que é
  a usada no EER oficial; grupo = fase.

O protocolo e as chaves são lidos do submódulo third_party/SSL_Anti-spoofing.

Exemplo (dentro do container):
    python preparar_asvspoof21df.py --audio /dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac \
        --saida listas/asvspoof21df
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True

from _comum import (SSL_REPO, checar_taxas, conferir_contagem, erro, escrever_tsv,
                    garantir_fora_third_party, resumo, validar_id)

PROTOCOLO = os.path.join(SSL_REPO, 'database', 'ASVspoof_DF_cm_protocols', 'ASVspoof2021.DF.cm.eval.trl.txt')
CHAVES = os.path.join(SSL_REPO, 'DF-keys-stage-1', 'keys', 'CM', 'trial_metadata.txt')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio', required=True, help='pasta com os .flac (ASVspoof2021_DF_eval/flac)')
    ap.add_argument('--saida', required=True, help='pasta onde gravar lista.tsv e rotulos.tsv')
    ap.add_argument('--protocolo', default=PROTOCOLO)
    ap.add_argument('--chaves', default=CHAVES)
    ap.add_argument('--sem-checar-audio', action='store_true', help='não confere se os arquivos existem')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)

    with open(args.protocolo, encoding='utf-8') as f:
        ids = [linha.strip() for linha in f if linha.strip()]
    for id_ in ids:
        validar_id(id_)
    caminhos = [os.path.join(args.audio, id_ + '.flac') for id_ in ids]

    if not args.sem_checar_audio:
        faltando = [c for c in caminhos if not os.path.exists(c)]
        if faltando:
            erro(f'{len(faltando)} de {len(caminhos)} áudios não encontrados (ex.: {faltando[:3]})')

    rotulos = []
    fases = {}
    with open(args.chaves, encoding='utf-8') as f:
        for linha in f:
            col = linha.split()
            fases[col[7]] = fases.get(col[7], 0) + 1
            if col[7] == 'eval':
                rotulos.append((col[1], col[5], col[7]))
    if set(r[0] for r in rotulos) - set(ids):
        erro('há ids nas chaves que não estão no protocolo')

    escrever_tsv(os.path.join(args.saida, 'lista.tsv'), ['id', 'caminho'], zip(ids, caminhos))
    escrever_tsv(os.path.join(args.saida, 'rotulos.tsv'), ['id', 'rotulo', 'grupo'], rotulos)

    n_bona = sum(r[1] == 'bonafide' for r in rotulos)
    conferir_contagem('trials no protocolo', len(ids), 611829)
    conferir_contagem('trials eval', len(rotulos), 533928)
    conferir_contagem('genuínos eval', n_bona, 14869)
    itens = [
        ('trials no protocolo (lista)', len(ids)),
        ('fases nas chaves', fases),
        ('rotulados (fase eval)', f'{len(rotulos)} ({n_bona} genuínos, {len(rotulos) - n_bona} falsos)'),
    ]
    if not args.sem_checar_audio:
        itens.append(('taxas de amostragem (amostra de 200)', checar_taxas(caminhos)))
    itens.append(('saída', args.saida))
    resumo('preparar_asvspoof21df.py', itens)


if __name__ == '__main__':
    main()
