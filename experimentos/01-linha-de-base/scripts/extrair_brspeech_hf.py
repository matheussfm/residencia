"""Extrai os áudios da divisão de teste do BRSpeech-DF a partir dos parquet do Hugging Face.

Fonte: https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF (configurações
"bonafide" e "spoof", revisão 6a2c47eb228511f7a6b3d67c9c0797d5be8a58e8). Nessa versão
os áudios foram reamostrados para 24 kHz pelos autores.

Os bytes de cada áudio são gravados exatamente como estão no parquet, sem recodificar.
- genuínos: bonafide/<nome original>.flac (o parquet traz o nome do arquivo)
- falsos:   spoof/<modelo>/<modelo>_<n>.wav, em que n é a posição da linha no parquet
            dentro do modelo (o parquet não traz o nome original dos falsos)
Grava também <saida>/indice.tsv, usado por preparar_brspeech_hf.py.

Roda fora do container (precisa de pyarrow: pip3 install --user pyarrow), porque grava
na pasta de dados, que o container monta somente leitura.

Exemplo:
    python3 -B extrair_brspeech_hf.py --parquets ~/matheus/dados/brspeech_df/hf \
        --saida ~/matheus/dados/brspeech_df/hf_test
"""
import argparse
import glob
import os
import sys

sys.dont_write_bytecode = True

import pyarrow.parquet as pq

from _comum import erro, escrever_tsv, garantir_fora_third_party, resumo

ROTULOS = {0: 'bonafide', 1: 'spoof'}
EXTENSOES = {b'fLaC': '.flac', b'RIFF': '.wav'}


def extensao(dados):
    ext = EXTENSOES.get(bytes(dados[:4]))
    if ext is None:
        erro(f'formato de áudio desconhecido (primeiros bytes: {bytes(dados[:4])!r})')
    return ext


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquets', required=True,
                    help='pasta com bonafide-test-*.parquet e spoof-test-*.parquet')
    ap.add_argument('--saida', required=True, help='pasta onde gravar os áudios e o indice.tsv')
    args = ap.parse_args()
    garantir_fora_third_party(args.saida)

    arquivos = (sorted(glob.glob(os.path.join(args.parquets, 'bonafide-test-*.parquet')))
                + sorted(glob.glob(os.path.join(args.parquets, 'spoof-test-*.parquet'))))
    if not arquivos:
        erro(f'nenhum parquet encontrado em {args.parquets}')

    indice, por_modelo, formatos = [], {}, {}
    for arq in arquivos:
        classe_cfg = os.path.basename(arq).split('-')[0]  # bonafide ou spoof
        tabela = pq.read_table(arq)
        for linha, r in enumerate(tabela.to_pylist()):
            rotulo = ROTULOS[r['label']]
            if rotulo != classe_cfg:
                erro(f'{arq}:{linha}: rótulo {rotulo} numa configuração {classe_cfg}')
            dados = r['audio']['bytes']
            ext = extensao(dados)
            if rotulo == 'bonafide':
                nome_hf = r['audio']['path']
                if not nome_hf:
                    erro(f'{arq}:{linha}: genuíno sem nome de arquivo')
                modelo = 'bonafide'
                rel = f'bonafide/{os.path.splitext(nome_hf)[0]}{ext}'
            else:
                nome_hf = r['audio'].get('path') or ''
                modelo = r['model']
                n = por_modelo.get(modelo, 0)
                rel = f'spoof/{modelo}/{modelo}_{n:05d}{ext}'
            por_modelo[modelo] = por_modelo.get(modelo, 0) + 1
            formatos[(modelo, ext)] = formatos.get((modelo, ext), 0) + 1

            destino = os.path.join(args.saida, rel)
            if not (os.path.exists(destino) and os.path.getsize(destino) == len(dados)):
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                with open(destino + '.tmp', 'wb') as f:
                    f.write(dados)
                os.replace(destino + '.tmp', destino)
            indice.append((rel, rotulo, modelo, os.path.basename(arq), linha, nome_hf))
        print(f'  {os.path.basename(arq)}: {tabela.num_rows} linhas')

    if len({i[0] for i in indice}) != len(indice):
        erro('caminhos repetidos na extração')
    escrever_tsv(os.path.join(args.saida, 'indice.tsv'),
                 ['caminho', 'rotulo', 'modelo', 'parquet', 'linha', 'nome_hf'], indice)
    resumo('extrair_brspeech_hf.py', [
        ('arquivos parquet', len(arquivos)),
        ('áudios extraídos', len(indice)),
        ('por modelo', por_modelo),
        ('formatos', {f'{m}{e}': n for (m, e), n in sorted(formatos.items())}),
        ('saída', args.saida),
    ])


if __name__ == '__main__':
    main()
