"""Inferência do detector SSL_Anti-spoofing (Tak et al., 2022) sobre uma lista de áudios.

Usa o código original do submódulo third_party/SSL_Anti-spoofing sem modificá-lo
(montado somente leitura no Docker) e o mesmo pré-processamento da avaliação original:
áudio a 16 kHz, corte nas primeiras 64.600 amostras e preenchimento por repetição.

Entrada: lista TSV com colunas `id` e `caminho` (gerada pelos scripts preparar_*.py).
Saída: arquivo texto "id pontuacao" por linha, no formato do Tak (aceito pelo
evaluate_2021_DF.py). Pontuação = logit da classe bonafide; maior => genuíno.
Grava também <saida>.meta.json com versões, hashes e tempos.

Com --sem-corte, os áudios mais longos que 64.600 amostras são usados inteiros (até
--max-segundos), em vez de cortados. Os mais curtos continuam preenchidos até 64.600
amostras, como no original, de modo que só o efeito do corte muda. Nesse modo os áudios
são processados um por vez, porque têm durações diferentes.

Exemplo:
    python inferir.py --lista listas/brspeech_test/lista.tsv \
        --saida resultados/brspeech_test/pontuacoes_repeat.txt \
        --checkpoint /ckpt/Best_LA_model_for_DF.pth --xlsr /ckpt/xlsr2_300m.pt
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # antes de importar qualquer coisa do submódulo

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from _comum import RAIZ_REPO, SSL_REPO, erro, garantir_fora_third_party, ler_tsv, resumo, sha256

AMOSTRAS = 64600  # ~4 s a 16 kHz, igual a Dataset_ASVspoof2021_eval.cut


# ---------------------------------------------------------------------------
# Copiado de third_party/SSL_Anti-spoofing/data_utils_SSL.py, sem alterações.
# MIT License, Copyright (c) 2022 Hemlata (https://github.com/TakHemlata/SSL_Anti-spoofing)
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
# ---------------------------------------------------------------------------


def pad_zeros(x, max_len=64600):
    """Variante com zeros, equivalente à função `pad` do notebook
    models/ssl_antispoof/notebooks/SSL-inference.ipynb do BRSpeech-DF-Dataset (AKCIT)."""
    if len(x) >= max_len:
        return x[:max_len]
    return np.pad(x, (0, max_len - len(x)), mode='constant')


PADDINGS = {'repeat': pad, 'zero': pad_zeros}


class ListaAudios(Dataset):
    def __init__(self, itens, padding, max_amostras=None):
        self.itens = itens
        self.pad = PADDINGS[padding]
        self.max_amostras = max_amostras  # None = corte original de AMOSTRAS

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        id_, caminho = self.itens[i]
        x, _ = librosa.load(caminho, sr=16000)
        if x.shape[0] == 0:
            raise RuntimeError(f'áudio vazio: {caminho}')
        if self.max_amostras is None or x.shape[0] < AMOSTRAS:
            return torch.Tensor(self.pad(x, AMOSTRAS)), id_
        return torch.Tensor(x[:self.max_amostras]), id_


def carregar_modelo(checkpoint, xlsr, dispositivo):
    if os.path.basename(xlsr) != 'xlsr2_300m.pt':
        erro('--xlsr deve apontar para um arquivo chamado xlsr2_300m.pt (nome fixo em model.py)')
    sys.path.insert(0, SSL_REPO)
    from model import Model  # código original do Tak, sem modificações

    # model.py carrega 'xlsr2_300m.pt' relativo ao diretório atual; mudamos de diretório
    # só durante a construção, em vez de editar o arquivo do submódulo.
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(xlsr)))
    try:
        modelo = Model(None, dispositivo)
    finally:
        os.chdir(cwd)

    estado = torch.load(checkpoint, map_location='cpu')
    # O checkpoint foi salvo a partir de nn.DataParallel (chaves com prefixo "module.").
    if all(k.startswith('module.') for k in estado):
        estado = {k[len('module.'):]: v for k, v in estado.items()}
    faltando, sobrando = modelo.load_state_dict(estado, strict=False)
    if faltando or sobrando:
        erro(f'checkpoint incompatível com o modelo.\n  faltando: {faltando}\n  sobrando: {sobrando}')
    # Mover para o dispositivo ANTES do forward: SSLModel.extract_feat chama .train() no XLS-R
    # se ele estiver em outro device/dtype, o que ligaria o dropout.
    modelo.to(dispositivo).eval()
    return modelo


def ler_parcial(caminho):
    """Lê um .partial de uma execução interrompida, descartando uma última linha incompleta."""
    feitos = {}
    if os.path.exists(caminho):
        with open(caminho, encoding='utf-8') as f:
            for linha in f:
                partes = linha.split()
                if len(partes) == 2:
                    try:
                        feitos[partes[0]] = float(partes[1])
                    except ValueError:
                        pass
    return feitos


def commit_submodulo():
    try:
        saida = subprocess.run(['git', '-C', RAIZ_REPO, 'ls-tree', 'HEAD', 'third_party/SSL_Anti-spoofing'],
                               capture_output=True, text=True, timeout=30).stdout.split()
        return saida[2] if len(saida) >= 3 else 'desconhecido'
    except Exception:
        return 'desconhecido'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lista', required=True, help='TSV com colunas id e caminho')
    ap.add_argument('--saida', required=True, help='arquivo de pontuações "id pontuacao"')
    ap.add_argument('--checkpoint', required=True, help='Best_LA_model_for_DF.pth')
    ap.add_argument('--xlsr', required=True, help='caminho do xlsr2_300m.pt')
    ap.add_argument('--padding', choices=sorted(PADDINGS), default='repeat',
                    help='repeat = original do Tak (padrão); zero = como o notebook do AKCIT')
    ap.add_argument('--lote', type=int, default=14, help='tamanho do lote (14 = avaliação do Tak)')
    ap.add_argument('--workers', type=int, default=4, help='processos do DataLoader')
    ap.add_argument('--dispositivo', default='cuda')
    ap.add_argument('--sem-corte', action='store_true',
                    help='usa os áudios longos inteiros em vez de cortar em 64.600 amostras')
    ap.add_argument('--max-segundos', type=float, default=60.0,
                    help='com --sem-corte, limite de duração por áudio (memória); padrão 60 s')
    ap.add_argument('--limite', type=int, default=None, help='processa só os N primeiros (teste rápido)')
    ap.add_argument('--sobrescrever', action='store_true', help='permite substituir uma saída existente')
    args = ap.parse_args()

    garantir_fora_third_party(args.saida)
    if os.path.exists(args.saida) and not args.sobrescrever:
        erro(f'{args.saida} já existe (use --sobrescrever)')
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    if args.dispositivo.startswith('cuda') and not torch.cuda.is_available():
        erro('CUDA indisponível: confira o --gpus do docker run')

    itens = [(r['id'], r['caminho']) for r in ler_tsv(args.lista)]
    if args.limite:
        itens = itens[:args.limite]
    parcial = args.saida + '.partial'
    feitos = ler_parcial(parcial)
    pendentes = [it for it in itens if it[0] not in feitos]
    print(f'{len(itens)} arquivos na lista; {len(feitos)} já feitos (.partial); {len(pendentes)} a processar')

    # Mesmo ajuste de cuDNN do Tak (core_scripts/startup_config.set_random_seed com os padrões do main).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t0 = time.time()
    modelo = carregar_modelo(args.checkpoint, args.xlsr, args.dispositivo)
    print(f'modelo carregado em {time.time() - t0:.0f}s')

    # Reescreve o .partial só com linhas válidas e continua a partir dele.
    with open(parcial, 'w', encoding='utf-8') as f:
        f.writelines(f'{k} {v}\n' for k, v in feitos.items())

    max_amostras = int(args.max_segundos * 16000) if args.sem_corte else None
    lote = 1 if args.sem_corte else args.lote
    carregador = DataLoader(ListaAudios(pendentes, args.padding, max_amostras), batch_size=lote,
                            shuffle=False, drop_last=False, num_workers=args.workers)
    t1 = time.time()
    n = 0
    with torch.no_grad(), open(parcial, 'a', encoding='utf-8') as f:
        for i, (x, ids) in enumerate(carregador):
            saida = modelo(x.to(args.dispositivo))
            if i == 0 and (modelo.training or modelo.ssl_model.model.training):
                erro('modelo saiu do modo eval durante o forward (dropout ligado)')
            pont = saida[:, 1].data.cpu().numpy().ravel().tolist()
            for id_, p in zip(ids, pont):
                feitos[id_] = p
                f.write(f'{id_} {p}\n')
            f.flush()
            n += len(ids)
            if i % (200 * args.lote // lote) == 0:
                dt = time.time() - t1
                print(f'  {n}/{len(pendentes)}  ({n / max(dt, 1e-9):.0f} arquivos/s)', flush=True)
    t_inf = time.time() - t1

    faltando = [id_ for id_, _ in itens if id_ not in feitos]
    if faltando:
        erro(f'{len(faltando)} arquivos sem pontuação (ex.: {faltando[:3]})')
    # Saída final na ordem da lista.
    tmp = args.saida + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.writelines(f'{id_} {feitos[id_]}\n' for id_, _ in itens)
    os.replace(tmp, args.saida)
    os.remove(parcial)

    import fairseq
    meta = {
        'data': datetime.datetime.now().isoformat(timespec='seconds'),
        'host': platform.node(),
        'lista': args.lista,
        'saida': args.saida,
        'n_arquivos': len(itens),
        'padding': args.padding,
        'lote': lote,
        'amostras': AMOSTRAS,
        'sem_corte': args.sem_corte,
        'max_amostras': max_amostras,
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': sha256(args.checkpoint),
        'xlsr_sha256': sha256(args.xlsr),
        'saida_sha256': sha256(args.saida),
        'commit_ssl_anti_spoofing': commit_submodulo(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'fairseq': getattr(fairseq, '__version__', '?'),
        'librosa': librosa.__version__,
        'numpy': np.__version__,
        'segundos_inferencia': round(t_inf, 1),
    }
    with open(args.saida + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    p = np.array([feitos[id_] for id_, _ in itens])
    resumo('inferir.py', [
        ('arquivos', len(itens)),
        ('padding', args.padding),
        ('corte', f'sem corte (até {args.max_segundos:g} s)' if args.sem_corte else f'{AMOSTRAS} amostras'),
        ('gpu', meta['gpu']),
        ('torch / cuda', f"{meta['torch']} / {meta['cuda']}"),
        ('tempo de inferência', f'{t_inf / 60:.1f} min'),
        ('pontuação min / média / max', f'{p.min():.3f} / {p.mean():.3f} / {p.max():.3f}'),
        ('checkpoint sha256', meta['checkpoint_sha256'][:16] + '...'),
        ('saída', args.saida),
    ])


if __name__ == '__main__':
    main()
