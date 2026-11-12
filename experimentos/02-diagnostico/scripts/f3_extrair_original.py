"""Fase 3, controle: representações do XLS-R ORIGINAL (pré-treinado, antes do ajuste no ASVspoof).

O detector do Tak ajusta o XLS-R 300M junto com o AASIST no ASVspoof 2019 LA. Para saber se esse
ajuste destruiu informação que generalizava, as mesmas representações por camada da Fase 3 são
extraídas do XLS-R original (xlsr2_300m.pt) e passam pelo mesmo probing (f3_probing.py).

- Mesmos áudios (lista da Fase 3) e mesmo pré-processamento do Exp. 01 (16 kHz, 64.600 amostras,
  preenchimento por repetição).
- Se a configuração do checkpoint indicar que o pré-treino usou o áudio normalizado
  (task.normalize), a mesma normalização é aplicada (o Tak não normaliza, mas o modelo dele foi
  ajustado assim; o original deve receber a entrada que viu no pré-treino). A escolha fica no
  .meta.json.
- Captura por hooks nas 24 camadas do transformer e a saída final (features_only), média no tempo.

Saída: .npz com ids, camadas (N × 24 × 1024, float16) e final (N × 1024).

Exemplo (no servidor):
    bash experimentos/01-linha-de-base/scripts/em_docker.sh 0 python ../../02-diagnostico/scripts/f3_extrair_original.py \
        --lista ../../02-diagnostico/listas/fase3/lista.tsv --xlsr /ckpt/xlsr2_300m.pt \
        --saida ../../02-diagnostico/resultados/fase3/features_xlsr_original.npz
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))

import numpy as np


def extrair(modelo, carregador, normalizar, dispositivo):
    import torch
    import torch.nn.functional as Fn
    camadas = modelo.encoder.layers
    buf = {}
    alcas = [c.register_forward_hook(lambda m, e, s, i=i: buf.__setitem__(
        i, (s[0] if isinstance(s, tuple) else s).mean(dim=0).detach().float().cpu().numpy()))
        for i, c in enumerate(camadas)]
    out = {'ids': [], 'camadas': [], 'final': []}
    t0, n = time.time(), 0
    try:
        with torch.no_grad():
            for i, (x, ids) in enumerate(carregador):
                x = x.to(dispositivo)
                if normalizar:
                    x = Fn.layer_norm(x, x.shape[1:])
                r = modelo(x, mask=False, features_only=True)
                out['ids'] += list(ids)
                out['camadas'].append(np.stack([buf[k] for k in range(len(camadas))], axis=1).astype(np.float16))
                out['final'].append(r['x'].mean(dim=1).detach().float().cpu().numpy().astype(np.float16))
                buf.clear()
                n += len(ids)
                if i % 50 == 0:
                    print(f'  {n} áudios ({n / max(time.time() - t0, 1e-9):.0f}/s)', flush=True)
    finally:
        for a in alcas:
            a.remove()
    return {'ids': np.array(out['ids']), 'camadas': np.concatenate(out['camadas']), 'final': np.concatenate(out['final'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lista', required=True)
    ap.add_argument('--xlsr', required=True)
    ap.add_argument('--saida', required=True)
    ap.add_argument('--lote', type=int, default=14)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dispositivo', default='cuda')
    ap.add_argument('--limite', type=int, default=None)
    ap.add_argument('--normalizar', choices=['auto', 'sim', 'nao'], default='auto')
    args = ap.parse_args()

    import fairseq
    import torch
    from torch.utils.data import DataLoader
    from _comum import garantir_fora_third_party, ler_tsv, sha256
    from inferir import ListaAudios

    garantir_fora_third_party(args.saida)
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    itens = [(r['id'], r['caminho']) for r in ler_tsv(args.lista)]
    if args.limite:
        itens = itens[:args.limite]
    modelos, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.xlsr])
    modelo = modelos[0].to(args.dispositivo).eval()
    norm_cfg = None
    for fonte in (getattr(task, 'cfg', None), getattr(cfg, 'task', None) if cfg is not None else None):
        if fonte is not None and hasattr(fonte, 'normalize'):
            norm_cfg = bool(fonte.normalize)
            break
    normalizar = norm_cfg if args.normalizar == 'auto' else (args.normalizar == 'sim')
    print(f'task.normalize no checkpoint: {norm_cfg}; normalizando a entrada: {normalizar}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    car = DataLoader(ListaAudios(itens, 'repeat'), batch_size=args.lote, shuffle=False, num_workers=args.workers)
    r = extrair(modelo, car, bool(normalizar), args.dispositivo)
    if list(r['ids']) != [i for i, _ in itens]:
        raise RuntimeError('ordem dos ids mudou')
    np.savez(args.saida, **r)
    with open(args.saida + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'lista': args.lista, 'n': len(itens), 'xlsr_sha256': sha256(args.xlsr),
                   'task_normalize_checkpoint': norm_cfg, 'entrada_normalizada': bool(normalizar),
                   'saida_sha256': sha256(args.saida), 'formas': {k: list(v.shape) for k, v in r.items()},
                   'torch': torch.__version__, 'fairseq': getattr(fairseq, '__version__', '?'),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, f, indent=2, ensure_ascii=False)
    print(f"ok: {len(itens)} áudios; camadas {r['camadas'].shape}; {args.saida}")


if __name__ == '__main__':
    main()
