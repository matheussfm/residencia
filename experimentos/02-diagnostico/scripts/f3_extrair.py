"""Fase 3: extrai as representações internas do detector do Tak, sem modificá-lo.

Para cada áudio da lista (gerada por f3_selecionar.py), com o MESMO pré-processamento do
Experimento 01 (16 kHz, primeiras 64.600 amostras, preenchimento por repetição), grava:
  - camadas: média no tempo da saída de cada uma das 24 camadas do transformer do XLS-R
             (N × 24 × 1024);
  - final:   média no tempo da representação que entra no back-end (saída do XLS-R após a
             normalização final; entrada da camada LL) (N × 1024);
  - aasist:  o embedding final do AASIST, a entrada da última camada linear (N × 160);
  - pontuacao: a pontuação do modelo (logit da classe genuíno), para conferir com o Exp. 01.

As representações são capturadas com hooks do PyTorch (register_forward_hook e
register_forward_pre_hook). O código do submódulo é importado como está, pelo mesmo
carregar_modelo() do inferir.py do Experimento 01.

Saída: um .npz (float16) fora do controle de versão (*.npz no .gitignore).

Exemplo (no servidor, GPU 0):
    bash experimentos/01-linha-de-base/scripts/em_docker.sh 0 python ../../02-diagnostico/scripts/f3_extrair.py \
        --lista ../../02-diagnostico/listas/fase3/lista.tsv \
        --checkpoint /ckpt/Best_LA_model_for_DF.pth --xlsr /ckpt/xlsr2_300m.pt \
        --saida ../../02-diagnostico/resultados/fase3/features.npz
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', '01-linha-de-base', 'scripts'))

import numpy as np


class Capturas:
    """Registra os hooks e guarda, por lote, as médias no tempo."""

    def __init__(self, modelo):
        self.buf = {}
        self.alcas = []
        camadas = modelo.ssl_model.model.encoder.layers
        self.n_camadas = len(camadas)
        for i, camada in enumerate(camadas):
            # saída da camada do transformer: (x, attn), x em T × B × C
            self.alcas.append(camada.register_forward_hook(self._camada(i)))
        # entrada da LL: B × T × 1024 (saída final do XLS-R)
        self.alcas.append(modelo.LL.register_forward_pre_hook(
            lambda m, ent: self.buf.__setitem__('final', ent[0].mean(dim=1).detach().float().cpu().numpy())))
        # entrada da out_layer: B × 160 (embedding final do AASIST)
        self.alcas.append(modelo.out_layer.register_forward_pre_hook(
            lambda m, ent: self.buf.__setitem__('aasist', ent[0].detach().float().cpu().numpy())))

    def _camada(self, i):
        def f(m, ent, saida):
            x = saida[0] if isinstance(saida, tuple) else saida
            self.buf[i] = x.mean(dim=0).detach().float().cpu().numpy()
        return f

    def lote(self):
        camadas = np.stack([self.buf[i] for i in range(self.n_camadas)], axis=1)  # B × L × C
        out = {'camadas': camadas, 'final': self.buf['final'], 'aasist': self.buf['aasist']}
        self.buf = {}
        return out

    def remover(self):
        for a in self.alcas:
            a.remove()


def extrair(modelo, carregador, dispositivo):
    """Roda o modelo e devolve dict com ids, pontuacao, camadas, final, aasist (float16)."""
    import torch
    cap = Capturas(modelo)
    partes = {'ids': [], 'pontuacao': [], 'camadas': [], 'final': [], 'aasist': []}
    t0 = time.time()
    n = 0
    try:
        with torch.no_grad():
            for i, (x, ids) in enumerate(carregador):
                saida = modelo(x.to(dispositivo))
                if i == 0 and (modelo.training or modelo.ssl_model.model.training):
                    raise RuntimeError('modelo saiu do modo eval durante o forward')
                b = cap.lote()
                partes['ids'] += list(ids)
                partes['pontuacao'].append(saida[:, 1].detach().float().cpu().numpy())
                for k in ('camadas', 'final', 'aasist'):
                    partes[k].append(b[k].astype(np.float16))
                n += len(ids)
                if i % 50 == 0:
                    print(f'  {n} áudios ({n / max(time.time() - t0, 1e-9):.0f}/s)', flush=True)
    finally:
        cap.remover()
    return {'ids': np.array(partes['ids']), 'pontuacao': np.concatenate(partes['pontuacao']),
            **{k: np.concatenate(partes[k]) for k in ('camadas', 'final', 'aasist')}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lista', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--xlsr', required=True)
    ap.add_argument('--saida', required=True)
    ap.add_argument('--lote', type=int, default=14)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dispositivo', default='cuda')
    ap.add_argument('--limite', type=int, default=None, help='só os N primeiros (teste rápido)')
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from _comum import garantir_fora_third_party, ler_tsv, sha256
    from inferir import ListaAudios, carregar_modelo  # mesmo pré-processamento e carregamento do Exp. 01

    garantir_fora_third_party(args.saida)
    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    itens = [(r['id'], r['caminho']) for r in ler_tsv(args.lista)]
    if args.limite:
        itens = itens[:args.limite]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    modelo = carregar_modelo(args.checkpoint, args.xlsr, args.dispositivo)
    carregador = DataLoader(ListaAudios(itens, 'repeat'), batch_size=args.lote, shuffle=False,
                            num_workers=args.workers)
    r = extrair(modelo, carregador, args.dispositivo)
    if list(r['ids']) != [i for i, _ in itens]:
        raise RuntimeError('ordem dos ids mudou')
    np.savez(args.saida, **r)
    import json
    with open(args.saida + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'lista': args.lista, 'n': len(itens), 'checkpoint_sha256': sha256(args.checkpoint),
                   'xlsr_sha256': sha256(args.xlsr), 'saida_sha256': sha256(args.saida),
                   'formas': {k: list(v.shape) for k, v in r.items()}, 'torch': torch.__version__,
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                  f, indent=2, ensure_ascii=False)
    print(f"ok: {len(itens)} áudios; camadas {r['camadas'].shape}; aasist {r['aasist'].shape}; {args.saida}")


if __name__ == '__main__':
    main()
