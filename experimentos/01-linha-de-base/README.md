# Experimento 01 – Linha de base acústica

Avaliação do detector publicado por Tak et al. (2022), XLS-R 300M + AASIST, em três
cenários com resultados já publicados: ASVspoof 2021 DF, In-the-Wild e BRSpeech-DF.
Não há treinamento. Este experimento é uma **avaliação do modelo publicado**, e não uma
reprodução por treinamento.

> Status: ASVspoof 2021 DF e In-the-Wild concluídos (2026-10-28). BRSpeech-DF pendente
> (download do dataset bloqueado pelo Google Drive). Os campos _(a preencher)_ dependem dele.

---

## 1. Ancoragem

**Área:** detecção de deepfakes de fala (*audio deepfake detection*, ADD), também
chamada de *spoofing countermeasures* (CM) no contexto de verificação de locutor.

**Linha de pesquisa:** generalização de detectores de deepfake de fala. A hipótese
da residência é que detectores que combinam evidências acústicas com evidências do
conteúdo semântico da fala generalizam melhor do que detectores apenas acústicos.

**Papel deste experimento:** estabelecer a linha de base acústica contra a qual a
hipótese será testada nos experimentos seguintes. Ele ainda não testa a hipótese.

## 2. Problema

Detectores treinados no ASVspoof 2019 LA atingem erros baixos em avaliações do mesmo
domínio, mas degradam quando o gerador, o domínio, o idioma ou as condições de gravação
diferem dos vistos no treino. Este experimento mede essa degradação para um detector
consolidado, em três níveis de distância do treino:

| Cenário | Distância do treino |
|---|---|
| ASVspoof 2021 DF | Mesmo protocolo e locutores da família VCTK, com novos vocoders e compressão |
| In-the-Wild | Inglês, áudio real da internet, figuras públicas, geradores desconhecidos |
| BRSpeech-DF | Outro idioma (português), TTS zero-shot modernos |

## 3. Frameworks de sustentação

- **Protocolo ASVspoof:** treino no ASVspoof 2019 LA e avaliação no ASVspoof 2021 DF,
  com as chaves oficiais (`DF-keys-stage-1`), considerando só a fase `eval`
  (533.928 trials).
- **EER (Equal Error Rate):** taxa em que a falsa aceitação de áudios falsos é igual à
  falsa rejeição de áudios genuínos. É a métrica principal. A métrica secundária é a
  acurácia no limiar fixo −3,5327, que é o limiar da EER do modelo no 2021 DF, usado
  pelo artigo do BRSpeech-DF.
- **wav2vec 2.0 / XLS-R 300M:** front-end auto-supervisionado, pré-treinado em fala
  multilíngue e ajustado em conjunto com o back-end.
- **AASIST:** back-end com atenção em grafos espectro-temporais.
- **RawBoost:** aumento de dados aplicado no treino do modelo publicado. Não é usado na
  avaliação.
- **Datasets:** ASVspoof 2021 DF, In-the-Wild (Müller et al., 2022) e BRSpeech-DF
  (Ferro Filho et al., 2025).

## 4. Artefato e resultados

### 4.1 Detector avaliado

- Modelo: SSL_Anti-spoofing (Tak et al., 2022), checkpoint `Best_LA_model_for_DF.pth`.
- Pré-processamento (conferido no código, `data_utils_SSL.py`): áudio a 16 kHz, primeiras
  64.600 amostras (~4 s) e, se for mais curto, preenchimento por **repetição**.
- Pontuação: logit da classe *bonafide*. Pontuação maior indica genuíno.

### 4.2 Resultados principais

| Cenário | Nº de áudios (gen./falsos) | EER de referência | Nossa EER | Acurácia em −3,5327 |
|---|---|---|---|---|
| ASVspoof 2021 DF (eval) | 14.869 / 519.059 | 2,85% [1] | **2,86%** | 97,15% |
| In-the-Wild | 19.963 / 11.816 | 10,46% [2], 11,2% [3] | **10,48%** | 88,97% |
| BRSpeech-DF (test) | 1.320 / 6.587 | 35,38% e acurácia de 70,22% [4] | _(a preencher)_ | _(a preencher)_ |

A EER do ASVspoof 2021 DF foi calculada com o script oficial `evaluate_2021_DF.py` e as
chaves `DF-keys-stage-1`, e conferida com o `metricas.py` (2,858%, limiar −3,5328). As
demais vêm do `metricas.py`, que usa a mesma função `compute_eer` do Tak.

Do 2021 DF para o In-the-Wild, a EER quase quadruplica. As pontuações médias mostram
de onde vem a perda: os falsos continuam com pontuação parecida (−3,64 no 2021 DF,
−3,61 no In-the-Wild), mas os genuínos caem de +2,17 para −0,53. O modelo passa a
desconfiar de fala real gravada fora das condições do treino.

### 4.3 Efeito do preenchimento de áudios curtos

O código do Tak preenche por repetição. O notebook de inferência do AKCIT preenche
com zeros.

| Cenário | EER (repetição) | EER (zeros) |
|---|---|---|
| In-the-Wild | 10,48% | 11,96% |
| BRSpeech-DF (test) | _(a preencher)_ | _(a preencher)_ |

No In-the-Wild, preencher com zeros piora a EER em 1,5 ponto e desloca as pontuações
para cima (média dos genuínos de −0,53 para +0,47). A escolha do preenchimento é,
portanto, uma fonte de variação entre resultados publicados com o mesmo modelo.

### 4.4 BRSpeech-DF por gerador (EER de todos os genuínos contra cada TTS)

| TTS | Referência (pontuações publicadas pelo AKCIT) | Nossa |
|---|---|---|
| ToucanTTS | 25,23% | _(a preencher)_ |
| Fish-Speech | 31,80% | _(a preencher)_ |
| YourTTS | 32,97% | _(a preencher)_ |
| XTTS | 40,30% | _(a preencher)_ |
| F5-TTS | 42,41% | _(a preencher)_ |

## 5. Validação

### 5.1 Validação da métrica (antes de usar GPU)

As pontuações publicadas pelos autores foram passadas pelos nossos scripts, sem
nenhuma inferência, para conferir a cadeia de métricas:

| Fonte | EER obtida | Referência |
|---|---|---|
| `Scores/DF/Scores_DF.txt` (Tak) | 2,85%, limiar −3,5327 | 2,85% |
| `.pkl` do AKCIT, BRSpeech-DF test | 35,38%, acurácia de 70,22% em −3,5327 | 35,38% / 70,22% |

As pontuações do 2021 DF publicadas pelo Tak e pelo AKCIT são idênticas (mesmo md5).

### 5.2 Validação da inferência

Comparação arquivo a arquivo entre as nossas pontuações e as publicadas
(`comparar_pontuacoes.py`):

| Cenário | Correlação | Diferença máx. | Mesma decisão em −3,5327 |
|---|---|---|---|
| ASVspoof 2021 DF (611.829 arquivos) | 0,9999996 | 0,076 (média 2,7·10⁻⁴) | 99,98% |
| BRSpeech-DF, repetição | _(a preencher)_ | _(a preencher)_ | _(a preencher)_ |
| BRSpeech-DF, zeros | _(a preencher)_ | _(a preencher)_ | _(a preencher)_ |

As diferenças no 2021 DF são numéricas, e não de método: com pré-processamento
diferente, a correlação cairia e muitas decisões mudariam. Uma causa provável, ainda
não verificada, é o TF32, que o PyTorch 1.8 usa por padrão nas multiplicações de
matrizes em GPUs Ampere/Ada (como a RTX 4090).

### 5.3 O que o resultado confirma ou contraria

- **Reprodução:** confirmada no 2021 DF (2,86% contra 2,85% publicados) e coerente
  com os números de terceiros no In-the-Wild (10,48% contra 10,46% e 11,2%).
- **Generalização:** a EER sobe de 2,86% para 10,48% ao sair do domínio do treino,
  e a perda se concentra nos áudios genuínos. BRSpeech-DF: _(a preencher)_
- **Implicações para a hipótese:** _(a preencher após o BRSpeech-DF)_

---

## Reprodutibilidade

### Versões

- Submódulos (`git submodule status`):
  - `third_party/SSL_Anti-spoofing` @ `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`
  - `third_party/BRSpeech-DF-Dataset` @ `13c9ad4ba64bb54cd0b9f6e0c415dd577d346db6`
- Checkpoints (SHA-256):
  - `Best_LA_model_for_DF.pth` (1.271.642.081 bytes):
    `1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804`
  - `xlsr2_300m.pt` (3.808.868.242 bytes):
    `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9`
- Ambiente: imagem `residencia-exp01:pt181`, gerada por `docker/Dockerfile` com a base
  padrão `pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime`. Python 3.8.8, PyTorch 1.8.1,
  CUDA 11.1, cuDNN 8.0.5, fairseq 1.0.0a0 (commit a540213), librosa 0.9.1, numpy 1.22.4.
  GPU NVIDIA RTX 4090 (driver 580.173.02).
- Tempo de inferência: 45,1 min no 2021 DF e 2,4 min em cada rodada do In-the-Wild
  (~220 arquivos/s). Os detalhes de cada execução, incluindo o SHA-256 de cada arquivo de
  pontuações, estão em `resultados/**/*.meta.json`.

### Dados

| Dataset | Origem |
|---|---|
| ASVspoof 2021 DF eval | Zenodo, registro 4835108 (4 partes `.tar.gz`, md5 conferido); 611.829 `.flac` a 16 kHz |
| In-the-Wild | Hugging Face `mueller91/In-The-Wild` (`release_in_the_wild.zip`, 8.161.489.023 bytes); 31.779 `.wav` a 16 kHz |
| BRSpeech-DF | Google Drive do AKCIT (link no README do submódulo) |

O `meta.csv` do In-the-Wild tem 54 nomes de locutor distintos, e não os 58 citados no
artigo. A diferença não afeta a EER, que não usa o locutor.

### Integridade dos submódulos

Os submódulos nunca são alterados: `third_party/` é montado somente leitura no
container, e `scripts/verificar_submodulos.sh` é executado antes e depois de cada
cenário.

### Comandos

No servidor, a partir da raiz do repositório, com `config/servidor.env` apontando para
as pastas de dados e checkpoints:

```bash
# Ambiente e checkpoints
docker build -f experimentos/01-linha-de-base/docker/Dockerfile -t residencia-exp01:pt181 .
bash experimentos/01-linha-de-base/scripts/baixar_checkpoints.sh ~/matheus/checkpoints

S=experimentos/01-linha-de-base/scripts/em_docker.sh
CK="--checkpoint /ckpt/Best_LA_model_for_DF.pth --xlsr /ckpt/xlsr2_300m.pt"

# ASVspoof 2021 DF
bash $S nenhuma python preparar_asvspoof21df.py --audio /dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac --saida ../listas/asvspoof21df
bash $S 0 python inferir.py --lista ../listas/asvspoof21df/lista.tsv --workers 8 $CK --saida ../resultados/asvspoof21df/pontuacoes_repeat.txt

# In-the-Wild (dois preenchimentos)
bash $S nenhuma python preparar_itw.py --raiz /dados/in_the_wild/release_in_the_wild --saida ../listas/in_the_wild
bash $S 0 python inferir.py --lista ../listas/in_the_wild/lista.tsv --workers 8 $CK --padding repeat --saida ../resultados/in_the_wild/pontuacoes_repeat.txt
bash $S 0 python inferir.py --lista ../listas/in_the_wild/lista.tsv --workers 8 $CK --padding zero --saida ../resultados/in_the_wild/pontuacoes_zero.txt

# Métricas
TP=../../../third_party/SSL_Anti-spoofing
R=../resultados
bash $S nenhuma python -B $TP/evaluate_2021_DF.py $R/asvspoof21df/pontuacoes_repeat.txt $TP/DF-keys-stage-1/keys eval
bash $S nenhuma python metricas.py --pontuacoes $R/asvspoof21df/pontuacoes_repeat.txt --rotulos ../listas/asvspoof21df/rotulos.tsv --permitir-extras --saida $R/asvspoof21df/metricas_repeat.json
bash $S nenhuma python comparar_pontuacoes.py --nossas $R/asvspoof21df/pontuacoes_repeat.txt --referencia $TP/Scores/DF/Scores_DF.txt --saida $R/asvspoof21df/comparacao_tak.json
bash $S nenhuma python metricas.py --pontuacoes $R/in_the_wild/pontuacoes_repeat.txt --rotulos ../listas/in_the_wild/rotulos.tsv --saida $R/in_the_wild/metricas_repeat.json
bash $S nenhuma python metricas.py --pontuacoes $R/in_the_wild/pontuacoes_zero.txt --rotulos ../listas/in_the_wild/rotulos.tsv --saida $R/in_the_wild/metricas_zero.json

# Antes e depois de cada etapa
bash experimentos/01-linha-de-base/scripts/verificar_submodulos.sh
```

Um teste rápido com 56 arquivos (`--limite 56`, em `resultados/teste_rapido/`) foi feito
antes da execução completa, para validar a GPU e comparar com as pontuações do Tak.

---

## Uso de LLM

Os scripts em `scripts/`, o ambiente em `docker/` e este README foram escritos com apoio
de um assistente de IA (Claude, da Anthropic), sob a minha orientação e revisão. As
decisões metodológicas, a execução e a interpretação dos resultados são de minha
responsabilidade.

---

## Referências

1. H. Tak et al. *Automatic speaker verification spoofing and deepfake detection using
   wav2vec 2.0 and data augmentation*. Odyssey 2022. README oficial:
   https://github.com/TakHemlata/SSL_Anti-spoofing
2. Y. Xiao, R. K. Das. *XLSR-Mamba: A Dual-Column Bidirectional State Space Model for
   Spoofing Attack Detection*. Tabela com XLS-R + AASIST no In-the-Wild: 10,46%.
3. _(a preencher: estudo de auditoria com o modelo oficial, 11,2%)_
4. A. C. Ferro Filho et al. *BRSpeech-DF: A Deep Fake Synthetic Speech Dataset for
   Portuguese Zero-Shot TTS*. EMNLP 2025.
5. N. Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.
6. J. Jung et al. *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
   Attention Networks*. ICASSP 2022.
7. H. Tak et al. *RawBoost: A Raw Data Boosting and Augmentation Method applied to
   Automatic Speaker Verification Anti-Spoofing*. ICASSP 2022.
8. A. Babu et al. *XLS-R: Self-supervised Cross-lingual Speech Representation Learning
   at Scale*. 2021.
