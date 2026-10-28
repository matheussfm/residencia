# Experimento 01 – Linha de base acústica

Avaliação do detector publicado por Tak et al. (2022), XLS-R 300M + AASIST, em três
cenários com resultados já publicados: ASVspoof 2021 DF, In-the-Wild e BRSpeech-DF.
Não há treinamento. Este experimento é uma **avaliação do modelo publicado**, e não uma
reprodução por treinamento.

> Status: _em andamento_. Os campos marcados com _(a preencher)_ dependem das execuções.

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
| ASVspoof 2021 DF (eval) | 14.869 / 519.059 | 2,85% [1] | _(a preencher)_ | _(a preencher)_ |
| In-the-Wild | _(a preencher)_ | ~10–11% [2][3], aproximada | _(a preencher)_ | _(a preencher)_ |
| BRSpeech-DF (test) | 1.320 / 6.587 | 35,38% e acurácia de 70,22% [4] | _(a preencher)_ | _(a preencher)_ |

### 4.3 Efeito do preenchimento de áudios curtos

O código do Tak preenche por repetição. O notebook de inferência do AKCIT preenche
com zeros.

| Cenário | EER (repetição) | EER (zeros) |
|---|---|---|
| In-the-Wild | _(a preencher)_ | _(a preencher)_ |
| BRSpeech-DF (test) | _(a preencher)_ | _(a preencher)_ |

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
| ASVspoof 2021 DF | _(a preencher)_ | _(a preencher)_ | _(a preencher)_ |
| BRSpeech-DF, repetição | _(a preencher)_ | _(a preencher)_ | _(a preencher)_ |
| BRSpeech-DF, zeros | _(a preencher)_ | _(a preencher)_ | _(a preencher)_ |

### 5.3 O que o resultado confirma ou contraria

_(a preencher após as execuções)_

- Reprodução: a execução reproduz os números publicados?
- Generalização: tamanho da perda do 2021 DF para o In-the-Wild e para o BRSpeech-DF.
- Implicações para a hipótese: _(a preencher)_

---

## Reprodutibilidade

### Versões

- Submódulos (`git submodule status`):
  - `third_party/SSL_Anti-spoofing` @ `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`
  - `third_party/BRSpeech-DF-Dataset` @ `13c9ad4ba64bb54cd0b9f6e0c415dd577d346db6`
- Checkpoints (SHA-256):
  - `Best_LA_model_for_DF.pth`: _(a preencher)_
  - `xlsr2_300m.pt`: _(a preencher)_
- Ambiente: imagem de `docker/Dockerfile`, com base _(a preencher: pt181 ou pt1131)_,
  PyTorch _(a preencher)_, CUDA _(a preencher)_, GPU NVIDIA RTX 4090. Os detalhes de
  cada execução estão em `resultados/**/*.meta.json`.

### Dados

| Dataset | Origem |
|---|---|
| ASVspoof 2021 DF eval | Zenodo, registro 4835108 (4 partes `.tar.gz`) |
| In-the-Wild | Hugging Face `mueller91/In-The-Wild` (`release_in_the_wild.zip`) |
| BRSpeech-DF | Google Drive do AKCIT (link no README do submódulo) |

### Integridade dos submódulos

Os submódulos nunca são alterados: `third_party/` é montado somente leitura no
container, e `scripts/verificar_submodulos.sh` é executado antes e depois de cada
cenário.

### Comandos

_(a preencher com os comandos exatos usados no servidor)_

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
