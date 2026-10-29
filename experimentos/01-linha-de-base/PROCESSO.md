# Experimento 01 – Registro do processo

Este documento registra **como** o Experimento 01 foi feito: o que foi implementado, como
foi executado, os problemas encontrados, as decisões tomadas e por quê. Os resultados e as
instruções de reprodução estão no [README.md](README.md). Todo o trabalho foi feito em
2026-10-28.

---

## 1. Objetivo

Avaliar um detector de deepfake de fala puramente acústico, consolidado na literatura, em três
cenários com resultados já publicados, para:

1. conferir se a nossa execução reproduz os números publicados;
2. medir a perda de generalização fora do domínio de treino;
3. estabelecer a linha de base contra a qual a hipótese da residência será testada depois. A
   hipótese é que combinar evidências acústicas com evidências semânticas melhora a
   generalização.

O **detector** é o de Tak et al. (2022), Odyssey: front-end XLS-R 300M (wav2vec 2.0) e back-end
AASIST, treinado no ASVspoof 2019 LA. Usamos o checkpoint pré-treinado publicado pelos autores
(`Best_LA_model_for_DF.pth`). **Nada foi treinado**, então o experimento é uma *avaliação do
modelo publicado*, e não uma reprodução por treinamento.

Os **cenários**:

- ASVspoof 2021 DF, a avaliação oficial;
- In-the-Wild, em inglês, com áudio real da internet;
- BRSpeech-DF, em português, com TTS zero-shot modernos.

---

## 2. Ponto de partida

O repositório já tinha dois submódulos, que são os repositórios originais e nunca foram
alterados:

- `third_party/SSL_Anti-spoofing`, o repositório oficial do Tak. Traz o modelo, o
  pré-processamento, o script oficial de EER do 2021 DF, as chaves de avaliação e **as
  pontuações publicadas do modelo para cada arquivo do 2021 DF** (`Scores/DF/Scores_DF.txt`).
- `third_party/BRSpeech-DF-Dataset`, o repositório do AKCIT, autores do BRSpeech-DF. Traz o
  Dockerfile, os notebooks de inferência e avaliação, os metadados do dataset e **as pontuações
  publicadas do modelo para cada arquivo do BRSpeech-DF** (arquivos `.pkl`).

A regra mais importante do trabalho foi **nunca alterar nada dentro de `third_party/`**.

---

## 3. Análise inicial do código (antes de implementar)

A leitura do código original levantou pontos que orientaram toda a implementação:

| Ponto | O que o código faz | Consequência |
|---|---|---|
| Pré-processamento do Tak (`data_utils_SSL.py`) | 16 kHz; corte nas primeiras 64.600 amostras (~4 s); áudios curtos são completados **repetindo o próprio áudio** | Reproduzido sem alterações |
| Pré-processamento do AKCIT (notebook de inferência) | Completa áudios curtos **com zeros** | Diferença do Tak; rodamos as duas versões |
| Pontuação | Logit da classe "genuíno"; **maior = genuíno** | Confirmado com as pontuações publicadas |
| `main_SSL_DF.py` (script original de avaliação) | Grava a saída em modo de acréscimo (rodar duas vezes corrompe o arquivo), cria pastas dentro do repositório e é lento | Não foi usado; escrevemos o nosso script de inferência |
| Notebook do AKCIT | Carrega o modelo com `strict=False`, que ignora em silêncio pesos que não batem | Usamos carregamento estrito |
| Dockerfile do AKCIT | Usa PyTorch 1.8.1 + CUDA 11.1, baixa o código do Tak na versão do dia e abre o Jupyter ao iniciar | Escrevemos um Dockerfile próprio, baseado nele |
| `evaluate_2021_DF.py` (EER oficial) | Exige as 611.829 linhas e calcula a EER só na fase "eval" (533.928 trials) | Nossa saída segue exatamente esse formato |

**Validação da métrica antes de qualquer inferência:** passamos as pontuações publicadas pelos
autores pelo nosso cálculo de EER e obtivemos exatamente 2,85% (2021 DF) e 35,38% (BRSpeech-DF).
Assim, qualquer diferença posterior viria da inferência, e não da métrica.

---

## 4. O que foi implementado

Tudo fica em `experimentos/01-linha-de-base/`.

### 4.1 Scripts (`scripts/`)

| Script | Função |
|---|---|
| `inferir.py` | Roda o modelo sobre uma lista de áudios. Importa o modelo **original** do Tak direto do submódulo, sem modificá-lo. Usa o mesmo pré-processamento, com a opção `--padding repeat\|zero`. Carrega os pesos de forma estrita e garante que o modelo fica em modo de avaliação. Grava as pontuações no formato do Tak e um `.meta.json` com o SHA-256 dos checkpoints e da saída, as versões do ambiente, a GPU e o tempo. Retoma de onde parou se for interrompido. |
| `metricas.py` | Calcula a EER (com a função `compute_eer` copiada do Tak, mantendo o aviso de licença MIT), a acurácia no limiar fixo −3,5327 e as métricas por grupo (por exemplo, por TTS). Avisa se a EER passar de 50%, o que indicaria orientação invertida das pontuações. |
| `comparar_pontuacoes.py` | Compara, arquivo a arquivo, as nossas pontuações com as publicadas pelos autores: correlação, diferença máxima e média, e proporção de decisões iguais no limiar. Com `--mapa`, traduz ids quando os nomes diferem. |
| `preparar_asvspoof21df.py` | Monta a lista de áudios e os rótulos do 2021 DF a partir do protocolo e das chaves do submódulo. |
| `preparar_itw.py` | Monta a lista e os rótulos do In-the-Wild a partir do `meta.csv`, conferindo o formato. |
| `preparar_brspeech.py` | Monta a lista e os rótulos do BRSpeech-DF a partir dos metadados do AKCIT, para a versão original do dataset (Google Drive). Não chegou a ser usado, porque o download foi bloqueado. |
| `extrair_brspeech_hf.py` | Extrai os áudios do teste do BRSpeech-DF dos arquivos parquet do Hugging Face, sem recodificar. |
| `preparar_brspeech_hf.py` | Monta a lista e os rótulos do BRSpeech-DF na versão Hugging Face, e os "mapas" de correspondência com os ids do AKCIT. |
| `verificar_submodulos.sh` | Confere que os submódulos estão no commit registrado e sem nenhum arquivo alterado, novo ou ignorado. Foi rodado antes e depois de cada etapa. |
| `em_docker.sh` | Roda qualquer comando no container, com `third_party/` montado somente leitura. |
| `baixar_checkpoints.sh` | Baixa os checkpoints e registra o SHA-256 de cada um. |
| `_comum.py` | Funções compartilhadas, incluindo a trava que impede qualquer script de gravar dentro de `third_party/` e a leitura segura dos `.pkl` do AKCIT. |

### 4.2 Ambiente (`docker/`)

O Dockerfile é baseado no do AKCIT (PyTorch 1.8.1, CUDA 11.1, Python 3.8), com três mudanças:

- o código do Tak vem do submódulo, garantindo a versão registrada;
- o fairseq é copiado do submódulo para dentro da imagem;
- não há Jupyter nem checkpoints na imagem.

As versões estão fixadas: librosa 0.9.1, numpy 1.22.4 e outras.

### 4.3 Proteções para não alterar os repositórios originais

- `third_party/` é montado **somente leitura** no container, então qualquer tentativa de escrita
  falharia.
- O Python roda sem gravar cache (`PYTHONDONTWRITEBYTECODE`).
- Os scripts recusam qualquer caminho de saída dentro de `third_party/`.
- O fairseq é instalado a partir de uma cópia, e não da pasta do submódulo.
- `verificar_submodulos.sh` foi rodado antes e depois de cada execução. **Em todas as
  verificações, os dois submódulos estavam intactos.**

### 4.4 Testes antes do uso real

Antes de ir para o servidor, os scripts foram testados localmente, sem GPU:

- a sintaxe de todos;
- os preparadores, as métricas e o comparador sobre as pontuações publicadas, reproduzindo
  2,85% e 35,38%;
- a trava contra escrita em `third_party/`;
- o fluxo completo do BRSpeech-DF na versão Hugging Face, com dados simulados.

---

## 5. Infraestrutura e fluxo de trabalho

- **Máquina:** servidor da universidade com uma NVIDIA RTX 4090 (24 GB), acessado
  remotamente. Há uma segunda máquina com outra 4090, que não foi necessária.
- **Fluxo:** os scripts foram escritos no Windows, enviados ao GitHub, baixados no servidor
  (`git clone --recurse-submodules`) e executados lá. Os resultados voltaram ao PC por `scp`, sem
  guardar credenciais do GitHub no servidor, que é uma conta compartilhada.
- **Execuções longas** rodaram dentro do `tmux`, para não depender da conexão.
- **Organização no servidor:** `~/matheus/` com `dados/`, `checkpoints/` e `residencia/`.

---

## 6. Etapas de execução e verificações

| Etapa | Resultado |
|---|---|
| Download do ASVspoof 2021 DF (Zenodo, 4 partes, 34,5 GB) | md5 das 4 partes conferido; 611.829 `.flac` extraídos |
| Download do In-the-Wild (Hugging Face, 7,6 GB) | Tamanho conferido; 31.779 `.wav` (19.963 genuínos, 11.816 falsos) |
| Construção da imagem Docker | Sem erros |
| Download dos checkpoints | SHA-256 registrados |
| **Teste rápido** (56 áudios do 2021 DF) | PyTorch 1.8.1 funcionou na RTX 4090; decisões 100% iguais às do Tak |
| 2021 DF completo | 45 min (~220 áudios/s); EER oficial de **2,86%** |
| In-the-Wild, com repetição e com zeros | 2,4 min cada; **10,48%** e 11,96% |
| BRSpeech-DF (versão Hugging Face), com repetição e com zeros | 1,3 min cada; **39,24%** e 35,38% |

Todos os arquivos de pontuação foram conferidos por SHA-256 depois de copiados para o PC. A EER
do 2021 DF também foi recalculada de forma independente no PC.

---

## 7. Problemas encontrados e como foram resolvidos

1. **Dúvida sobre a compatibilidade da RTX 4090 com o ambiente antigo** (PyTorch 1.8.1 / CUDA
   11.1). Resolvida com o teste rápido, que confirmou que funciona. Havia uma imagem alternativa
   preparada (PyTorch 1.13), que não foi necessária.
2. **Download do BRSpeech-DF bloqueado pelo Google Drive** ("too many users"), mesmo numa cópia
   do arquivo no Drive do próprio usuário. Tentativas sem sucesso: o `gdown`, o link direto do
   navegador (que exige login) e a cópia própria. **Solução:** foi encontrada a versão que os
   próprios autores publicaram no Hugging Face (`AKCIT-Deepfake/BRSpeech-DF`). Ela permite baixar
   só a divisão de teste, com 3,5 GB, em vez dos 73 GB do pacote completo.
3. **Diferenças da versão Hugging Face:**
   - os áudios foram reamostrados para 24 kHz;
   - os falsos não trazem o nome original do arquivo.

   As duas diferenças foram **verificadas**:
   - a correspondência dos falsos foi inferida pela ordem alfabética, e a hipótese foi testada:
     a ordem certa dá correlação de 1,00 com as pontuações do AKCIT, e a errada, 0,11;
   - com isso, as nossas pontuações coincidem com as do AKCIT (correlação de 1,0000), o que
     mostra que a reamostragem não afeta o resultado.
4. **Pequena diferença numérica no 2021 DF:** 2,86% contra 2,85%, com correlação de 0,9999996 e
   99,98% das decisões iguais. É ruído numérico de GPU e biblioteca diferentes. Causa provável,
   não verificada: o modo TF32 do PyTorch em GPUs RTX 30/40.
5. **Detalhes dos dados registrados:**
   - o `meta.csv` do In-the-Wild tem 54 locutores, e não os 58 citados no artigo;
   - um arquivo de metadados citado num notebook do AKCIT (`brspeech_df_meta_fixed.csv`) não
     existe no repositório. Os metadados disponíveis foram suficientes.

---

## 8. Decisões metodológicas

| Decisão | Justificativa |
|---|---|
| Usar o checkpoint publicado, sem treinar | O objetivo é uma linha de base comparável à literatura |
| Pré-processamento do Tak (repetição) como **resultado principal** | Fidelidade ao método original |
| Rodar também com zeros | Reproduzir o número do AKCIT e medir o efeito dessa escolha |
| Script próprio de inferência, e não o `main_SSL_DF.py` | Problemas práticos do original (seção 3); a fidelidade foi comprovada pela comparação arquivo a arquivo |
| EER do 2021 DF pelo **script oficial** | Comparabilidade com a literatura |
| EER como métrica principal; acurácia em −3,5327 como secundária | O limiar fixo vem do 2021 DF e não transfere bem para outros domínios |
| Comparar arquivo a arquivo com as pontuações publicadas | Prova muito mais forte de reprodução do que comparar só a EER |
| Versionar as pontuações no git | Permite reanalisar sem rodar o modelo de novo |

---

## 9. Resultados (resumo)

| Cenário | EER (repetição, Tak) | EER (zeros, AKCIT) | Referência publicada |
|---|---|---|---|
| ASVspoof 2021 DF | **2,86%** | — | 2,85% |
| In-the-Wild | **10,48%** | 11,96% | 10,46% e 11,2% (terceiros) |
| BRSpeech-DF (teste) | **39,24%** | **35,38%** | 35,38% (artigo, que usou zeros) |

**Principais achados:**

1. **Reprodução:** os números publicados foram reproduzidos nos três cenários. No BRSpeech-DF, a
   reprodução é exata quando se usa o mesmo preenchimento do AKCIT.
2. **Perda de generalização:** a EER passa de 2,86% para 10,48% e depois para 39,24% à medida
   que o cenário se afasta do treino (domínio, condições de gravação, idioma).
3. **A perda vem dos áudios genuínos:** fora do domínio, a pontuação dos falsos quase não muda,
   mas a dos genuínos cai muito (de +2,17 no 2021 DF para −0,53 no In-the-Wild e −2,15 no
   BRSpeech-DF). O detector passa a desconfiar de fala real gravada em condições desconhecidas.
4. **O pré-processamento importa:** o preenchimento de áudios curtos muda a EER em até 3,9
   pontos, em sentidos opostos conforme o dataset. O número publicado do BRSpeech-DF (35,38%)
   usa zeros, e não o pré-processamento original do Tak, que daria 39,24%.
5. **Dificuldade por gerador (BRSpeech-DF):** F5-TTS e XTTS são os mais difíceis de detectar
   (~42–45% de EER), e ToucanTTS o mais fácil (~25–28%).
6. **O BRSpeech-DF é paralelo:** cada falso refaz a mesma frase de um genuíno. Isso é
   relevante para o desenho do teste da hipótese semântica (ver README, seção 5.3).

---

## 10. Pendências e próximos passos

- Revisar o rascunho das "Implicações para a hipótese" (README, seção 5.3).
- Opcional: verificar se a diferença de 0,01 ponto no 2021 DF vem do TF32.
- Próximo experimento: testar a hipótese (acústico + semântico) contra esta linha de base,
  usando o mesmo pré-processamento e os mesmos dados.

---

## 11. Onde está cada coisa

| O quê | Onde |
|---|---|
| Resultados e reprodução | `README.md` |
| Scripts | `scripts/` |
| Ambiente | `docker/` |
| Pontuações, métricas e comparações | `resultados/asvspoof21df/`, `resultados/in_the_wild/`, `resultados/brspeech_test_hf/` |
| Metadados de cada execução (SHA-256, versões, tempos) | `resultados/**/*.meta.json` |

Commits no GitHub (`matheussfm/residencia`):

- `69c0767`: scripts, ambiente e esqueleto do README;
- `6b903c2`: resultados do 2021 DF e do In-the-Wild;
- `9816bf2`: suporte à versão Hugging Face do BRSpeech-DF;
- `cebeff4`: resultados do BRSpeech-DF.

---

## 12. Uso de LLM

O trabalho foi feito com apoio de um assistente de IA (Claude, da Anthropic, via Claude Code).

- **O assistente:**
  - leu e analisou o código dos repositórios originais;
  - propôs o plano;
  - escreveu os scripts, o Dockerfile e a documentação;
  - testou os scripts localmente;
  - indicou os comandos de execução no servidor;
  - analisou os resultados.
- **O pesquisador:**
  - definiu o objetivo, os cenários e as regras;
  - aprovou cada etapa;
  - tomou as decisões metodológicas;
  - executou tudo no servidor;
  - revisou os resultados.

O assistente não teve acesso ao servidor.
