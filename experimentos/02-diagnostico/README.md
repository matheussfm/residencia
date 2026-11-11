# Experimento 02 – Diagnóstico da perda de generalização

Investiga **de onde vem e o que causa** a perda de desempenho do detector do Experimento 01
(Tak et al., 2022: XLS-R 300M + AASIST, checkpoint publicado, sem treino) em áudios
diferentes dos de treino. O experimento é dividido em fases; este documento cobre a
**Fase 1 (decompor o erro existente, sem GPU)**, concluída, e o desenho das fases seguintes.

> Status: Fase 1 concluída (rascunho, a revisar). Fases 2 e 3 em andamento.

---

## 1. Ancoragem

**Área:** detecção de deepfakes de fala (*audio deepfake detection*), linha de generalização.

**Papel deste experimento:** o Experimento 01 mostrou **que** o detector perde desempenho fora
do domínio (EER de 2,86% no ASVspoof 2021 DF, 10,48% no In-the-Wild e 39,24% no BRSpeech-DF),
mas os três cenários diferem em vários fatores ao mesmo tempo. Este experimento tenta isolar
**o que** causa a perda. A hipótese semântica da residência fica em pausa; o diagnóstico deve
indicar em que casos uma evidência semântica poderia ajudar.

## 2. Problema

Quando o detector erra em áudios novos, o que causa o erro? Respostas candidatas, que não se
excluem: (1) condição de gravação ou canal; (2) gerador novo; (3) idioma; (4) duração e
pré-processamento; (5) calibração; (6) onde a informação se perde no modelo (front-end XLS-R
ou back-end AASIST).

## 3. Frameworks de sustentação

- **Protocolo ASVspoof 2021 DF** com a **chave completa** (codec, origem do áudio, ataque, tipo
  de vocoder, *trim*), distribuída no repositório do AKCIT (`data/metadata/trial_metadata.txt`).
  As chaves `DF-keys-stage-1` do Tak trazem só rótulo e fase. O conjunto `eval` e os rótulos
  das duas chaves foram conferidos: são idênticos (PROCESSO.md, seção 2).
- **EER** como métrica principal, com **intervalos de confiança de 95% por bootstrap de
  locutores** (reamostra locutores, não áudios, porque áudios do mesmo locutor não são
  independentes).
- **Cllr e minCllr** (custo da log-razão de verossimilhança, da verificação de locutor): separam
  a perda de ordenação (minCllr, o melhor possível com qualquer transformação monotônica) da
  perda de calibração (Cllr − minCllr). 1 bit corresponde a um detector que não informa nada.
- **AUC** e **taxa de acerto pareada** (BRSpeech-DF: cada falso contra o genuíno com a mesma frase).
- Pontuações do Experimento 01 (repetição, corte de 4 s) e, como controle, as pontuações
  publicadas pelo AKCIT para SSL, SLS e AASIST no 2021 DF e no BRSpeech-DF (todas as divisões).

## 4. Fase 1: resultados

Todos os números são do SSL (XLS-R + AASIST) com as nossas pontuações, preenchimento por
repetição e corte de 4 s, salvo indicação. Os JSON completos estão em `resultados/fase1/`.

### 4.1 Quem se desloca: o genuíno, não o falso

**EER cruzada** (`global.json`, `figuras/matriz_cruzada.png`): genuínos de um cenário contra
falsos de outro. As pontuações vêm do mesmo modelo e do mesmo pré-processamento, então são
comparáveis entre cenários.

| Genuínos de ↓ / falsos de → | 2021 DF | In-the-Wild | BRSpeech-DF |
|---|---|---|---|
| 2021 DF | 2,86% | 4,61% | 6,91% |
| In-the-Wild | 7,92% | 10,48% | 15,06% |
| BRSpeech-DF | 22,50% | 26,15% | 39,24% |

- Os **falsos fora do domínio continuam quase tão detectáveis** quanto os de dentro: contra os
  genuínos do 2021 DF, os falsos do BRSpeech-DF (TTS zero-shot, português) dão 6,91% de EER.
- Trocar só os **genuínos** leva a EER muito mais longe: genuínos do BRSpeech-DF contra os falsos
  do 2021 DF já dão 22,50%.
- Distância de Wasserstein até o 2021 DF, em unidades de pontuação: genuínos 2,70 (In-the-Wild) e
  4,32 (BRSpeech-DF); falsos 0,03 e 0,15.
- Os dois efeitos se somam: a pior célula (39,24%) combina genuínos e falsos fora do domínio.

**Leitura:** a perda vem sobretudo do deslocamento dos **genuínos**, o que reforça a resposta
candidata 1 (condição de gravação). Mas os falsos novos também contribuem (2,86% → 6,91%), então
a resposta 2 (gerador novo) não é nula.

**Forma das pontuações** (`figuras/distribuicoes.png`): a saída é bimodal e quase saturada, com um
pico perto de −3,6 (falso) e outro perto de +4,5 (genuíno). Fora do domínio, os genuínos que
erram não ficam "no meio": caem no pico dos falsos. Isso limita o que uma recalibração pode fazer
(seção 4.3).

### 4.2 O fator silêncio: a hidden track do 2021 DF

A fase `hidden` do 2021 DF (18.576 trials, todos de origem VCTK/ASVspoof) tem o **silêncio do
início e do fim removido** (`trim`). As nossas pontuações já cobriam esses trials.
(`df21_subgrupos.json`)

| | Eval, origem VCTK (com silêncio) | Hidden track (sem silêncio) |
|---|---|---|
| EER | 1,27% | **11,88%** |
| Genuínos rejeitados no limiar da EER do eval | 0,00% | **30,71%** |
| Falsos aceitos no mesmo limiar | 10,09% | 4,80% |
| Pontuação média dos genuínos / falsos | +3,99 / −3,6 | **−2,94** / −3,64 |

- O efeito aparece em **todos os 9 codecs**, sempre na mesma direção: a EER vai para 11–13% e a
  rejeição de genuínos, para 22–40%.
- Cruzada: genuínos sem silêncio contra falsos com silêncio, 10,71%; genuínos com silêncio contra
  falsos sem silêncio, 3,52%. De novo, é o **genuíno** que se desloca.
- **Leitura:** o detector usa o silêncio das pontas como evidência de fala genuína. Sem ele, a
  fala genuína do próprio corpus de treino passa a parecer falsa. Isso é coerente com um atalho
  conhecido do ASVspoof 2019 (ver seção 4.3 do contexto: leitura do grupo do Müller **a
  confirmar** antes de citar).
- **Cuidados:** (a) os trials da hidden track não são os mesmos áudios do eval com o silêncio
  cortado, então a comparação é entre conjuntos, com a mesma origem e o mesmo codec, mas não é
  pareada por arquivo; (b) a proporção de vocoders difere entre as fases (mais tradicionais na
  hidden), mas os falsos quase não mudam, então isso não explica o efeito nos genuínos; (c) a
  Fase 2 testa o fator diretamente, cortando e acrescentando silêncio nos mesmos áudios.

### 4.3 Calibração ou ordenação? As duas, mas sobretudo ordenação

(`calibracao.json`) Calibração afim ajustada no 2021 DF e aplicada aos outros cenários:

| Cenário | EER | Erro balanceado no limiar do 2021 DF | Cllr | minCllr | Recalibração com 200 por classe |
|---|---|---|---|---|---|
| 2021 DF | 2,86% | 2,86% | 0,20 | 0,11 | 3,2% |
| In-the-Wild | 10,48% | 10,31% | 0,63 | 0,37 | 12,8% |
| BRSpeech-DF (teste) | 39,24% | 33,41% | 1,94 | 0,85 | 35,4% |

- No limiar do treino, o erro balanceado já fica perto da EER. Com dados do próprio domínio,
  a recalibração **não melhora**. O limiar não é o gargalo principal.
- A calibração está ruim (Cllr de 1,94 bit no BRSpeech-DF, pior que um detector sem informação),
  mas o minCllr de 0,85 bit mostra que mesmo uma calibração perfeita deixaria pouca informação:
  a **ordenação** em si se perde. Resposta candidata 5: contribui para o Cllr, mas não explica a
  EER.

### 4.4 Locutores: o erro se concentra em poucos

**In-the-Wild** (`itw_locutores.tsv`, `figuras/itw_locutores.png`; reproduz a análise do Termo da
Semana 8): no limiar da EER global, 2.091 genuínos são rejeitados, e **metade vem de 5 dos 54
locutores** (Alec Guinness, Winston Churchill, Martin Luther King, Barack Obama, Dwight Eisenhower).
As maiores taxas, entre locutores com pelo menos 30 genuínos, são de Calvin Coolidge (79%), 2Pac
(73%), Eisenhower (60%), Malcolm X (59%), Martin Luther King (40%), Churchill (36%) e FDR (32%).
*Interpretação, a verificar:* vários são gravações antigas ou de baixa qualidade. Não há
metadado de época nem de canal no In-the-Wild, e isso só pode ser confirmado ouvindo os áudios
ou com a Fase 2.

**BRSpeech-DF, teste** (`brs_pares.json`): **3 dos 10 locutores** (5739, 13063 e 12626) têm genuínos
com pontuação média de cerca de −3,6, o mesmo nível dos falsos. Eles concentram 40% dos genuínos e
**87% dos genuínos rejeitados**. A EER de cada um deles passa de 50%. Nos outros 7, a EER do
próprio locutor vai de 7% a 33%.

Nos dois cenários, a média dos genuínos varia muito mais entre locutores (desvio-padrão de 1,83 no
In-the-Wild e 1,35 no BRSpeech-DF) do que a dos falsos (0,24 e 0,06).

**Leitura:** o erro é "por locutor", ou por condição de gravação de cada locutor. Com estes dados,
não dá para separar a identidade do locutor da condição de gravação, porque cada locutor tem a
sua. A Fase 2 separa as duas coisas.

### 4.5 BRSpeech-DF pareado: a frase e o locutor do par não explicam a pontuação

- Taxa de acerto pareada (genuíno > o seu falso): 67,6% no teste, contra AUC não pareada de 0,670.
  Por TTS, as duas medidas ficam a menos de 3 pontos uma da outra.
- Correlação entre a pontuação de um genuíno e a do seu falso: r = 0,04.
- **Leitura:** o que o par compartilha (frase, locutor, provavelmente o texto) não move a
  pontuação. O falso tende a ir para o pico de −3,6, qualquer que seja o genuíno. O que varia é o
  genuíno, por locutor. Isso também é relevante para a hipótese semântica (pausada): nesse
  dataset, a frase não é o que distingue as classes.

### 4.6 A divisão de teste do BRSpeech-DF é atípica

Com as pontuações do AKCIT para o SSL (preenchimento por zeros) em **todas as divisões** (62
locutores, 76.644 genuínos e 382.493 falsos; nenhum dos detectores foi treinado no BRSpeech-DF):

- EER de todas as divisões: **22,11%**, contra 35,38% no teste oficial;
- com 10 locutores sorteados (1.000 sorteios), a EER mediana é de 19,67% (5–95%: 12,24% a
  36,46%). O teste oficial fica no **percentil 94**;
- o IC de 95% por locutores da EER do teste vai de 18,8% a 51,9% (`global.json`).

**Implicação:** a EER do BRSpeech-DF publicada e reproduzida (35–39%) depende muito de quais
locutores estão no teste. Para o diagnóstico, as divisões de treino e validação do BRSpeech-DF
também servem de teste deste detector. Elas exigem baixar o resto do dataset; por enquanto só
temos as pontuações do AKCIT para elas.

### 4.7 Duração, janela de 4 s e preenchimento

(`duracao.json`, `figuras/duracao.png`; reproduz a análise do Termo da Semana 8)

| Faixa | 2021 DF | In-the-Wild | BRSpeech-DF |
|---|---|---|---|
| < 2 s | 8,72% | 18,06% | 53,79% |
| 2–3 s | 2,51% | 10,48% | 51,56% |
| 3–4,04 s | 1,80% | 8,65% | 46,75% |
| 4,04–6 s | 0,97% | 6,69% | 40,81% |
| 6–10 s | 3,95% | 6,39% | 31,73% |
| > 10 s | (n = 5) | 7,39% | 32,54% |

- Nos três cenários, **áudios curtos são piores**. No BRSpeech-DF, abaixo de ~4 s, a EER fica perto
  do acaso. Os IC por locutores do BRSpeech-DF são largos (por exemplo, 32% a 58% abaixo de 2 s).
- **Não há atalho forte de duração:** a duração sozinha, usada como pontuação, dá 43,3% (2021 DF),
  41,6% (In-the-Wild) e 49,9% (BRSpeech-DF). Estratificar por faixa quase não muda a EER do
  In-the-Wild (10,90%). A diferença de duração entre as classes do In-the-Wild não explica a EER.
- **Preenchimento:** nos áudios curtos, trocar repetição por zeros vai de 12,66% para 14,54% no
  In-the-Wild e de 51,28% para 47,27% no BRSpeech-DF. Os longos não mudam (controle: diferença zero).
- **Corte:** nos áudios longos, ouvir o áudio inteiro piora: 6,69% → 8,28% no In-the-Wild e
  33,17% → 35,60% no BRSpeech-DF.

### 4.8 Subgrupos do 2021 DF: o padrão já aparece dentro do domínio

(`df21_subgrupos.json`, `figuras/df21_subgrupos.png`)

- **Origem dos genuínos:** os genuínos do VCTK, o corpus do treino, têm média de +3,99 e 0% de
  rejeição no limiar global. Os do **VCC 2018** têm média de −0,63 e **9,95%** de rejeição; os do
  VCC 2020, 1,56%. Mesmo dentro do protocolo de avaliação, a fala genuína de outro corpus parece
  mais falsa.
- **Codec:** o pior é o `low_mp3` (EER de 4,30%, 7,72% de genuínos rejeitados, média dos genuínos
  de +0,24); sem codec, 2,34%.
- **Vocoder:** o autorregressivo neural é o mais difícil (4,04%, contra 1,84% a 2,85% dos demais).
  Algumas equipes do VCC 2020 chegam a 6–7% de EER e 20% de falsos aceitos.

### 4.9 Controle com outros detectores (a verificar)

As pontuações do AKCIT para SLS e AASIST dão, respectivamente, 1,92% e 21,06% no 2021 DF, e 52,88% e
48,12% no teste do BRSpeech-DF. No BRSpeech-DF, o minCllr do SLS é de 0,999 bit (sem informação), e
todos os seus áudios ficam acima do limiar do 2021 DF. **Não usamos esses dois detectores como
evidência:** antes, é preciso conferir se essas pontuações foram geradas corretamente. Por
exemplo, o áudio de 24 kHz pode ter entrado sem reamostragem, e convém comparar com os números
do artigo do BRSpeech-DF. O segundo detector da Fase 4 continua em aberto.

### 4.10 Síntese provisória (fator × evidência)

| Resposta candidata | Evidência da Fase 1 | Situação |
|---|---|---|
| 1. Condição de gravação/canal | Perda vem dos genuínos; concentrada por locutor; VCC 2018 e `low_mp3` já deslocam no 2021 DF | **Mais apoiada**; falta o teste causal (Fase 2) |
| Silêncio (parte de 1 e 4) | Hidden track: EER de 1,27% → 11,88% na mesma origem, com rejeição de 0% → 31% | **Forte**; falta o teste causal pareado (Fase 2) |
| 2. Gerador novo | Falsos do BRSpeech-DF dão 6,91% contra genuínos do 2021 DF; F5-TTS e XTTS piores | Contribui, secundário |
| 3. Idioma | Não isolado. Em português, 7 dos 10 locutores do teste têm EER própria de 7% a 33%, enquanto 3 ficam acima de 50%: o idioma sozinho não explica essa diferença entre locutores | Em aberto (o mais difícil) |
| 4. Duração e pré-processamento | Curtos piores; sem atalho de duração; preenchimento muda até ~4 pontos nos curtos | Contribui, secundário |
| 5. Calibração | Cllr ruim, mas o limiar do treino e a recalibração não mudam a EER; minCllr alto | Não é o gargalo principal |
| 6. Front-end × back-end | Não testado | Fase 3 |

## 5. Validação

- Os carregadores do Experimento 02 reproduzem **exatamente** todas as EER do Experimento 01 (2,86%,
  10,48%, 11,96%, 12,18%, 39,24%, 35,38% e 41,90%), e as do AKCIT (2,85% no 2021 DF e 35,38% no
  BRSpeech-DF).
- A chave completa do 2021 DF foi conferida contra a do Tak: os ids, a fase `eval` e os rótulos
  são idênticos.
- Os 6.587 falsos do teste do BRSpeech-DF têm o genuíno correspondente pelo mapa validado no
  Experimento 01.
- Todo JSON tem um `.meta.json` com o SHA-256 das entradas, as versões e os parâmetros.
- Limites: vários subgrupos são pequenos (veja os IC). No BRSpeech-DF, os IC por locutores se
  apoiam em só 10 grupos. As análises são observacionais: indicam fatores, mas não os isolam
  (é o papel da Fase 2).

## 6. Próximas fases (desenho)

### Fase 2: perturbações controladas (GPU, servidor)

Aplicar a genuínos do domínio de treino um fator por vez e medir o deslocamento da pontuação do
**mesmo áudio** (análise pareada). O pré-processamento do Tak continua fixo.

- **Amostra:** todos os 386 genuínos do 2021 DF eval de origem VCTK sem codec (`asvspoof`, `nocodec`),
  e 400 falsos sorteados com semente fixa, para medir os dois lados. Os dois
  conjuntos vêm da chave completa. São 29 condições × 786 áudios (~23 mil), poucos minutos de GPU.
- **Fatores e níveis**, nesta ordem de prioridade:
  1. **Silêncio:** cortar as pontas (como na hidden track); acrescentar 0,25/0,5/1/2 s de
     silêncio digital ou de ruído de fundo baixo no início. A janela de 4 s faz o silêncio
     inicial pesar mais;
  2. **Banda:** passa-faixa de telefone (300–3.400 Hz) e passa-baixas de 4/6 kHz (gravação
     antiga);
  3. **Compressão:** MP3 a 16/32/64 kbps e Opus (precisa do ffmpeg na imagem; ver PROCESSO);
  4. **Ruído:** branco e rosa com SNR de 30/20/10 dB;
  5. **Reverberação:** resposta ao impulso sintética com RT60 de 0,3/0,6/1,0 s;
  6. **Volume e reamostragem:** ganho de −20/−10/+6 dB (com limitação); reamostragem para 8 kHz e
     de volta.
- **Medidas:** deslocamento mediano da pontuação por condição, fração de genuínos que caem abaixo
  do limiar, EER contra os falsos sem perturbação e EER com os dois lados perturbados.
- **Scripts prontos:** `scripts/f2_perturbar.py` gera os áudios perturbados (29 condições,
  incluindo a original) e a `lista.tsv` para o `inferir.py` do Experimento 01.
  `scripts/f2_analisar.py` faz a análise pareada. Os dois foram testados de ponta a ponta no PC,
  com áudio sintético e pontuações simuladas. Ainda não rodaram com dados reais.

### Fase 3: dentro do modelo (GPU, servidor)

Extrair as representações por camada do XLS-R (as 24 camadas do transformer e a saída final,
média no tempo) e o embedding final do AASIST, com *hooks* do PyTorch, sem alterar o modelo. A
amostra tem 10.640 áudios, balanceados por classe:
- 2021 DF eval: 500 por classe de cada origem (VCTK, VCC 2018 e VCC 2020);
- hidden track: 1.000 por classe;
- In-the-Wild: 1.500 por classe;
- BRSpeech-DF: os 1.320 genuínos do teste e 1.320 falsos, 264 por TTS.

Scripts: `f3_selecionar.py`, `f3_extrair.py` e `f3_probing.py`, testados no PC com um modelo de
mentira de mesma estrutura e com features sintéticas.
- *Probing* linear por camada: treinar no 2021 DF e testar nos outros cenários; depois, ajustar
  só o classificador linear com k exemplos do alvo. Se a separação reaparecer, o gargalo está no
  back-end ou na calibração. Se não, está no front-end.
- Visualizar com UMAP ou t-SNE por cenário e por rótulo, e por locutor nos 3 locutores "ruins" do
  BRSpeech-DF contra os outros 7.
- Incluir os trials da hidden track: o silêncio se perde no front-end ou no back-end?

## Reprodutibilidade

Fase 1, sem GPU e sem áudio: só as pontuações do Experimento 01 e os submódulos. Foi rodada
localmente com Python 3.13.12, numpy 2.4.4, scipy 1.17.1 e scikit-learn 1.8.0. Os scripts são
compatíveis com Python 3.8, a versão da imagem do Experimento 01. A partir da raiz do
repositório, com os submódulos inicializados (`git submodule update --init`):

```bash
cd experimentos/02-diagnostico/scripts
python -B f1_global.py          # 4.1      (~2 min)
python -B f1_df21_subgrupos.py  # 4.2, 4.8 (~10 min)
python -B f1_calibracao.py      # 4.3, 4.9 (~10 min)
python -B f1_itw_locutores.py   # 4.4
python -B f1_brs_pares.py       # 4.5, 4.6
python -B f1_duracao.py         # 4.7
bash ../../01-linha-de-base/scripts/verificar_submodulos.sh
```

A semente é fixa (0) em todos os bootstraps e sorteios. Os parâmetros (B, k e faixas) estão
no `.meta.json` de cada resultado.

## Referências

1. H. Tak et al. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0
   and data augmentation*. Odyssey 2022.
2. A. C. Ferro Filho et al. *BRSpeech-DF: A Deep Fake Synthetic Speech Dataset for Portuguese
   Zero-Shot TTS*. EMNLP 2025. Repositório com a chave completa do 2021 DF e as pontuações de SSL,
   SLS e AASIST.
3. N. Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022 (In-the-Wild).
4. N. Brümmer, J. du Preez. *Application-independent evaluation of speaker detection*. Computer
   Speech & Language, 2006 (Cllr). _(referência de memória: conferir os dados antes de citar)_
5. ASVspoof 2021: plano de avaliação e chaves completas do DF (hidden track). _(conferir a citação
   exata)_
