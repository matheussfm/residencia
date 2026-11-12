# Experimento 02 – Diagnóstico da perda de generalização

Investiga **de onde vem e o que causa** a perda de desempenho do detector do Experimento 01
(Tak et al., 2022: XLS-R 300M + AASIST, checkpoint publicado, sem treino) em áudios
diferentes dos de treino. O experimento é dividido em fases: a **Fase 1** decompõe o erro existente
(observacional, sem GPU), a **Fase 2** testa causas com perturbações controladas e a **Fase 3**
procura em que parte do modelo a informação se perde.

> Status: Fases 1, 2 (com 2b e 2c) e 3 executadas (Semana 9).

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
  fala genuína do próprio corpus de treino passa a parecer falsa. Müller et al. (2021) [6] já
  tinham mostrado que, no ASVspoof 2019 e 2021, a duração do silêncio se correlaciona com o rótulo e
  que modelos treinados nesses dados se apoiam nesse atalho. O resultado aqui mostra que o atalho
  continua presente num detector mais recente, com front-end auto-supervisionado (XLS-R + AASIST).
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

### 4.10 Medidas acústicas dos áudios (sem modelo)

(`f1_qualidade_audio.py` no servidor, sobre os 10.640 áudios da amostra da Fase 3;
`f1_qualidade_analise.py`; `resultados/qualidade/`.) Para cada áudio: silêncio no início e no fim
(trecho abaixo de máximo − 40 dB, o mesmo critério do corte da Fase 2), SNR estimada, nível,
banda que contém 99% da energia, fração de energia abaixo de 300 Hz e duração.

| Cenário | Silêncio inicial, genuínos (mediana) | Silêncio inicial, falsos | Silêncio inicial sozinho como pontuação |
|---|---|---|---|
| 2021 DF eval | 0,24 s | 0,11 s | EER de 35,9% (AUC 0,71): o atalho existe no protocolo |
| In-the-Wild | **0 s** | **0,23 s** | **AUC 0,34: invertido** |
| BRSpeech-DF | 0,27 s | 0 s | EER de 33,8% (AUC 0,72) |

- **No In-the-Wild, o atalho do silêncio aponta para o lado errado.** A fala real quase não tem
  silêncio inicial, e a sintética tem. A regra aprendida no ASVspoof ("silêncio no início =
  genuíno", seção 5.2) faz a voz real parecer falsa e a falsa parecer real nesse cenário. É uma
  explicação concreta, medida nos próprios dados, para parte da perda no In-the-Wild.
- **Dentro dos genuínos, a pontuação acompanha o silêncio inicial:** correlação de Spearman de
  +0,60 no 2021 DF eval, +0,26 no In-the-Wild e +0,37 no BRSpeech-DF.
- **Os 3 locutores "ruins" do BRSpeech-DF não se explicam por nenhuma medida isolada.** Silêncio
  inicial, SNR, nível, banda e graves ficam na mesma faixa dos locutores "bons" (por exemplo, o
  locutor 13063, ruim, tem 0,42 s de silêncio inicial). A duração explica parte do locutor 5739
  (68% dos áudios abaixo de 4 s), mas os áudios **longos** dos três também recebem −3,6. Restam a
  reverberação e as características da voz, que estas medidas não captam; a escuta é a checagem
  seguinte.
- Dentro dos locutores "bons", o efeito da duração aparece claramente (locutor 3050: pontuação
  mediana de −0,32 nos áudios longos e −3,28 nos curtos), coerente com a seção 4.7.

## 5. Fase 2: perturbações controladas (teste causal)

### 5.1 Desenho

A Fase 1 é observacional: os cenários diferem em muitos fatores ao mesmo tempo. Aqui cada fator
é aplicado **sozinho** a áudios do domínio de treino, e cada áudio é comparado **com ele mesmo**
antes da perturbação. Locutor, frase, idioma e gravação ficam fixos; só o fator muda.

- **Amostra:** todos os 386 genuínos do 2021 DF eval de origem VCTK sem codec e 400 falsos
  sorteados (semente 0), da chave completa. É o caso mais "dentro do domínio" (Fase 1: EER de
  1,27% e 0% de rejeição de genuínos na origem VCTK).
- **29 condições** (`f2_perturbar.py`): original; silêncio (cortar as pontas a 40 e 60 dB abaixo
  do máximo; acrescentar 0,25 a 2 s de silêncio digital ou de ruído baixo no início); banda
  (telefone 300–3.400 Hz, passa-baixas de 4 e 6 kHz, RMS restaurado); codec (MP3 a 16/32/64
  kbps, Opus a 16 kbps); ruído branco e rosa (SNR de 30/20/10 dB); reverberação (resposta ao
  impulso sintética, RT60 de 0,3/0,6/1,0 s); volume (−20/−10/+6 dB); reamostragem para 8 kHz.
- **Inferência:** `inferir.py` do Experimento 01, sem mudanças (repetição, corte de 4 s); 22.794
  áudios em 1,7 min na RTX 4090.
- **Análise** (`f2_analisar.py`): deslocamento mediano da pontuação de cada áudio (IC de 95% por
  bootstrap), EER com só os genuínos perturbados (contra os falsos originais) e EER com as duas
  classes perturbadas.

### 5.2 Resultados

(`resultados/fase2/perturbacoes.json`, `figuras/perturbacoes.png`)

| Fator | Condição | Δ genuínos (mediana, IC 95%) | EER só genuínos perturbados | EER ambos perturbados | Genuínos rejeitados* |
|---|---|---|---|---|---|
| Controle | original | 0,00 | 0,3% | 0,3% | 0,3% |
| **Reverberação** | RT60 0,3 s | **−7,71** [−7,78; −7,61] | **11,2%** | **15,3%** | **82,6%** |
| | RT60 1,0 s | **−7,80** [−7,84; −7,77] | **11,2%** | **24,0%** | **96,1%** |
| **Silêncio** | cortar as pontas (40 dB) | −0,87 [−1,74; −0,58] | 5,7% | 5,7% | 33,2% |
| | +2 s de silêncio no início | +0,05 | 0,0% | 9,5% | 0,0% (falsos aceitos: **22,2%**) |
| Codec | MP3 16 kbps | −2,95 [−3,29; −2,02] | 5,7% | 3,1% | 36,3% |
| | MP3 32/64 kbps, Opus 16 kbps | ≈ 0 | 1,0–2,5% | 1,0–2,3% | ≈ 0% |
| Banda | telefone (300–3.400 Hz) | −1,18 [−2,00; −0,70] | 5,5% | 3,6% | 25,9% |
| | passa-baixas 4 e 6 kHz; reamostrar 8 kHz | ≈ 0 | 0,3–0,5% | 0,0–0,3% | ≈ 0% |
| Volume | −20 dB | −2,41 [−2,86; −2,00] | 5,7% | 6,5% | 31,6% |
| | −10 e +6 dB | ≈ 0 | 0,5–1,0% | 0,3–1,5% | ≈ 0% |
| Ruído | branco e rosa, SNR 10–30 dB | ≈ 0 | 0,0–0,8% | 0,0–0,5% | ≤ 1% |

\* No limiar da EER do conjunto original (0,086), isto é, o limiar em que o detector separa os
áudios limpos com 0,3% de erro. No limiar fixo do 2021 DF (−3,53), que fica colado no pico dos
falsos, as taxas são menores para os genuínos (reverberação: 6–8%) e muito instáveis para os
falsos; por isso a EER é a medida principal.

**Leituras:**

1. **A reverberação é, de longe, o fator mais forte.** Ela empurra a pontuação mediana dos
   genuínos do pico "verdadeiro" (+4,5) para o pico "falso" (−3,3), quase sem variação entre os
   áudios (IC estreito). Com eco nos dois lados, a EER chega a 24%: o detector deixa de separar
   as classes. É o mesmo padrão dos genuínos fora do domínio na Fase 1 (In-the-Wild, os 3
   locutores "ruins" do BRSpeech-DF), agora com uma causa demonstrada.
2. **O silêncio funciona como atalho nos dois sentidos.** Cortar o silêncio das pontas faz a fala
   real parecer falsa (EER de 0,3% → 5,7%); acrescentar silêncio no início faz a fala sintética
   parecer real (com 2 s, 22% dos falsos passam como genuínos e o percentil 90 do deslocamento
   dos falsos é +7,5, ou seja, 10% deles saltam para o pico "verdadeiro"). É o teste causal do
   que a hidden track sugeria (seção 4.2) e do que Müller et al. (2021) [6] descreveram.
3. **Compressão forte, banda de telefone e volume muito baixo** têm efeito médio (EER de 5–6%).
   Só a compressão mais agressiva pesa: MP3 a 32 e 64 kbps e Opus a 16 kbps quase não mudam nada.
   Na banda, o que pesa é o corte dos **graves** (abaixo de 300 Hz): cortar os agudos acima de 4
   ou 6 kHz não muda nada.
4. **Ruído aditivo não afeta o detector**, mesmo com SNR de 10 dB. O script de treino do DF do Tak
   (`main_SSL_DF.py`) usa por padrão o RawBoost `algo 3` (SSI): ruído aditivo estacionário e
   colorido com **SNR entre 10 e 40 dB**, exatamente a faixa em que a Fase 2 não vê efeito.
   Nenhum algoritmo do RawBoost simula reverberação (o "convolutivo" do algoritmo 1 é um filtro
   *notch* com não-linearidade, não a resposta de uma sala). Leitura: o detector é robusto ao que
   viu no aumento de dados e frágil ao que não viu. *Ressalva:* o README do Tak não diz qual
   `algo` gerou o checkpoint de DF; o padrão do script é indício forte, a confirmar no artigo [1]; o RawBoost está em [7].

### 5.3 Controles e limites

- **Reprodução da condição original.** As pontuações da condição "original" reproduzem as do
  Experimento 01 em 99,1% das decisões (correlação de 0,993), mas não arquivo a arquivo: 192 dos
  786 áudios diferem mais de 0,01 (um genuíno, 6,0). A causa provável é a decodificação: a
  libsndfile recente recusou 339 dos `.flac`, que foram lidos pelo ffmpeg (PROCESSO, problema 8).
  Como cada áudio é comparado com a sua própria versão original, a comparação pareada não é
  afetada. **Teste de robustez** (`robustez_consistentes.json`): refeita só com os 594 áudios cuja
  pontuação original difere menos de 0,01 do Exp. 01, a tabela mantém a mesma ordem de fatores
  (reverberação 9–10% e 16–25%; silêncio, MP3 16 kbps, telefone e volume em 3–5%).
- A amostra vem de um único domínio limpo (VCTK); os efeitos medem a sensibilidade do detector
  partindo do melhor caso.
- A reverberação é sintética (ruído com decaimento exponencial), não uma sala real; é uma
  aproximação do efeito de sala.
- O corte de silêncio aqui (40 dB) dá efeito menor que a hidden track (5,7% contra 11,88%). O
  método de corte do ASVspoof é desconhecido, e os conjuntos de áudio são outros.

### 5.4 Fase 2b: normalizar o silêncio em dados reais fora do domínio

As medidas acústicas (4.10) mostraram que, no In-the-Wild, o atalho do silêncio aponta para o lado
errado. Para medir quanto da perda real vem dele, o silêncio foi normalizado em **todos** os áudios
do In-the-Wild e do BRSpeech-DF, nas duas classes (`f2_silencio_real.py`), e os áudios foram
pontuados de novo, com o mesmo `inferir.py`:
- **aparar:** corta o silêncio do início e do fim (máximo − 40 dB);
- **padronizar:** corta e acrescenta exatamente 0,5 s de silêncio digital no início de todos.

(`f2_silencio_analise.py`, `resultados/silencio/silencio_real.json`; diferença de EER pareada com
IC de 95% por bootstrap de locutores, sorteando os mesmos locutores nas duas condições.)

| Cenário | Original | aparar | padronizar |
|---|---|---|---|
| **In-the-Wild** | 10,48% [7,77; 14,00] | 12,44% (Δ +1,97 pp [+1,21; +3,24]) | **7,64%** (Δ **−2,84 pp [−4,96; −1,05]**) |
| BRSpeech-DF | 39,24% [18,77; 51,85] | 40,21% (Δ +0,97 pp [−1,18; +4,84]) | 39,55% (Δ +0,30 pp [−0,72; +4,57]) |
| BRSpeech-DF, 3 locutores "ruins" | 58,90% | 55,88% | 64,71% |

- **No In-the-Wild, padronizar o silêncio reduz a EER em 2,8 pontos (27% relativos), sem nenhum
  treino**, e a diferença é significativa. A média dos genuínos sobe de −0,53 para +0,46: a fala
  real, que não tinha silêncio inicial, deixa de ser penalizada. Cortar o silêncio de todos piora
  (+2,0 pontos), porque reforça a pista "sem silêncio = falso".
- **No BRSpeech-DF, nada muda:** lá os genuínos já tinham mais silêncio que os falsos (4.10), e a
  perda vem de outro fator (canal, reverberação ou voz), concentrado nos 3 locutores "ruins".
- **Consequência para a aplicação:** padronizar o silêncio antes de pontuar também neutraliza o
  ataque de acrescentar silêncio a um áudio falso (Fase 2: com 2 s, 22% dos falsos passam), porque
  o corte remove esse silêncio antes da padronização.
- Limite: a diferença pareada é medida com os locutores do próprio dataset; um efeito de 2,8
  pontos em 10,5% não elimina a perda do In-the-Wild, que segue em 7,6%.

### 5.5 Fase 2c: outras intervenções no áudio de teste

Mesma lógica da Fase 2b, com outras transformações aplicadas a todos os áudios do In-the-Wild e do
BRSpeech-DF (`f2_normalizar_real.py`), comparadas com uma condição de **controle** que passa pelo
mesmo caminho de leitura, reamostragem para 16 kHz e gravação (`f2_silencio_analise.py --base
controle`; `resultados/normalizar/normalizar_real.json`). O controle reproduz o Exp. 01 no In-the-Wild
(10,46% contra 10,48%) e no BRSpeech-DF (39,17% contra 39,24%).

| Intervenção | In-the-Wild: Δ EER pareada (IC 95%) | BRSpeech-DF: Δ EER pareada (IC 95%) |
|---|---|---|
| **Volume** (RMS a −18 dBFS, a mediana dos genuínos do 2021 DF) | **−4,37 pp [−6,40; −3,00]**: 10,46% → **6,09%** | −1,06 pp [−3,54; +1,41] (n.s.) |
| Equalização espectral (espectro médio dos genuínos VCTK) | +1,71 pp [+0,39; +3,26] (piora) | +3,03 pp [+0,32; +8,38] (piora) |
| Janela do meio (64.600 amostras centrais) | −0,12 pp [−0,73; +0,52] (n.s.) | +0,68 pp [+0,08; +3,00] |
| Janela do fim | −0,17 pp [−0,84; +0,62] (n.s.) | +2,12 pp [+0,95; +5,67] (piora) |

- **O volume é o fator individual mais forte no In-the-Wild:** só normalizar o nível reduz a EER em
  42% relativos, sem treino. Os genuínos do In-the-Wild estão baixos (mediana de −25,9 dBFS, contra
  −18 dBFS no 2021 DF; seção 4.10), e a Fase 2 mostrou que fala baixa empurra a voz real para o
  lado "falso".
- **Igualar a "cor" do espectro piora** nos dois cenários. Ou o espectro médio não é o que
  diferencia os domínios, ou a equalização simples introduz artefatos; os dois não se separam aqui.
- **O detector não depende especificamente do começo do áudio** além do silêncio: mudar a janela
  não ajuda, e no BRSpeech-DF a janela inicial é a melhor.
- **No BRSpeech-DF, nenhuma intervenção no áudio ajuda**, e os 3 locutores "ruins" continuam perto
  de 60% em todas. A perda em português não se resolve no pré-processamento.

### 5.6 Volume e silêncio juntos, e o efeito no domínio de treino

As duas intervenções que ajudaram no In-the-Wild foram combinadas (`volume_padronizar`: corta o
silêncio das pontas, leva o RMS da fala a −18 dBFS e acrescenta 0,5 s de silêncio no início) e todas
foram aplicadas também a uma amostra do **2021 DF** (os 3.000 trials do eval e os 2.000 da hidden track
da amostra da Fase 3), para verificar que não pioram o domínio de treino. Base: a condição controle.
(`resultados/normalizar/combinado.json`)

| Cenário | Controle | Volume | Silêncio padronizado | **Volume + silêncio** |
|---|---|---|---|---|
| **In-the-Wild** | 10,46% | 6,09% (−4,37 pp) | 7,64% (−2,82 pp) | **4,56%** (Δ **−5,90 pp [−8,76; −3,76]**) |
| **2021 DF eval** | 3,00% | 3,27% (+0,27 [−0,35; +0,46]) | 3,60% (+0,60 [−0,30; +1,58]) | 3,80% (Δ **+0,80 pp [−0,23; +1,80]**, n.s.) |
| **Hidden track** | 11,10% | 11,10% (0,00) | **7,30%** (−3,80 [−5,70; −1,81]) | 7,60% (−3,50 [−5,40; −2,01]) |
| BRSpeech-DF | 39,17% | 38,11% (n.s.) | 41,29% (+2,12 [+0,13; +5,99]) | 40,93% (+1,76 [−0,97; +5,28], n.s.) |

- **Sem nenhum treino, o In-the-Wild cai de 10,5% para 4,6%** (56% relativos). Os dois efeitos se
  somam em parte (−4,4 e −2,8 sozinhos; −5,9 juntos).
- **O domínio de treino não piora de forma significativa:** no 2021 DF eval, a diferença pareada tem
  IC que inclui o zero; a estimativa pontual sobe de 3,0% para 3,8%, o que indica um possível custo
  pequeno, a medir com mais trials.
- **A hidden track funciona como controle positivo da intervenção:** devolver o silêncio padronizado
  aos áudios em que ele foi cortado reduz a EER de 11,1% para 7,3%, mais uma confirmação causal do
  atalho do silêncio.
- **No BRSpeech-DF, nada adianta**, e o silêncio padronizado piora um pouco os locutores "ruins"
  (60,6% → 69,0%). A perda em português não se resolve no pré-processamento.
- No 2021 DF eval, o controle dá 3,00%, contra 3,73% do Exp. 01 nos mesmos áudios: 2.297 dos 5.000
  arquivos foram decodificados pelo ffmpeg e a reamostragem é outra (PROCESSO, problema 14). Como
  todas as comparações são feitas contra o controle, que passa pelo mesmo caminho, elas continuam
  válidas.

## 6. Fase 3: onde a informação se perde (probing por camada)

### 6.1 Desenho

O detector tem duas partes: o **XLS-R** (front-end, 24 camadas), que transforma o som numa
representação, e o **AASIST** (back-end), que decide. A pergunta é: quando o detector erra fora do
domínio, a informação para separar real de falso já não existe nas representações, ou existe e
não é usada?

- **Amostra** (`f3_selecionar.py`, semente 0): 10.640 áudios balanceados por classe: 2021 DF eval
  (500 por classe de cada origem: VCTK, VCC 2018 e VCC 2020), hidden track (1.000 por classe),
  In-the-Wild (1.500 por classe) e BRSpeech-DF teste (os 1.320 genuínos e 1.320 falsos, 264 por TTS).
- **Extração** (`f3_extrair.py`): média no tempo da saída de cada uma das 24 camadas do XLS-R, a
  saída final (entrada do back-end) e o embedding final do AASIST (160 dimensões), capturados por
  *hooks* do PyTorch, com o mesmo pré-processamento do Experimento 01 e o modelo intacto. As
  pontuações extraídas reproduzem as do Experimento 01 (correlação de 0,9999999; diferença máxima
  de 0,0003).
- **Probe linear** (`f3_probing.py`): padronização + regressão logística (C = 0,1), avaliado de
  duas formas:
  - **transferência:** treinado no 2021 DF eval (metade dos locutores) e testado nos outros
    cenários. Imita o detector: aprende no domínio de treino;
  - **no próprio domínio:** validação cruzada em 5 partes **por locutor** (os locutores de teste
    nunca aparecem no treino). Mede se a informação existe naquela representação, naquele domínio.
- **Poucos exemplos** (`f3_poucos_exemplos.py`): probe treinado com k exemplos por classe de
  metade dos locutores do cenário-alvo e testado na outra metade; 20 sorteios por k.

### 6.2 Resultados

(`resultados/fase3/probing.json`, `poucos_exemplos.json`, `figuras/probing.png`)

| Cenário | Modelo completo | Probe treinado no 2021 DF (melhor camada) | Probe no próprio domínio, CV por locutor (melhor camada) | Embedding do AASIST, no próprio domínio |
|---|---|---|---|---|
| 2021 DF eval (locutores não vistos) | 3,5% | 0,6% (camada 11) | 0,8% (camadas 10–11) | 2,9% |
| Hidden track (sem silêncio) | 12,3% | **5,3%** (camada 7) | 3,2% (camada 7) | 7,4% |
| In-the-Wild | 9,8% | **6,3%** (camada 23) | **1,3%** (camadas 3–6) | 6,7% |
| BRSpeech-DF | 39,5% | 26,8% (camada 3) | **1,2%** (camadas 3–5) | 29,5% |

Poucos exemplos do domínio-alvo (EER mediana em locutores não vistos; entre colchetes, p10–p90):

| Cenário | Camada | k = 10 por classe | k = 50 | k = 200 |
|---|---|---|---|---|
| BRSpeech-DF | 5 | 24,0% [19,8–32,4] | **8,7%** [4,6–12,2] | **2,1%** [1,0–7,8] |
| BRSpeech-DF | 24 | 29,6% | 24,6% | 14,7% |
| In-the-Wild | 5 | 7,8% | 3,0% | 1,7% |
| In-the-Wild | 24 | 6,4% | 5,2% | 4,7% |

**Leituras:**

1. **A informação existe no front-end.** No próprio domínio, um probe linear separa o BRSpeech-DF
   com 1,2% de EER e o In-the-Wild com 1,3%, em locutores nunca vistos, usando camadas iniciais
   do XLS-R (3–6). O XLS-R "ouve" a diferença entre fala real e sintética em português.
2. **O que se perde é o que foi aprendido no 2021 DF.** Treinado no domínio de treino, o mesmo
   probe fica em 27–37% no BRSpeech-DF em todas as camadas. A regra de decisão aprendida no 2021
   DF não usa as direções que funcionam em português. Resposta candidata 6: **o gargalo está no
   treino (back-end e camadas finais ajustadas no 2021 DF), não na capacidade do front-end.**
3. **A informação útil para o BRSpeech-DF se perde ao longo da rede:** no próprio domínio, 1,2% nas
   camadas 3–5, 8,8% na saída final do XLS-R e 29,5% no embedding do AASIST. As camadas finais e o
   back-end foram ajustados no 2021 DF e se especializam nele.
4. **Um probe linear simples generaliza melhor que o detector completo** no In-the-Wild (6,3% contra
   9,8%, camada 23) e na hidden track (5,3% contra 12,3%, camada 7), sem nenhum dado do alvo.
   *Interpretação, a verificar:* a média no tempo dilui a posição do silêncio, o que reduziria o
   atalho que o AASIST explora.
5. **Poucos exemplos do domínio-alvo recuperam boa parte da separação:** com 50 exemplos de cada
   classe de 5 locutores, a EER no BRSpeech-DF cai de 39,5% (modelo completo) para 8,7% em locutores
   novos; com 200, para 2,1%. As camadas iniciais adaptam muito melhor que as finais.

### 6.3 Controles complementares (`f3_complementos.py`, `complementos.json`)

**TTS deixado de fora.** Treinando o probe no próprio domínio sem um dos 5 TTS (e com locutores
disjuntos) e testando justamente nele:

| Camada | XTTS | YourTTS | ToucanTTS | F5-TTS | Fish-Speech | Referência (todos os TTS no treino) |
|---|---|---|---|---|---|---|
| 3 | 1,4% | 3,0% | 7,9% | **17,0%** | **20,1%** | 1,2% |
| 5 | 2,7% | 1,1% | 6,1% | **18,6%** | **25,4%** | 1,2% |
| 24 | 10,2% | 14,8% | 20,5% | 41,4% | 32,2% | 9,2% |

O 1,2% depende em parte de o probe ter visto o gerador. Para XTTS e YourTTS, a separação
transfere quase intacta; para os TTS mais recentes (F5-TTS e Fish-Speech), nunca vistos, o erro
sobe para 17–25%. **O gerador novo (resposta 2) é secundário no detector atual, mas é a fronteira
que sobra quando o problema de canal e de atalhos é resolvido.**

**Locutores "ruins" × "bons" do BRSpeech-DF.** O detector completo erra 59,5% nos 3 locutores
"ruins" e 17,9% nos 7 "bons". Um probe na camada 3 treinado **só nos locutores bons** separa os
ruins com **2,3%** (camada 5: 6,8%). A fala real desses locutores é distinguível da sintética nas
representações; o que falha é a regra de decisão do detector.

### 6.4 Em que camada o domínio aparece, e para que lado ele empurra (`f3_dominio.py`)

Para cada camada: (1) a AUC de um classificador linear que distingue os genuínos de um cenário dos
genuínos do 2021 DF (validação cruzada por locutor); (2) a projeção do deslocamento médio dos
genuínos sobre a direção "real × falso" aprendida no 2021 DF, como fração da distância entre as
médias de genuínos e falsos do 2021 DF (0 = não empurra; 1 = leva os genuínos até o lugar dos
falsos). (`resultados/fase3/dominio.json`, `figuras/dominio.png`)

| Representação | Hidden track | In-the-Wild | BRSpeech-DF |
|---|---|---|---|
| Camada 1 | 0,81 | **0,83** | 0,42 |
| Camada 6 | 0,30 | **0,02** | 0,23 |
| Camada 12 | 0,47 | 0,08 | 0,32 |
| Camada 24 | 0,55 | 0,27 | 0,38 |
| Embedding do AASIST | 0,41 | 0,17 | **0,50** |

- **Os domínios são totalmente distinguíveis** em todas as camadas do XLS-R (AUC ≈ 1,00) e quase
  totalmente no embedding do AASIST (0,91–0,94): cada corpus deixa uma assinatura fácil de
  reconhecer.
- **O deslocamento de domínio dos genuínos aponta sempre para o lado "falso"** da regra aprendida no
  2021 DF, em todas as camadas e nos três cenários. É o mecanismo do erro visto por dentro do
  modelo.
- O tamanho do empurrão acompanha o probing (6.2): no In-the-Wild, a camada 1 empurra 83% do
  caminho, e é onde a transferência falha (48% de EER); nas camadas 6–13 o empurrão some, e é onde
  ela funciona. No BRSpeech-DF o empurrão fica entre 22% e 50% em todas as representações, e a
  transferência falha em todas.

### 6.5 XLS-R original × XLS-R ajustado (`f3_extrair_original.py`, `probing_xlsr_original.json`)

O mesmo probing, com as representações do XLS-R 300M **original** (pré-treinado, antes do ajuste no
ASVspoof; entrada normalizada, como no pré-treino, `task.normalize = True` no checkpoint):

| | XLS-R original | XLS-R ajustado (Tak) |
|---|---|---|
| BRSpeech-DF, no próprio domínio (melhor camada) | **0,2%** (camada 4) | 1,2% (camadas 3–5) |
| In-the-Wild, no próprio domínio | 0,9–1,8% | 1,3–3,2% |
| BRSpeech-DF, treinado no 2021 DF (melhor camada) | 24,8% (camada 3); 34–36% nas camadas finais | 26,8% (camada 3); 34–37% nas finais |
| In-the-Wild, treinado no 2021 DF | 6,5% (camada 6) | 6,3% (camada 23) |
| Hidden track, treinado no 2021 DF | 4,7% (camadas 7 e 19) | 5,3% (camada 7) |
| 2021 DF eval, treinado no 2021 DF | 1,1% | **0,6%** |

- **O ajuste no ASVspoof não é o que impede a generalização.** Com o XLS-R original, que nunca viu
  um deepfake, o classificador treinado no 2021 DF falha igual no BRSpeech-DF (25–36%). O ajuste
  melhora o domínio de treino (0,6% contra 1,1%) e quase não muda a transferência.
- **A informação que separa real de sintético já vem do pré-treino auto-supervisionado:** o XLS-R
  original separa o BRSpeech-DF com 0,2% no próprio domínio, sem nenhum treino de detecção.
- **Conclusão para a resposta 6:** o gargalo são os **dados de treino da regra de decisão** (o que se
  aprende com o ASVspoof não serve para o português), não o front-end nem o seu ajuste.

### 6.6 Teste de assinatura (`f3_assinaturas.py`, `assinaturas.json`, `figuras/assinaturas.png`)

Cada fator da Fase 2 desloca a representação dos genuínos VCTK numa direção própria (a
"assinatura"). O cosseno entre essa assinatura e o deslocamento observado nos dados reais diz com que
fator o deslocamento real se parece.

- **Controle positivo, aprovado:** nas camadas 3–12, o deslocamento da hidden track se parece quase
  só com a assinatura do corte de silêncio (cosseno de 0,94–0,96), e a assinatura de acrescentar
  silêncio aponta no sentido oposto (−0,37 a −0,47).
- **Nas camadas finais as assinaturas deixam de ser específicas:** todas as degradações apontam
  para a mesma direção (cossenos de 0,6–0,95 para quase todos os fatores), provavelmente a direção
  "parece falso". A atribuição vale nas camadas 3–12.
- **In-the-Wild × VCTK** (camadas 3–12): parecido com o corte de silêncio (0,40–0,61) e com a
  reverberação (0,40–0,46). Coerente com as Fases 2b e 2c.
- **BRSpeech-DF, 3 locutores "ruins" × 7 "bons"** (camada 12): parecido com **MP3 a 16 kbps (0,66)**,
  banda de telefone (0,53), reverberação (0,50) e volume baixo (0,43); quase nada com silêncio
  (0,08–0,14). **Os locutores "ruins" têm assinatura de canal degradado, não de silêncio**,
  coerente com as medidas acústicas (4.10). O teste não separa entre compressão, banda e eco,
  porque essas assinaturas se parecem entre si.

### 6.7 Limites

- **O probe no próprio domínio pode aprender artefatos do dataset**, e não "sinais de síntese" em
  geral. O controle de TTS deixado de fora (6.3) mostra que parte da separação depende do gerador;
  o que é comum a todos os falsos do BRSpeech-DF (o mesmo processo de geração) e a todos os
  genuínos (o mesmo corpus de origem) continua sem controle. Testar em outro dataset em português
  é o controle que falta.
- Médias no tempo descartam a estrutura temporal; o AASIST usa essa estrutura.
- O BRSpeech-DF tem só 10 locutores no teste; as faixas p10–p90 dos poucos exemplos são largas.

## 7. Síntese: fator × efeito

| Resposta candidata | Fase 1 (observacional) | Fase 2 (causal) | Fase 3 (no modelo) | Situação |
|---|---|---|---|---|
| 1. Condição de gravação/canal | Perda vem dos genuínos; concentrada por locutor; VCC 2018 e `low_mp3` já deslocam no 2021 DF; SNR, nível e banda não separam os locutores "ruins"; os "ruins" têm assinatura de canal degradado (compressão, banda, eco) (6.6) | **Reverberação: EER de 0,3% → 11–24%, voz real vai para o pico dos falsos**; MP3 16 kbps, telefone e volume baixo: 5–6%; ruído aditivo (coberto pelo RawBoost): nenhum efeito | — | **Confirmada como causa; a reverberação é o fator dominante** |
| Silêncio (parte de 1 e 4) | Hidden track: 1,27% → 11,88% na mesma origem; **no In-the-Wild o silêncio inicial aponta para o lado errado** (genuínos 0 s, falsos 0,23 s) | **Atalho nos dois sentidos:** cortar → 5,7%; acrescentar 2 s → 22% dos falsos aceitos | Um probe sobre médias no tempo sofre menos (5,3% contra 12,3%) | **Confirmado**, inclusive como causa de erro em dados reais: padronizar o silêncio reduz a EER do In-the-Wild de 10,48% para 7,64% (Δ pareada −2,84 pp [−4,96; −1,05]); no BRSpeech-DF não muda (coerente com Müller et al., 2021) |
| 2. Gerador novo | Falsos do BRSpeech-DF dão 6,91% contra genuínos do 2021 DF | — | Com o TTS fora do treino do probe: XTTS e YourTTS ~1–3%; F5-TTS e Fish-Speech 17–25% | Secundário hoje; **fronteira restante** para os TTS mais recentes |
| 3. Idioma | Não isolado; 7 dos 10 locutores em português têm EER própria de 7–33% | — | O XLS-R separa o português no próprio domínio (1,2%) | **O idioma não impede a separação nas representações**; o efeito no detector segue em aberto |
| 4. Duração e pré-processamento | Curtos piores; sem atalho de duração | Mudar a janela de 4 s não ajuda (2c) | — | Contribui (duração), secundário |
| Volume | Genuínos do In-the-Wild a −25,9 dBFS (VCTK: −18) | −20 dB: EER 0,3% → 5,7%; **normalizar o volume no In-the-Wild: 10,46% → 6,09%**; com o silêncio padronizado, **4,56%**, sem piora significativa no 2021 DF (2c) | — | **Fator forte no In-the-Wild** |
| 5. Calibração | Recalibrar não muda a EER; minCllr alto | — | — | Não é o gargalo principal |
| 6. Front-end × back-end | — | — | **A informação existe no front-end, inclusive no XLS-R original (0,2% no BRSpeech-DF)**; com o XLS-R original ou com o ajustado, o que se aprende no 2021 DF não transfere; o deslocamento de domínio aponta sempre para o lado "falso"; os 3 locutores "ruins" (detector: 59,5%) são separados com 2,3% por um probe treinado nos outros 7 | **Gargalo nos dados de treino da regra de decisão** |

**Resposta à pergunta norteadora.** Fora do domínio, o detector erra principalmente porque **a
fala genuína gravada em condições diferentes do treino**, sobretudo com **reverberação** e **sem o
silêncio** que o ASVspoof associa à fala real, **é empurrada para o lado "falso"**. O peso de cada
fator muda com o cenário: no In-the-Wild, o atalho do silêncio responde por cerca de 2,8 dos 10,5
pontos de EER e o volume baixo por cerca de 4,4 (cada um medido sozinho); no BRSpeech-DF, nem o
silêncio nem o volume pesam, e a perda se concentra em 3 locutores cuja fala real tem assinatura de
canal degradado. Em todos os casos, a informação para separar as classes continua nas
representações, inclusive no XLS-R antes de qualquer treino de detecção: **o gargalo são os dados com
que a regra de decisão foi treinada.** A informação
para separar real de sintético continua presente nas camadas do XLS-R, inclusive em português,
mas a regra de decisão aprendida no ASVspoof 2019 LA não a usa. O problema é o que o detector
aprendeu (atalhos do conjunto de treino), não o que ele consegue ouvir.

## 8. Validação

- Os carregadores reproduzem **exatamente** todas as EER do Experimento 01 (2,86%, 10,48%, 11,96%,
  12,18%, 39,24%, 35,38% e 41,90%) e as do AKCIT (2,85% e 35,38%).
- A chave completa do 2021 DF foi conferida contra a do Tak: ids, fase `eval` e rótulos idênticos.
- Os 6.587 falsos do teste do BRSpeech-DF têm o genuíno correspondente pelo mapa validado no Exp. 01.
- Fase 2: a condição original reproduz o Exp. 01 em 99,1% das decisões, e a conclusão se mantém
  só com os 594 áudios idênticos ao Exp. 01 (seção 5.3).
- Fase 3: as pontuações extraídas junto com as representações reproduzem o Exp. 01 (correlação de
  0,9999999). O memory-map das camadas foi conferido byte a byte contra o arquivo.
- Todos os scripts foram testados antes da execução real (áudio sintético, modelo de mentira com a
  mesma estrutura, features sintéticas com padrão conhecido, teste rápido no servidor).
- Todo JSON tem um `.meta.json` com o SHA-256 das entradas, as versões e os parâmetros.

## 9. Semana 10: intervenção

O diagnóstico aponta três intervenções com justificativa direta:

0. **Normalizar volume e silêncio antes de pontuar** (sem treino): **já testada** (seção 5.6). Leva o
   In-the-Wild de 10,46% para 4,56% sem piora significativa no 2021 DF e neutraliza o ataque de
   acrescentar silêncio. Custo quase nulo; não resolve o BRSpeech-DF. Na Semana 10, entra como
   pré-processamento padrão, e as intervenções abaixo são medidas sobre ela.
1. **Back-end leve sobre as camadas iniciais do XLS-R, adaptado com poucos exemplos do domínio.**
   É a mais barata e a mais apoiada pela Fase 3: com 50 exemplos por classe em português, a EER
   no BRSpeech-DF cai de 39,5% para 8,7% em locutores não vistos. A avaliação justa precisa de
   locutores e, idealmente, de TTS deixados de fora, e de dados de adaptação separados do teste
   (por exemplo, a divisão de treino do BRSpeech-DF).
2. **Aumento de dados direcionado:** retreinar ou ajustar o detector com reverberação (respostas
   ao impulso) e com silêncio cortado e acrescentado nas duas classes, para remover o atalho e
   cobrir o fator dominante. É a intervenção mais ligada à causa, e a mais cara.

Em qualquer caso: avaliar nos três cenários com o mesmo protocolo (IC por locutor, matriz
cruzada, relato por locutor) e verificar que o 2021 DF não piora.

## Reprodutibilidade

Fase 1, sem GPU e sem áudio: só as pontuações do Experimento 01 e os submódulos. Foi rodada
localmente com Python 3.13.12, numpy 2.4.4, scipy 1.17.1 e scikit-learn 1.8.0. Os scripts são
compatíveis com Python 3.8, a versão da imagem do Experimento 01. A partir da raiz do
repositório, com os submódulos inicializados (`git submodule update --init`):

Fase 1 (PC):

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

Fase 2 (servidor; perturbações fora do container, num ambiente Python isolado com numpy, scipy,
soundfile e imageio-ffmpeg; ver PROCESSO, problema 7):

```bash
cd experimentos/02-diagnostico/scripts
python -B f2_perturbar.py --audio ~/matheus/dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac \
    --saida ~/matheus/dados/exp02_perturbacoes --raiz-container /dados/exp02_perturbacoes --n 400
cd ~/matheus/residencia
S=experimentos/01-linha-de-base/scripts/em_docker.sh
CK="--checkpoint /ckpt/Best_LA_model_for_DF.pth --xlsr /ckpt/xlsr2_300m.pt"
bash $S 0 python inferir.py --lista /dados/exp02_perturbacoes/lista.tsv --workers 8 $CK \
    --saida ../../02-diagnostico/resultados/fase2/pontuacoes_repeat.txt
bash $S nenhuma python ../../02-diagnostico/scripts/f2_analisar.py \
    --pontuacoes ../../02-diagnostico/resultados/fase2/pontuacoes_repeat.txt \
    --rotulos /dados/exp02_perturbacoes/rotulos.tsv --saida ../../02-diagnostico/resultados/fase2/perturbacoes.json
# no PC:
python -B f2_figura.py --resultado ../resultados/fase2/perturbacoes.json --meta ../resultados/fase2/perturbacoes.meta.json
```

Fase 3 (extração no servidor; probing no PC):

```bash
bash $S nenhuma python ../../02-diagnostico/scripts/f3_selecionar.py
bash $S 0 python ../../02-diagnostico/scripts/f3_extrair.py --lista ../../02-diagnostico/listas/fase3/lista.tsv \
    $CK --saida ../../02-diagnostico/resultados/fase3/features.npz
# no PC (features.npz fica fora do git, ~526 MB; o SHA-256 está no .meta.json):
python -B f3_probing.py --features ../resultados/fase3/features.npz --amostra ../listas/fase3/amostra.tsv
python -B f3_poucos_exemplos.py --features ../resultados/fase3/features.npz --amostra ../listas/fase3/amostra.tsv
python -B f3_complementos.py --features ../resultados/fase3/features.npz --amostra ../listas/fase3/amostra.tsv
```

Medidas acústicas (servidor, depois PC):

```bash
bash $S nenhuma python ../../02-diagnostico/scripts/f1_qualidade_audio.py \
    --lista ../../02-diagnostico/listas/fase3/lista.tsv --saida ../../02-diagnostico/resultados/qualidade/medidas.tsv
# no PC:
python -B f1_qualidade_analise.py --medidas ../resultados/qualidade/medidas.tsv --amostra ../listas/fase3/amostra.tsv
```

Fase 2b, normalização do silêncio (servidor: geração fora do container no ambiente isolado;
inferência e análise no container):

```bash
python -B f2_silencio_real.py --raiz-host-dados ~/matheus/dados \
    --saida ~/matheus/dados/exp02_silencio --raiz-container /dados/exp02_silencio
for c in aparar padronizar; do for s in itw brs; do
  bash $S 0 python inferir.py --lista /dados/exp02_silencio/${c}_${s}.lista.tsv --workers 8 $CK \
      --saida ../../02-diagnostico/resultados/silencio/pontuacoes_${c}_${s}.txt
done; done
bash $S nenhuma python ../../02-diagnostico/scripts/f2_silencio_analise.py
```

Fase 2c, outras intervenções e a combinação (geração no ambiente isolado; inferência no container;
análise no ambiente isolado):

```bash
python -B f2_normalizar_real.py --raiz-host-dados ~/matheus/dados \
    --audio-df21 ~/matheus/dados/asvspoof2021_df/ASVspoof2021_DF_eval/flac \
    --saida ~/matheus/dados/exp02_normalizar --raiz-container /dados/exp02_normalizar \
    --cenarios itw brs df21 --condicoes controle equalizar volume janela_meio janela_fim padronizar volume_padronizar
for c in controle equalizar volume janela_meio janela_fim padronizar volume_padronizar; do for s in itw brs; do
  bash $S 0 python inferir.py --lista /dados/exp02_normalizar/${c}_${s}.lista.tsv --workers 8 $CK \
      --saida ../../02-diagnostico/resultados/normalizar/pontuacoes_${c}_${s}.txt
done; done
for c in controle volume padronizar volume_padronizar; do
  bash $S 0 python inferir.py --lista /dados/exp02_normalizar/${c}_df21.lista.tsv --workers 8 $CK \
      --saida ../../02-diagnostico/resultados/normalizar/pontuacoes_${c}_df21.txt
done
python -B f2_silencio_analise.py --pasta ../resultados/normalizar \
    --condicoes controle equalizar volume janela_meio janela_fim --base controle --saida normalizar_real.json
python -B f2_silencio_analise.py --pasta ../resultados/normalizar --cenarios itw brs df21_eval df21_hidden \
    --condicoes controle volume padronizar volume_padronizar --base controle --saida combinado.json
```

Fase 3, controles com as features no servidor (ambiente isolado com scikit-learn e matplotlib):

```bash
bash $S 0 python ../../02-diagnostico/scripts/f3_extrair_original.py --lista ../../02-diagnostico/listas/fase3/lista.tsv \
    --xlsr /ckpt/xlsr2_300m.pt --saida ../../02-diagnostico/resultados/fase3/features_xlsr_original.npz
grep -E $'^(id\t|(original|reverb_0\\.6s|aparar_40db|mp3_16k|banda_telefone|ganho_-20db|silencio_1\\.0s|ruido_branco_10db)__)' \
    ~/matheus/dados/exp02_perturbacoes/lista.tsv > ../../02-diagnostico/listas/fase3/lista_assinaturas.tsv
bash $S 0 python ../../02-diagnostico/scripts/f3_extrair.py --lista ../../02-diagnostico/listas/fase3/lista_assinaturas.tsv \
    $CK --saida ../../02-diagnostico/resultados/fase3/features_assinaturas.npz
python -B f3_probing.py --features ../resultados/fase3/features_xlsr_original.npz --amostra ../listas/fase3/amostra.tsv --nome probing_xlsr_original
python -B f3_assinaturas.py --features ../resultados/fase3/features.npz --amostra ../listas/fase3/amostra.tsv \
    --features-f2 ../resultados/fase3/features_assinaturas.npz --selecao ../resultados/fase2/selecao.tsv
python -B f3_dominio.py --features ../resultados/fase3/features.npz --amostra ../listas/fase3/amostra.tsv
```

A semente é fixa (0) em todos os bootstraps e sorteios. Os parâmetros (B, k e faixas) estão
no `.meta.json` de cada resultado.

## Referências

1. H. Tak et al. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0
   and data augmentation*. Odyssey 2022. arXiv:2202.12233.
2. A. C. Ferro Filho et al. *BRSpeech-DF: A Deep Fake Synthetic Speech Dataset for Portuguese
   Zero-Shot TTS*. EMNLP 2025. Repositório com a chave completa do 2021 DF e as pontuações de SSL,
   SLS e AASIST.
3. N. Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022 (In-the-Wild).
4. N. Brümmer, J. du Preez. *Application-independent evaluation of speaker detection*. Computer
   Speech & Language, 2006 (Cllr). _(referência de memória: conferir os dados antes de citar)_
5. X. Liu et al. *ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild*. IEEE/ACM
   TASLP, v. 31, p. 2507–2522, 2023. arXiv:2210.02437. (Análise dos subconjuntos ocultos; conferir na
   seção correspondente o efeito do corte de silêncio antes de citar números.)
6. N. M. Müller, F. Dieckmann, P. Czempin, R. Canals, K. Böttinger, J. Williams. *Speech is Silver,
   Silence is Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop.
   arXiv:2106.12914. (Os números de EER do artigo variam entre as versões do arXiv; aqui só se cita
   o achado qualitativo.)
7. H. Tak et al. *RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker
   Verification Anti-Spoofing*. ICASSP 2022. arXiv:2111.04433.
8. N. M. Müller et al. *MLAAD: The Multi-Language Audio Anti-Spoofing Dataset*. IJCNN 2024.
   arXiv:2401.09512. (Candidato para isolar o fator idioma; os números variam entre versões.)
