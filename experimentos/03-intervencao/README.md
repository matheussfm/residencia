# Experimento 03 – Intervenção na regra de decisão

Testa, intervindo nela, a conclusão do Experimento 02: a informação para separar fala real e
sintética existe nas camadas do XLS-R, mas a regra de decisão aprendida no ASVspoof não transfere
("o gargalo são os dados da regra"). O detector é sempre o de Tak et al. (2022), XLS-R 300M +
AASIST, com o checkpoint publicado e **congelado**: nada do modelo é treinado. Só se treinam
classificadores lineares (probes) sobre as representações dele.

> Status: Etapas 0 (pré-registro, métricas e divisão), 1 (linhas de base), 2 (I1) e 3 (I2)
> concluídas. Etapa 4 (opcional) e fechamento pendentes. As seções 1 a 4 foram escritas **antes** de qualquer resultado do Exp. 03.

---

## 1. Ancoragem

**Área:** detecção de deepfakes de fala, linha de generalização.

**Papel deste experimento:** o Exp. 01 mediu a perda fora do domínio (EER de 2,86% no ASVspoof 2021
DF, 10,48% no In-the-Wild e 39,24% no BRSpeech-DF). O Exp. 02 encontrou as causas: fala genuína
com reverberação, volume baixo, canal degradado e sem o silêncio que o ASVspoof associa à fala real
é empurrada para o lado "falso". Ele também mostrou que um probe linear treinado no próprio domínio
separa as classes, até no XLS-R original. O Exp. 03 é a intervenção que o diagnóstico sugere. Ele
fecha a coluna "intervir no treino" da matriz de testes da Semana 9.

## 2. Problema

1. Se a regra for aprendida com dados do domínio de origem que cubram os fatores identificados
   (reverberação, silêncio, volume, canal), ela generaliza melhor **sem nenhum dado do alvo**? (I1)
2. Com poucos exemplos do alvo, quanto se recupera **numa avaliação justa**, com locutores e TTS
   fora da adaptação? (I2)
3. Combinando tudo, qual é a melhor "receita" barata, e quanto ela custa no domínio de origem?

## 3. Frameworks de sustentação

- **Probe linear da Fase 3 do Exp. 02** (`f3_probing.py`, importado): padronização + regressão
  logística com classes de peso igual, sobre a média no tempo de cada representação.
- **Representações da Fase 3** (`f3_extrair.py`, importado sem alteração): 24 camadas do XLS-R, a
  saída final e o embedding do AASIST, com o pré-processamento do Tak (repetição, corte de 4 s).
- **EER** (métrica principal; `compute_eer` do Tak, via `_diag.py`).
- **minDCF do ASVspoof 5, Track 1** [2, seção 3.1]: DCF(τ) = β·P_miss(τ) + P_fa(τ), com
  β = (C_miss / C_fa)·(1 − π_spf)/π_spf, C_miss = 1, C_fa = 10 e π_spf = 0,05, o que dá β ≈ 1,90.
  P_miss é a taxa de genuínos rejeitados e P_fa, a de falsos aceitos. O minDCF é o mínimo sobre os
  limiares. Aceitar tudo dá DCF = 1, então minDCF ≤ 1. Os parâmetros foram conferidos no artigo do
  ASVspoof 5. O artigo remete os detalhes a um plano de avaliação, que não foi consultado
  diretamente. Os parâmetros ficam gravados em todo `.meta.json`.
- **IC de 95% por bootstrap de locutores** (B = 1.000) e **diferença pareada**: os mesmos locutores
  sorteados nas duas condições comparadas (`_diag.py`, como na Fase 2b do Exp. 02).
- **Aumento de dados** com as funções do `f2_perturbar.py` do Exp. 02 (reverberação sintética,
  corte e acréscimo de silêncio, ganho, banda de telefone, MP3).

## 4. Pré-registro

### 4.1 Dados e divisão

- **Origem:** a amostra df21_eval da Fase 3 (3.000 trials, 500 por classe de cada origem: VCTK,
  VCC 2018 e VCC 2020; 88 locutores), dividida **por locutor** em treino, validação e teste
  (`e0_dividir.py`, semente 0; `listas/divisao_df21_eval.tsv`). A divisão é feita dentro de cada
  origem, com metas de 60/20/20 dos áudios nas duas classes:

  | Partição | Locutores | Genuínos | Falsos |
  |---|---|---|---|
  | treino | 48 | 791 | 803 |
  | validação | 20 | 338 | 360 |
  | teste | 20 | 371 | 337 |

  As três partições têm as três origens. O VCC 2018 tem só 3 locutores na validação e 3 no teste.
- **Testes fora do domínio** (amostra da Fase 3): In-the-Wild (1.500 por classe, 53 locutores),
  BRSpeech-DF teste (1.320 por classe, 10 locutores) e hidden track (1.000 por classe).
- **Hidden track, duas formas.** Os 46 locutores da hidden track estão todos no df21_eval, e 27
  deles caíram no treino. A hidden track é avaliada (a) completa e (b) só com os 10 locutores da
  partição de teste (200 genuínos e 205 falsos). A diferença entre (a) e (b) mede quanto pesa ter
  visto o locutor. Na Fase 3 do Exp. 02, o probe foi treinado com metade dos locutores do eval e
  testado na hidden track completa; esse vazamento não tinha sido declarado.
- **Domínio de origem, teste:** a partição de teste do df21_eval ("df21 teste"), para medir o custo
  de cada intervenção.

### 4.2 Linhas de base

- **B0, detector completo:** as pontuações do Exp. 01 nos mesmos ids.
- **B1, detector + `volume_padronizar`:** as pontuações da Fase 2c do Exp. 02 (corta o silêncio das
  pontas, leva o RMS a −18 dBFS e acrescenta 0,5 s de silêncio). A diferença pareada de B1 é medida
  contra a condição **controle** da Fase 2c, que passa pelo mesmo caminho de leitura e gravação (Exp.
  02, PROCESSO, problema 14). B0 também é reportado nos mesmos ids, para referência.
- **B2, probe limpo:** o probe treinado na partição de treino, sem aumento.

### 4.3 Escolha do probe (vale para B2, I1 e I2)

- **Representações fixadas a priori:** camadas 5, 12 e 24 do XLS-R e o embedding do AASIST.
- **Grade de C fixada a priori:** 0,001, 0,01, 0,1 e 1.
- O probe é treinado **só na partição de treino**. A representação e o C são escolhidos pela menor
  EER na **validação da origem**, com desempate pelo minDCF e depois pelo menor C. **O teste nunca
  é usado para escolher nada.** Na Fase 3 do Exp. 02, a "melhor camada" foi escolhida no próprio
  teste; os números de lá (por exemplo, 6,3% no In-the-Wild) são, portanto, otimistas.
- Para separar o efeito da intervenção do efeito da escolha, toda comparação é reportada também em
  **cada uma das quatro representações**, com o C escolhido na validação para aquela
  representação.

### 4.4 Intervenções

**I1, aumento de dados nas duas classes, sem dado do alvo** (`i1_aumentar.py`). Uma cópia
aumentada de cada áudio da partição de treino, com semente fixa e a condição sorteada registrada
por áudio:

| Variante | Aumento |
|---|---|
| (a) +reverberação | RT60 sorteado em [0,2; 1,0] s (resposta ao impulso sintética do `f2_perturbar.py`) |
| (b) +silêncio/volume | cortar o silêncio das pontas ou acrescentar silêncio no início; ganho em [−20; +6] dB |
| (c) +canal | banda de telefone ou MP3 a 16 kbps |
| (d) tudo | sorteio combinado de (a), (b) e (c): cada fator com probabilidade 1/2, pelo menos um, na ordem sala → canal → silêncio/volume |
| controle | a mesma cópia, sem aumento (só lida, convertida para 16 kHz e regravada) |

Detalhes do sorteio: em (b), corta o silêncio das pontas (máximo − 40 dB) ou acrescenta de 0,25 a
2 s de silêncio digital no início, com probabilidade 1/2 cada, e depois aplica o ganho; em (c),
telefone ou MP3, com probabilidade 1/2 cada.

O probe de cada variante é treinado com o treino limpo **mais** a cópia aumentada. A variante
**controle** foi acrescentada ao pré-registro antes de qualquer execução da Etapa 2. Ela separa o
efeito do aumento do efeito de duplicar o treino e de passar pelo caminho de leitura e gravação das
cópias. Por isso, cada variante também é comparada com o controle (I1_v − I1_controle). As
representações vêm do `f3_extrair.py`, sem alteração. A escolha segue a seção 4.3, na validação
limpa da origem. **As perturbações sintéticas da Fase 2 não são usadas como teste de ganho**: seria
circular. O teste é sempre em dados reais fora do domínio.

**I1 + B1:** o probe (d) aplicado às representações dos áudios de teste normalizados
(`volume_padronizar`, só os ids da amostra da Fase 3). O probe limpo também é aplicado a eles
(B2 + B1), para separar o efeito da normalização do efeito do aumento.

**I2, poucos exemplos em português, avaliação justa** (só o BRSpeech-DF):
- k = 10, 50 e 200 exemplos por classe; 20 sorteios por k. Os mesmos sorteios valem para todos
  os métodos e todos os k, de modo que as comparações são pareadas.
- Cada sorteio usa 3 dos 5 TTS para a adaptação e deixa os outros 2 para o teste, em rodízio: cada
  uma das 10 combinações aparece em 2 sorteios. Os 10 locutores são divididos ao acaso em 5 para a
  adaptação e 5 para o teste.
- A incerteza é dada pela mediana, pelo p10 e pelo p90 sobre os 20 sorteios. As diferenças são
  pareadas por sorteio, com a fração de sorteios em que o método fica melhor. A adaptação usa só genuínos dos locutores de
  adaptação e falsos desses locutores feitos pelos 3 TTS de adaptação. O teste usa só genuínos dos
  locutores de teste e falsos desses locutores feitos pelos 2 TTS restantes.
- **Forma A (principal):** probe treinado com o treino da origem (limpo, ou limpo + (d)) **mais**
  os k exemplos do alvo. Os exemplos do alvo recebem, juntos, o mesmo peso total que os da origem,
  para não ficarem diluídos.
- **Forma B (referência):** probe treinado **só** com os k exemplos do alvo, como no Exp. 02.
- Representação e C: os escolhidos na validação da origem para o probe de partida (limpo ou (d)).
  Na forma B, a mesma representação, com C = 0,1, o valor do Exp. 02. As quatro representações
  também são reportadas.
- O detector completo (B0) é avaliado nos mesmos subconjuntos de teste de cada sorteio.

**Etapa 4 (opcional):** as divisões de treino e validação do BRSpeech-DF, se o download for viável,
como (a) fonte de adaptação da I2, com o teste oficial intocado, e (b) segundo teste, mais
representativo que a divisão de teste oficial (Exp. 02, seção 4.6: o teste oficial fica no
percentil 94 dos sorteios de locutores).

### 4.5 Critério e leitura prevista

Uma intervenção **melhora** um cenário quando o IC de 95% da diferença pareada de EER fica todo
abaixo de zero. Ela **custa** no domínio de origem quando o IC da diferença no df21 teste fica todo
acima de zero. São muitas comparações; uma diferença isolada no limite do IC deve ser lida com
cuidado. Todo resultado nulo é registrado.

| Resultado | Leitura prevista |
|---|---|
| B2 não supera B0 fora do domínio, com a escolha honesta | A vantagem do probe vista no Exp. 02 dependia de escolher a camada no teste |
| Hidden track completa bem melhor que só com os locutores de teste | O número do Exp. 02 na hidden track estava inflado por locutores vistos |
| I1 melhora In-the-Wild e hidden track, mas não o BRSpeech-DF | A perda em português não é coberta por esses fatores |
| I1 melhora também o BRSpeech-DF | Reforço causal de "o gargalo são os dados da regra" |
| (b) melhora a hidden track | Esperado por construção: o aumento inclui o corte de silêncio, o fator da hidden track. Não conta como evidência de generalização |
| (a) ou (c) melhoram os 3 locutores "ruins" do BRSpeech-DF | Coerente com a assinatura de canal degradado (Exp. 02, seção 6.6) |
| Ablação (a), (b), (c) | Qual fator responde pelo ganho em cada cenário |
| I1 + B1 melhor que I1 e que B1 sozinhos | Normalizar a entrada e aumentar o treino se somam |
| Custo no df21 teste com IC acima de zero | Registrado como custo da intervenção |
| I2 com TTS fora continua baixo | A separação é geral, não depende de ter visto o gerador |
| I2 com TTS fora sobe muito em relação ao Exp. 02 (8,7% com 50 exemplos, TTS vistos) | Parte da separação no próprio domínio era artefato do gerador ou do processo de geração do dataset |
| Forma A melhor que a forma B | O treino na origem ajuda a adaptação; pior, ele atrapalha |
| A com (d) melhor que A com o limpo | O aumento de dados e os poucos exemplos se somam (I1 + I2) |

## 5. Resultados

### 5.1 Etapa 1: linhas de base

(`e1_linhas_de_base.py`; `resultados/etapa1/linhas_de_base.json`; as pontuações dos probes B2 em
`pontuacoes_b2.tsv`.) Todos os números são da amostra da Fase 3, com IC de 95% por bootstrap de
locutores (B = 1.000).

**Escolha do B2 na validação da origem.** Melhor representação: **camada 12, C = 0,1** (EER de 0,86%
na validação). Com o C escolhido para cada uma, a camada 5 dá 1,72%, a camada 24 dá 2,43% e o
embedding do AASIST, 3,29%.

**Conferência.** B0 nos mesmos ids reproduz o Exp. 02 (Fase 3: 9,8% no In-the-Wild, 39,5% no
BRSpeech-DF e 12,3% na hidden track), e B1 na hidden track reproduz a Fase 2c (7,60%).

| Conjunto | B0 detector | Controle 2c | B1 detector + normalização | B2 probe limpo (camada 12) |
|---|---|---|---|---|
| In-the-Wild | 9,80% [7,18; 14,19] · 0,278 | 9,87% · 0,276 | **4,20%** [2,56; 6,26] · 0,115 | 7,60% [5,67; 10,95] · 0,206 |
| BRSpeech-DF | 39,47% [19,06; 52,57] · 1,000 | 39,47% · 1,000 | 41,21% [18,25; 58,67] · 0,998 | 36,21% [16,71; 52,29] · 1,000 |
| Hidden track, completa | 12,30% [10,16; 14,59] · 0,315 | 11,10% · 0,262 | 7,60% [6,02; 9,10] · 0,143 | **6,60%** [5,39; 8,03] · 0,152 |
| Hidden track, só locutores de teste | 13,09% [8,75; 15,37] · 0,318 | 11,60% · 0,267 | 8,40% [4,28; 11,69] · 0,150 | **6,91%** [3,96; 12,49] · 0,174 |
| df21 teste | 4,10% [2,08; 6,32] · 0,104 | 2,97% · 0,075 | 3,25% [1,18; 6,74] · 0,050 | 0,57% [0,22; 2,14] · 0,016 |

Cada célula traz a EER [IC 95%] e o minDCF.

Diferenças pareadas de EER, em pontos percentuais (IC 95%):

| Conjunto | B1 − controle | B2 (camada 12) − B0 |
|---|---|---|
| In-the-Wild | **−5,67 [−9,17; −3,82]** | **−2,20 [−4,90; −0,27]** |
| BRSpeech-DF | +1,74 [−1,46; +5,94] (n.s.) | −3,26 [−5,30; +0,90] (n.s.) |
| Hidden track, completa | **−3,50 [−5,40; −2,01]** | **−5,70 [−7,53; −4,01]** |
| Hidden track, só teste | −3,21 [−7,01; 0,00] (limite) | −6,17 [−8,25; 0,00] (limite) |
| df21 teste | +0,28 [−1,38; +3,18] (n.s.) | **−3,53 [−5,17; −1,24]** |

EER do probe limpo em cada representação fixada a priori (o C de cada uma foi escolhido na validação):

| Conjunto | Camada 5 | Camada 12 | Camada 24 | AASIST | B0 |
|---|---|---|---|---|---|
| In-the-Wild | 14,47% | 7,60% | 7,93% | 8,27% | 9,80% |
| BRSpeech-DF | **30,38%** | 36,21% | 36,44% | 37,95% | 39,47% |
| Hidden track, completa | 7,10% | 6,60% | 7,00% | 7,80% | 12,30% |
| df21 teste | 1,55% | 0,57% | 1,83% | 2,12% | 4,10% |

**Leituras (em relação ao pré-registro, seção 4.5):**

1. **Com a escolha honesta, o probe limpo ainda supera o detector no In-the-Wild, mas por menos.**
   A diferença é de −2,2 pontos, com IC que exclui o zero (7,60% contra 9,80%). O Exp. 02 dava 6,3%
   com a camada 23, escolhida no teste. A vantagem existe, mas o número de lá era otimista.
2. **Nenhuma representação é boa em todos os cenários.** A camada 5 é a melhor no BRSpeech-DF
   (30,4%, Δ de −9,1 pontos [−13,3; −3,3] contra B0) e a pior no In-the-Wild (14,5%, pior que o
   detector). A validação da origem escolhe a camada 12, que vai bem no In-the-Wild e na hidden
   track, mas não muda o BRSpeech-DF de forma significativa. Escolher a camada por cenário exigiria
   dados do alvo; esse é o papel da I2.
3. **O vazamento de locutores na hidden track pesa pouco.** Completa ou só com os locutores de teste,
   o probe dá 6,60% e 6,91%, dentro do IC um do outro. A leitura prevista de "número inflado" não se
   confirma. Com só 10 locutores, o IC da versão "só teste" é largo, e as diferenças pareadas dela
   ficam no limite do zero.
4. **A normalização da entrada (B1) se reproduz na amostra:** −5,7 pontos no In-the-Wild, −3,5 na
   hidden track, nada no BRSpeech-DF e nenhum custo significativo no df21 teste.
5. **O df21 teste não mede o custo de B2 contra B0 de forma justa.** O probe foi treinado no próprio
   df21 eval (outros locutores, mesmos corpora e ataques); o detector, no ASVspoof 2019 LA. O 0,57%
   do B2 reflete esse treino dentro do domínio. O custo das intervenções da Etapa 2 será medido
   **contra o B2**, que tem a mesma origem.
6. **O minDCF satura no BRSpeech-DF.** Com π_spf = 0,05, rejeitar um genuíno custa β ≈ 1,9 vezes
   mais que aceitar um falso. Quando as classes se sobrepõem muito, o melhor limiar é aceitar tudo,
   e o minDCF fica em 1. Por isso ele não distingue os métodos no BRSpeech-DF (todos entre 0,86 e
   1,00). Observação nossa, a partir da definição: no ponto de operação do ASVspoof 5, o minDCF só
   informa quando o detector já é razoável. A EER segue como métrica principal.

### 5.2 Etapa 2: I1, aumento de dados nas duas classes

(`i1_aumentar.py`, `i1_probing.py`; `resultados/etapa2/i1.json`, `condicoes.tsv` com o sorteio de cada
cópia, `i1_log.txt`.) Rodado no servidor (Python 3.10.12, scikit-learn 1.7.2). B = 1.000.

**O aumento muda o que o detector vê.** A EER do detector completo nas cópias do treino (descritiva,
não é teste de ganho) vai de 3,39% nos originais para 2,76% no controle, 34,76% com reverberação (a),
14,81% com silêncio/volume (b), 9,10% com canal (c) e 30,24% com tudo (d). É a mesma ordem de fatores
da Fase 2 do Exp. 02.

**Conferências.** O probe limpo retreinado no servidor reproduz o B2 da Etapa 1 (diferença máxima de
0,0001 na camada 12). As pontuações do detector gravadas na extração dos áudios normalizados
reproduzem as da Fase 2c (correlação de 0,99999999996).

**Escolha na validação limpa.** Camada 12 para limpo, controle, (c) e (d); camada 24 para (a);
camada 5 para (b).

EER [IC 95%] com a representação escolhida de cada método:

| Conjunto | B0 | B2 limpo | I1 controle | I1 (a) | I1 (b) | I1 (c) | I1 (d) |
|---|---|---|---|---|---|---|---|
| In-the-Wild | 9,80% | **7,60%** | 8,13% | 8,20% | 13,47% | 8,20% | 9,20% |
| BRSpeech-DF | 39,47% | 36,21% | 34,62% | 37,80% | **33,71%** | 36,36% | 36,36% |
| Hidden track, completa | 12,30% | 6,60% | 7,50% | 8,70% | **5,60%** | 6,90% | 7,50% |
| Hidden track, só teste | 13,09% | 6,91% | 9,38% | 7,41% | 7,90% | 6,91% | 6,91% |
| df21 teste | 4,10% | **0,57%** | 0,57% | 2,12% | 2,12% | 1,13% | 1,55% |

Diferença pareada de EER contra o B2, em pontos percentuais (IC 95%):

| Conjunto | Controle − B2 | (a) − B2 | (b) − B2 | (c) − B2 | (d) − B2 |
|---|---|---|---|---|---|
| In-the-Wild | +0,53 [−0,61; +1,15] | +0,60 [−1,65; +2,52] | **+5,87 [+2,50; +9,26]** | +0,60 [−0,44; +1,24] | +1,60 [−0,07; +2,68] |
| BRSpeech-DF | −1,59 [−3,60; +0,79] | +1,59 [−0,93; +4,07] | −2,50 [−4,77; +1,62] | +0,15 [−1,36; +2,35] | +0,15 [−1,93; +2,17] |
| Hidden track, completa | **+0,90 [+0,09; +1,72]** | **+2,10 [+0,37; +3,19]** | −1,00 [−2,58; +0,65] | +0,30 [−0,32; +1,12] | +0,90 [−0,21; +2,09] |
| Hidden track, só teste | +2,47 [0,00; +4,43] | +0,49 [−4,57; +5,05] | +0,99 [−3,97; +4,89] | 0,00 [−2,18; +1,35] | 0,00 [−2,23; +1,40] |
| df21 teste (custo) | 0,00 [−0,89; +0,29] | **+1,55 [+0,11; +2,49]** | +1,55 [0,00; +2,98] | +0,57 [−0,51; +0,87] | +0,98 [−0,24; +1,93] |

Com a **representação fixa**, cada variante contra o probe limpo na mesma representação: as diferenças
significativas, de um lado e de outro, são estas (o JSON tem todas):

- (b) melhora a hidden track completa nas três camadas do XLS-R: −1,5 [−2,4; −0,2] na 5,
  −1,4 [−2,7; −0,3] na 12 e −1,8 [−2,9; −0,5] na 24;
- (a) piora a hidden track completa nas três camadas (+1,7 a +2,3), o BRSpeech-DF na camada 12
  (+2,0 [+0,9; +4,6]) e o df21 teste na camada 5 (+2,0 [+0,6; +4,8]; na 12, no limite);
- no embedding do AASIST, (d) e (a) melhoram o In-the-Wild em −1,3 [−2,2; −0,4] e −1,1 [−2,1; −0,1],
  e (d) melhora a hidden track só teste em −2,5 [−3,7; −0,6];
- (c) piora a hidden track na camada 24 (+2,9 [+1,7; +4,4]).

**I1 + B1** (probes aplicados aos áudios de teste normalizados; base: o mesmo probe na condição controle):

| Conjunto | B1 detector | B2 + controle | B2 + B1 | I1 (d) + controle | I1 (d) + B1 | (I1 (d) + B1) − B1 |
|---|---|---|---|---|---|---|
| In-the-Wild | **4,20%** | 7,67% | 4,60% (Δ **−3,07 [−4,86; −1,78]**) | 9,20% | 4,80% (Δ **−4,40 [−6,44; −2,66]**) | +0,60 [−0,90; +2,42] |
| BRSpeech-DF | 41,21% | 36,89% | 38,26% (Δ +1,36 [−0,98; +3,86]) | 36,89% | 39,39% (Δ **+2,50 [+0,23; +4,35]**) | −1,82 [−3,76; +1,18] |
| Hidden track, completa | 7,60% | 6,70% | 6,90% (Δ +0,20) | 7,50% | 7,20% (Δ −0,30) | −0,40 [−1,43; +0,71] |
| df21 teste | 3,25% | 0,85% | 2,97% (Δ **+2,12 [+0,48; +4,93]**) | 1,55% | 2,12% (Δ +0,57 [0,00; +3,03]) | −1,13 [−2,61; 0,00] |

**Leituras (em relação ao pré-registro, seção 4.5):**

1. **O aumento de dados não melhorou a regra fora do domínio. É um resultado nulo.** Com a escolha
   honesta, nenhuma variante supera o probe limpo no In-the-Wild, no BRSpeech-DF ou na hidden track só
   com os locutores de teste. A leitura prevista "I1 melhora o In-the-Wild e a hidden track, mas não o
   BRSpeech-DF" não se confirma: nem o In-the-Wild melhora. A leitura "a perda em português não é
   coberta por esses fatores" se confirma. Nos 3 locutores "ruins" do BRSpeech-DF, todas as variantes
   ficam entre 55,7% e 61,7% (B2: 60,4%).
2. **A reverberação sintética, o fator causal mais forte da Fase 2, piora quando entra no treino.** Ela
   piora a hidden track e o BRSpeech-DF e custa 1,55 ponto no df21 teste. Interpretação, a verificar: a
   resposta ao impulso sintética (ruído com decaimento exponencial) não representa as salas reais
   desses dados, e a assinatura do deslocamento real só se parecia em parte com a da reverberação
   (cossenos de 0,40–0,50; Exp. 02, seção 6.6). Não há sinal de que a regra tenha só "perdido
   separação" em geral: na validação limpa, (a) é pior que o limpo nas camadas 5 e 12, mas melhor nas
   camadas 24 e AASIST.
3. **O (b) melhora a hidden track, como previsto por construção.** O aumento inclui o corte do
   silêncio, que é o fator da hidden track. Pelo pré-registro, isso não conta como evidência de
   generalização. Com a representação escolhida (camada 5), a variante (b) piora muito o
   In-the-Wild (+5,9 pontos), mas na camada 12 fixa não muda nada: é o efeito da escolha.
4. **O controle já move a EER em cerca de 1 ponto** (+0,9 na hidden track completa; +2,5, no limite, na
   só teste). Duplicar o treino com o caminho de leitura do servidor tem um efeito do mesmo tamanho que
   o das variantes. Diferenças da I1 abaixo de ~1–2 pontos não se distinguem desse efeito.
5. **Ganhos pequenos só no embedding do AASIST** ((d) e (a) no In-the-Wild, cerca de −1,2 ponto). Com
   dezenas de comparações, esperam-se alguns IC que excluem o zero por acaso. Fica registrado, sem peso
   de evidência.
6. **Normalizar e aumentar não se somam.** No In-the-Wild, a normalização faz todo o trabalho: o probe
   (d) sobre os áudios normalizados (4,80%) não supera o próprio detector normalizado (B1, 4,20%). No
   BRSpeech-DF, a normalização piora os probes (+2,5 pontos para o (d)). No df21 teste, B2 + B1 custa
   2,1 pontos.
7. **Custo no domínio de origem.** (a) custa +1,55 [+0,11; +2,49] e (b) +1,55 [0,00; +2,98] (limite). As
   demais não têm custo significativo.

**O que isso diz sobre "o gargalo são os dados da regra".** A conclusão do Exp. 02 não é refutada,
mas fica mais estreita. Os dados que faltam à regra não são substituídos por perturbações sintéticas
dos fatores identificados, pelo menos com um probe linear sobre médias no tempo e 1.594 áudios de
origem. Os fatores explicam o deslocamento (Exp. 02), mas simulá-los no treino não reproduz o que falta.
O retreino completo do detector com aumento (seção "Trabalho futuro") é o teste mais forte que fica em
aberto.

### 5.3 Etapa 3: I2, poucos exemplos em português com locutores e TTS fora

(`i2_poucos_exemplos.py`; `resultados/etapa3/i2.json`, com os 20 sorteios um a um.) Nenhum sorteio
foi pulado por falta de exemplos. Cada TTS ficou fora da adaptação em 8 dos 20 sorteios.

EER mediana sobre os 20 sorteios [p10; p90]. A representação principal (escolhida na validação da
origem) é a camada 12 para os três métodos. A camada 5 entra como secundária, porque era a melhor no
BRSpeech-DF na Etapa 1 e no Exp. 02.

| k por classe | B0 detector | A limpo (origem + alvo) | A (d) (origem aumentada + alvo) | B (só alvo) |
|---|---|---|---|---|
| 0 | 38,2% [22,8; 53,9] | 37,1% (B2) | 36,6% (I1 d) | — |
| 10 | | 32,8% [18,0; 46,9] | 32,9% [20,1; 47,8] | 42,2% [34,4; 51,4] |
| 50 | | 30,8% [20,9; 44,4] | 29,5% [22,0; 43,1] | 34,9% [26,7; 51,6] |
| 200 | | 31,6% [18,4; 43,5] | 29,9% [18,9; 41,2] | 36,6% [20,0; 56,7] |
| **Camada 5**, k = 0 | | 33,3% (B2) | 33,4% (I1 d) | — |
| Camada 5, k = 10 | | 27,2% [11,4; 38,8] | 24,4% [11,6; 35,2] | 33,4% [17,9; 46,8] |
| Camada 5, k = 50 | | 24,8% [8,3; 40,3] | **21,3%** [8,7; 36,7] | 22,1% [14,1; 41,2] |
| Camada 5, k = 200 | | 18,8% [5,0; 37,6] | **18,4%** [4,1; 31,3] | 19,7% [7,0; 47,6] |

Diferenças pareadas por sorteio (mediana em pontos percentuais [p10; p90]; fração dos sorteios em
que o primeiro método é melhor):

| Comparação | Camada 12, k = 10 | Camada 12, k = 50 | Camada 12, k = 200 | Camada 5, k = 50 | Camada 5, k = 200 |
|---|---|---|---|---|---|
| A limpo − B | −6,4 [−23,3; +0,3], 85% | −3,3 [−16,5; +2,1], 80% | −4,2 [−14,0; +2,0], 80% | +0,2 [−17,5; +10,3], 45% | +0,5 [−10,5; +11,0], 35% |
| A (d) − A limpo | 0,0 [−2,6; +2,6], 45% | +0,1 [−2,5; +2,2], 40% | 0,0 [−2,9; +2,3], 40% | −2,8 [−5,5; +0,9], 80% | −2,5 [−6,3; +0,8], 75% |
| A (d) − B0 | −3,0 [−7,4; +0,2], 80% | −6,2 [−11,4; +2,6], 80% | −7,0 [−13,3; +1,7], 85% | **−16,7 [−21,6; −9,2], 100%** | **−18,4 [−24,9; −10,8], 100%** |

EER mediana por TTS de teste (camada 5, k = 200; 8 sorteios por TTS):

| Método | YourTTS | XTTS | ToucanTTS | Fish-Speech | F5-TTS |
|---|---|---|---|---|---|
| A limpo | 4,9% | 18,0% | 17,5% | 24,4% | 32,5% |
| B (só alvo) | 2,4% | 5,9% | 11,0% | 30,7% | 25,3% |

**Leituras (em relação ao pré-registro, seção 4.5):**

1. **Com TTS e locutores fora, a adaptação recupera muito menos que no Exp. 02.** O Exp. 02 dava 8,7%
   com 50 exemplos e 2,1% com 200 (camada 5, os mesmos TTS dos dois lados). Aqui, na mesma camada, a
   mediana fica em 22,1% e 19,7% (só alvo). Confirma-se a leitura prevista: **parte da separação "no
   próprio domínio" vinha de ter visto o gerador** e o processo de geração do dataset. (A comparação
   justa com o Exp. 02 é a forma B, que, como lá, só usa o alvo.)
2. **Mesmo assim, poucos exemplos ajudam de verdade.** Com a origem aumentada e 50 exemplos por classe
   na camada 5, a EER cai 16,7 pontos em relação ao detector, em todos os 20 sorteios (de 38,2% para
   ~21%). É a maior melhora no BRSpeech-DF de todo o experimento.
3. **A origem ajuda quando o alvo é pouco.** A forma A é melhor que a B na camada 12 em 80–85% dos
   sorteios, em todos os k. Na camada 5, só com k = 10 (mediana de −6,2 pontos, em 60% dos sorteios);
   com 50 ou mais exemplos, a origem não faz mais diferença. A leitura "a origem atrapalha" não se confirma em nenhum caso.
4. **O aumento (d) e os poucos exemplos quase não se somam.** Na camada 12, A (d) e A limpo empatam.
   Na camada 5, o (d) ajuda 2,2 a 2,8 pontos, em 75–95% dos sorteios, um efeito pequeno diante da
   variação entre sorteios.
5. **A escolha da camada pesa mais que o método.** A validação da origem escolhe a camada 12, mas a 5 é
   melhor no português em todos os k e métodos, por 6 a 17 pontos. Escolher a camada com os próprios exemplos do alvo
   (por validação cruzada nos k) é o passo natural, mas ele não foi pré-registrado e fica como trabalho
   futuro.
6. **O gerador é a fronteira.** F5-TTS e Fish-Speech continuam os mais difíceis quando ficam fora da
   adaptação (25–33%), e YourTTS o mais fácil (2–5%), como no controle de TTS deixado de fora do
   Exp. 02 (seção 6.3).
7. **A variação entre sorteios é enorme** (p10–p90 de ~5% a ~40%) e vem sobretudo dos locutores: no
   sorteio 17, todos os métodos ficam em ~58%. Com 10 locutores, a I2 dá a direção, mas não um número
   preciso.

## 6. Validação

- `e0_testar.py`: o minDCF coincide com uma varredura por força bruta em 200 casos sintéticos, com
  empates; dá 0 com classes separadas e 1 com pontuações iguais ou invertidas; a diferença pareada
  de uma pontuação com ela mesma é 0, com IC [0, 0].
- `e0_dividir.py`: nenhum locutor em mais de uma partição; as três origens em todas as partições.

## Reprodutibilidade

No PC, a partir de `experimentos/03-intervencao/scripts` (Python 3.13.12, numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0; scripts compatíveis com Python 3.8):

```bash
python -B e0_testar.py   # testes do minDCF e da diferença pareada
python -B e0_dividir.py  # listas/divisao_df21_eval.tsv e .json (semente 0)
python -B e1_linhas_de_base.py  # resultados/etapa1/ (~5 min; usa o features.npz da Fase 3 do Exp. 02)
python -B i2_poucos_exemplos.py # resultados/etapa3/ (~25 min; precisa de resultados/etapa2/i1.json e i1_treino_reps.npz)
```

Etapa 2 (servidor; o aumento e o probing rodam no ambiente isolado `~/matheus/venv_fase2`, com o
ffmpeg no PATH como na Fase 2; a extração roda no container, com o `f3_extrair.py` sem alteração):

```bash
cd ~/matheus/residencia/experimentos/03-intervencao/scripts
source ~/matheus/venv_fase2/bin/activate
python -B i1_aumentar.py --raiz-host-dados ~/matheus/dados \
    --saida ~/matheus/dados/exp03_i1 --raiz-container /dados/exp03_i1 --processos 16
python -B e2_listas_b1.py --pasta ~/matheus/dados/exp02_normalizar
deactivate
cd ~/matheus/residencia
S=experimentos/01-linha-de-base/scripts/em_docker.sh
CK="--checkpoint /ckpt/Best_LA_model_for_DF.pth --xlsr /ckpt/xlsr2_300m.pt"
E=../../02-diagnostico/scripts/f3_extrair.py
R=../../03-intervencao/resultados/etapa2
bash $S 0 python $E --lista /dados/exp03_i1/lista.tsv $CK --saida $R/features_i1.npz
for c in volume_padronizar controle; do
  bash $S 0 python $E --lista ../../03-intervencao/listas/etapa2/$c/lista.tsv $CK --saida $R/features_$c.npz
done
cd experimentos/03-intervencao/scripts && source ~/matheus/venv_fase2/bin/activate
python -B i1_probing.py --features ../../02-diagnostico/resultados/fase3/features.npz \
    --features-i1 ../resultados/etapa2/features_i1.npz \
    --features-b1 ../resultados/etapa2/features_volume_padronizar.npz \
    --features-controle ../resultados/etapa2/features_controle.npz
```

Etapa 4 (servidor; a extração dos parquet roda fora do container, com pyarrow, como no Exp. 01):

```bash
HF=https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF/resolve/6a2c47eb228511f7a6b3d67c9c0797d5be8a58e8
cd ~/matheus/dados/brspeech_df/hf
wget -c "$HF/bonafide/validation-00000-of-00001.parquet" -O bonafide-validation-0.parquet
for i in 0 1 2 3 4 5 6; do wget -c "$HF/spoof/validation-0000$i-of-00007.parquet" -O spoof-validation-$i.parquet; done
cd ~/matheus/residencia/experimentos/01-linha-de-base/scripts
python3 -B extrair_brspeech_hf.py --parquets ~/matheus/dados/brspeech_df/hf --divisao validation --saida ~/matheus/dados/brspeech_df/hf_val
cd ~/matheus/residencia
bash $S nenhuma python preparar_brspeech_hf.py --raiz /dados/brspeech_df/hf_val --subset dev --saida ../../03-intervencao/listas/brs_val
bash $S 0 python $E --lista ../../03-intervencao/listas/brs_val/lista.tsv $CK --saida ../../03-intervencao/resultados/etapa4/features_brs_val.npz
cd experimentos/03-intervencao/scripts && source ~/matheus/venv_fase2/bin/activate
python -B e4_brs_val.py --features-val ../resultados/etapa4/features_brs_val.npz --lista-val ../listas/brs_val
```

## Referências

1. H. Tak et al. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0
   and data augmentation*. Odyssey 2022. arXiv:2202.12233.
2. X. Wang et al. *ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at
   Scale*. ASVspoof 5 Workshop, 2024. arXiv:2408.08739. (Seção 3.1: DCF da Track 1.)
3. A. C. Ferro Filho et al. *BRSpeech-DF: A Deep Fake Synthetic Speech Dataset for Portuguese
   Zero-Shot TTS*. EMNLP 2025.
4. N. Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022 (In-the-Wild).
5. N. M. Müller et al. *Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really
   Learn?* ASVspoof 2021 Workshop. arXiv:2106.12914.
6. Experimento 02 deste repositório: `experimentos/02-diagnostico/README.md`.
