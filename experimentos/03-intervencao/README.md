# Experimento 03 – Intervenção na regra de decisão

Testa, intervindo nela, a conclusão do Experimento 02: a informação para separar fala real e
sintética existe nas camadas do XLS-R, mas a regra de decisão aprendida no ASVspoof não transfere
("o gargalo são os dados da regra"). O detector é sempre o de Tak et al. (2022), XLS-R 300M +
AASIST, com o checkpoint publicado e **congelado**: nada do modelo é treinado. Só se treinam
classificadores lineares (probes) sobre as representações dele.

> Status: Etapas 0 (pré-registro, métricas e divisão) e 1 (linhas de base) concluídas. Etapas 2 a 4
> ainda não rodadas. As seções 1 a 4 foram escritas **antes** de qualquer resultado do Exp. 03.

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
  os métodos, de modo que as comparações são pareadas.
- Cada sorteio escolhe 3 dos 5 TTS para a adaptação (os outros 2 ficam para o teste) e divide os
  10 locutores em 5 para a adaptação e 5 para o teste. A adaptação usa só genuínos dos locutores de
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
