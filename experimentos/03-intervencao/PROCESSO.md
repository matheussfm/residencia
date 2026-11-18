# Experimento 03 – Registro do processo

Este documento registra **como** o Experimento 03 foi feito: o que foi implementado, as decisões e
por quê, e os problemas encontrados. O pré-registro e os resultados estão no [README.md](README.md).
O plano original está em [plano-experimento-03-intervencao.md](plano-experimento-03-intervencao.md).

---

## 1. Ponto de partida

O Exp. 02 terminou com a conclusão "o gargalo são os dados com que a regra de decisão foi
treinada" e com três intervenções sugeridas (README do Exp. 02, seção 9). A normalização de volume
e silêncio já tinha sido testada lá. O Exp. 03 testa as outras duas, o aumento de dados direcionado
(I1) e a adaptação com poucos exemplos (I2), com um protocolo mais rigoroso que o da Fase 3.

A linha foi organizada em etapas por prioridade. Cada etapa termina com resultados salvos (com
`.meta.json`), o README atualizado e uma pausa para revisão antes da seguinte.

## 2. Implementação

| Script | Função |
|---|---|
| `_interv.py` | Módulo comum. **Importa** o `_diag.py` do Exp. 02 (cenários, EER, bootstrap, `.meta.json`), que por sua vez importa o `_comum.py` e o `metricas.py` do Exp. 01. Acrescenta o minDCF, EER e minDCF com IC por locutor, a diferença pareada das duas métricas, a leitura da divisão e as representações e a grade de C do pré-registro. |
| `e0_testar.py` | Testes do minDCF (contra força bruta, com empates, e casos-limite) e da diferença pareada, com dados sintéticos. |
| `e0_dividir.py` | Divisão fixa, por locutor, da amostra df21_eval da Fase 3 em treino, validação e teste. |
| `i1_aumentar.py` | Etapa 2: cópias da partição de treino nas variantes controle, (a), (b), (c) e (d), com semente por (áudio, variante) e os parâmetros sorteados gravados em `condicoes.tsv`. **Importa** as perturbações do `f2_perturbar.py` e o leitor do `f2_silencio_real.py`. Roda fora do container. Testado com áudio sintético a 16 e 24 kHz. |
| `e2_listas_b1.py` | Etapa 2: filtra as listas da Fase 2c (volume_padronizar e controle) para os ids da amostra da Fase 3, com os ids da amostra. Testado com listas falsas. |
| `i1_probing.py` | Etapa 2: probes limpo, controle e (a)–(d) em cada representação, escolha na validação limpa, métodos combinados com B1, diferenças pareadas (com a representação escolhida e em cada representação fixa) e conferências. Grava as representações das cópias do treino para a I2. Testado com features sintéticas de dimensão 8 e os ids reais. |
| `i2_poucos_exemplos.py` | Etapa 3: 20 sorteios fixos (rodízio das 10 combinações de 3 TTS de adaptação, locutores 5/5), k exemplos por classe, formas A (origem limpa ou (d) + alvo, com pesos) e B (só alvo), referências sem alvo (B0, B2, I1_d) nos mesmos subconjuntos de teste e EER por TTS de teste. Testado com 2–3 sorteios (sem a origem (d) e com entradas falsas da Etapa 2). O teste rápido usou dados reais, mas nenhuma decisão foi tomada a partir dele. |
| `e4_brs_val.py` | Etapa 4: a validação do BRSpeech-DF como segundo teste (B0, B2 e I1 (d), com IC por locutor) e como fonte de adaptação da I2, com o teste na amostra do teste oficial, completa nos locutores. Confere o pareamento dos falsos contra as pontuações do AKCIT. Testado com uma validação falsa (ids e metadata reais, features aleatórias). Roda no servidor. |
| `extrair_brspeech_hf.py` e `preparar_brspeech_hf.py` (Exp. 01) | **Alterados de forma compatível:** ganharam `--divisao` e `--subset`, com padrão `test`, que reproduz o comportamento anterior. Na contagem esperada, os números fixos do teste (1.320 e 6.587) foram trocados pelo tamanho da divisão nos metadados do AKCIT (que, no teste, são os mesmos). |
| `e1_linhas_de_base.py` | Etapa 1: B0, controle e B1 (pontuações dos Exp. 01 e 02) e B2 (probe limpo, escolhido na validação). EER e minDCF com IC e diferenças pareadas. Grava as pontuações do B2 para as etapas seguintes. Teste rápido com B = 20 numa pasta temporária antes da execução real. |

## 3. Decisões metodológicas

| Decisão | Por quê |
|---|---|
| minDCF com C_miss = 1, C_fa = 10, π_spf = 0,05 (β ≈ 1,90) | Parâmetros da Track 1 do ASVspoof 5, conferidos no artigo (Wang et al., 2024, seção 3.1). A métrica principal continua a EER, por comparabilidade com os Exp. 01 e 02. |
| minDCF calculado sobre todos os limiares observados e +∞ | É o limiar "oráculo" do artigo. Com o limiar −∞ (aceitar tudo), DCF = 1, então o minDCF fica em [0, 1]. Conferido contra força bruta. |
| Divisão dentro de cada origem, com metas de 60/20/20 dos áudios nas duas classes | Muitos locutores têm só uma classe (14 dos 62 do VCTK só têm genuínos). Cortar uma lista sorteada de locutores deixaria partições desequilibradas. A alocação pelo maior déficit relativo, numa ordem sorteada (semente 0), equilibra as duas classes. |
| Treino do probe só na partição de treino; a validação serve só para escolher | Mantém a validação limpa também para a escolha da I1. |
| Representações (5, 12, 24, AASIST) e grade de C fixadas antes de rodar | Na Fase 3 do Exp. 02, a melhor camada foi escolhida no teste. Aqui a escolha é feita só na validação da origem, e as quatro representações são reportadas para separar o efeito da intervenção do efeito da escolha. |
| Hidden track avaliada completa e só com os locutores de teste | Os 46 locutores da hidden track estão todos no df21_eval; 27 caíram no treino. As duas formas, juntas, medem o efeito de ter visto o locutor. |
| B1 comparado contra a condição controle da Fase 2c | A controle passa pelo mesmo caminho de leitura e gravação que a normalizada; contra o Exp. 01, a comparação misturaria a decodificação (Exp. 02, PROCESSO, problema 14). |
| I2, forma A (origem + k do alvo, com peso total igual) como principal e forma B (só alvo) como referência | Uma regressão logística não continua um treino anterior. "Partir do probe limpo ou do (d)" só tem sentido se os dados da origem entram no treino (forma A). Comparar A com B diz se a origem ajuda ou atrapalha. Sem o peso, 20 exemplos do alvo ficariam diluídos entre ~1.600 da origem. |
| I1 com uma variante controle (cópia sem aumento), acrescentada antes de rodar | O probe aumentado é treinado com o dobro de exemplos, e as cópias passam por outro caminho de leitura (soundfile/ffmpeg, wav float32) que os áudios originais. Sem o controle, um ganho poderia vir só disso. |
| I1 com a escolha na validação limpa da origem | É a mesma regra do B2. Uma validação aumentada favoreceria o aumento por construção. |
| I1: (d) com cada fator a probabilidade 1/2, na ordem sala → canal → silêncio/volume | Imita a cadeia real: a sala antes do microfone, o canal na transmissão, o corte e o ganho na edição. Com 1/2 por fator, cerca de 1/8 das cópias teria todos os fatores e 1/8 nenhum; o "nenhum" é trocado por um fator sorteado. |
| I1 + B1 comparado contra o mesmo probe nas features da condição controle | Mesmo motivo da linha de B1: a normalização e a controle passam pelo mesmo caminho de E/S. |
| I2: rodízio sistemático das combinações de TTS (cada uma em 2 dos 20 sorteios) | Com sorteio livre, algumas combinações poderiam não aparecer. Assim, cada TTS fica fora da adaptação em 8 dos 20 sorteios. |
| I2: incerteza pelos sorteios (mediana, p10 e p90), não por bootstrap | A variação entre sorteios de locutores e TTS já é a incerteza relevante. Com 5 locutores de teste por sorteio, um bootstrap por locutor dentro de cada sorteio seria muito instável. |
| I2, forma A: pesos em que cada célula domínio × classe soma 1/4, com total igual a n | Mantém a escala do C igual à do probe com `class_weight='balanced'`, para que o C escolhido na validação da origem possa ser reaproveitado. |
| Etapa 4 só com a validação do BRSpeech-DF | A divisão de treino tem 147,5 GB no Hugging Face (inviável no prazo). A de validação tem 3,5 GB (baixada em ~2 min; os oito tamanhos conferidos com a API do Hugging Face), 1.161 genuínos e 5.795 falsos, e 10 locutores disjuntos dos do teste e do treino (conferido na metadata do AKCIT). |
| Etapa 4: alterar os scripts do Exp. 01 em vez de copiá-los | Os dois tinham "test" fixo. A opção nova, com padrão `test`, reaproveita a extração e o pareamento dos falsos já validados, sem duplicar código. |
| I2 com TTS e locutores disjuntos entre adaptação e teste | O controle de TTS deixado de fora do Exp. 02 (seção 6.3) mostrou que a separação depende em parte de ter visto o gerador. |

## 4. Problemas e observações

1. **Vazamento de locutores na Fase 3 do Exp. 02.** O probe de transferência foi treinado com metade
   dos locutores do df21_eval e testado na hidden track completa, cujos 46 locutores estão todos no
   df21_eval. Parte dos locutores testados tinha sido vista no treino. O Exp. 03 mede esse efeito
   (hidden track completa × só locutores de teste). O README do Exp. 02 não foi alterado.
2. **Escolha da camada no teste, na Fase 3 do Exp. 02.** Os números "melhor camada" de lá são
   otimistas. Está declarado no pré-registro (README, seção 4.3).
3. **Disco do PC.** O C: tem cerca de 0,3 GB livres. As features novas da I1 e da variante B1 não
   cabem no PC. O probing que as usa roda no servidor, no ambiente isolado `~/matheus/venv_fase2`
   (com scikit-learn), como no problema 13 do Exp. 02. Só os JSON vêm para o repositório.
4. **Parâmetros do minDCF.** O artigo do ASVspoof 5 remete os detalhes a um plano de avaliação
   (fase 2), que não foi aberto. Os valores usados são os que o artigo dá explicitamente.
5. **Partições pequenas no VCC 2018.** A validação e o teste têm 3 locutores do VCC 2018 cada. O
   df21 teste tem 20 locutores ao todo; os IC por locutor vão ser largos.

6. **O df21 teste favorece os probes.** Eles são treinados no df21 eval (mesmos corpora e ataques,
   outros locutores); o detector, no 2019 LA. Por isso o custo das intervenções no domínio de origem
   é medido contra o B2, e não contra o B0 (README, seção 5.1, leitura 5).
7. **O minDCF satura no BRSpeech-DF** (README, seção 5.1, leitura 6). A EER continua sendo a métrica
   principal; o minDCF é reportado em todos os casos.

8. **ffmpeg fora do PATH no ambiente isolado.** O `i1_aumentar.py` parou antes de gerar qualquer
   arquivo, porque as variantes (c) e (d) precisam do ffmpeg (MP3 a 16 kbps) e o `which ffmpeg` não o
   achava. O binário do `imageio-ffmpeg` (v9.0.2, com `libmp3lame`) já estava instalado no venv e foi
   ligado ao PATH com um link simbólico em `~/matheus/venv_fase2/bin/ffmpeg`. Nada foi instalado fora
   do venv.
9. **Leitura dos .flac pelo ffmpeg no aumento.** 790 dos 1.594 trials do treino (50%) foram lidos pelo
   ffmpeg, porque a libsndfile do venv recusou o arquivo (o mesmo caso dos problemas 8 e 14 do Exp. 02).
   As cópias aumentadas passam, portanto, por um caminho de leitura diferente do dos áudios originais
   (librosa, no container). A variante controle existe para isso: a EER do detector nas cópias controle
   (2,76%) fica perto da dos originais (3,39%).
10. **Conferências da Etapa 2 no servidor.** As pontuações do detector gravadas na extração dos áudios
    normalizados reproduzem as da Fase 2c (correlação de 0,99999999996; diferença máxima de 0,003). O
    probe limpo, retreinado no servidor, reproduz o B2 da Etapa 1, treinado no PC (diferença máxima de
    0,0001 na camada 12 e de até 0,015 nas outras, entre versões diferentes do scikit-learn).

## 5. Pendências

- Etapas 2 a 4 e fechamento (ver o plano).
