# Experimento 02 – Registro do processo

Este documento registra **como** o Experimento 02 foi feito: o que foi implementado, as decisões
e por quê, e os problemas encontrados. Os resultados estão no [README.md](README.md).

---

## 1. Ponto de partida (Semana 9)

O Experimento 01 terminou com a pergunta "de onde vem a perda?" sem resposta: os três cenários
diferem em vários fatores ao mesmo tempo. A decisão das Semanas 9 e 10 foi pausar a hipótese
semântica e diagnosticar a perda, em quatro fases. A primeira usa só as pontuações já
calculadas, sem GPU.

## 2. Preparação

1. **Submódulos.** No PC, os submódulos não estavam inicializados (as pastas de `third_party/`
   estavam vazias). Eles foram baixados com `git submodule update --init`, nos commits
   registrados, e o `verificar_submodulos.sh` confirmou que estão intactos. Ele foi rodado de
   novo no fim do trabalho, com o mesmo resultado.
2. **Metadados do 2021 DF.** O plano previa EER por codec, origem e vocoder a partir das chaves
   `DF-keys-stage-1`. Ao abrir o arquivo, as colunas de codec, origem e ataque estão todas
   preenchidas com "-": essas chaves trazem só o rótulo e a fase.
   - **Solução:** o repositório do AKCIT traz a chave **completa**
     (`data/metadata/trial_metadata.txt`, 13 colunas: locutor, trial, codec, origem, ataque,
     rótulo, trim, fase, vocoder, tarefa, equipe, gêneros e uma 13ª coluna com E/M/G/F nos
     trials do VCC 2020, provavelmente o idioma do alvo; **a confirmar**).
   - **Verificação antes de usar:** os 611.829 ids são iguais nas duas chaves. A fase `eval`
     (533.928 trials) e os rótulos são idênticos. A única diferença são os 18.576 trials da fase
     oculta, chamada `hidden_track` no Tak e `hidden` no AKCIT.
3. **Pontuações de outros detectores.** O repositório do AKCIT também publica pontuações de SLS e
   AASIST no 2021 DF e no BRSpeech-DF, em **todas as divisões** (treino, validação e teste).
   Elas foram usadas como controle e para medir a representatividade do teste.

## 3. Implementação

Tudo fica em `experimentos/02-diagnostico/`. A Fase 1 não precisa de `docker/`, `config/` nem
`listas/`, porque usa as listas e as pontuações do Experimento 01. As Fases 2 e 3 reutilizam a
imagem e o `inferir.py` do Experimento 01.

| Script | Função |
|---|---|
| `_diag.py` | Carrega os cenários num formato único, com bootstrap por grupos, Cllr/minCllr e calibração afim. Grava o `.meta.json` com o SHA-256 de cada entrada. **Importa** `_comum.py` e `metricas.py` do Experimento 01 (a mesma `compute_eer` do Tak), sem duplicar código. |
| `_figuras.py` | Estilo comum das figuras: genuíno em azul e falso em laranja, sempre. |
| `f1_global.py` | Distribuições, matriz cruzada e distância de Wasserstein (README 4.1). |
| `f1_df21_subgrupos.py` | Subgrupos do 2021 DF e hidden track (README 4.2 e 4.8). |
| `f1_calibracao.py` | Calibração × ordenação nos três detectores (README 4.3 e 4.9). |
| `f1_itw_locutores.py` | In-the-Wild por locutor: pendência da Semana 8 (README 4.4). |
| `f1_brs_pares.py` | BRSpeech-DF pareado, por locutor, e sorteio de locutores (README 4.4 a 4.6). |
| `f1_duracao.py` | Faixas de duração, atalho de duração, preenchimento e corte: pendência da Semana 8 (README 4.7). |
| `f2_perturbar.py` | Fase 2: gera os áudios perturbados, um fator por vez (não executado com dados reais). |
| `f2_analisar.py` | Fase 2: análise pareada das pontuações dos áudios perturbados (não executado com dados reais). |

## 4. Decisões metodológicas

| Decisão | Por quê |
|---|---|
| Reaproveitar as pontuações do Exp. 01, com o pré-processamento do Tak (repetição, corte de 4 s) | Mesmo modelo e mesmos números da linha de base; nenhuma inferência nova na Fase 1 |
| **Bootstrap por locutores** como IC principal | Áudios do mesmo locutor não são independentes. O bootstrap por áudio dá intervalos estreitos demais (no BRSpeech-DF, os 7.907 áudios vêm de só 10 locutores). O IC por áudio também foi gravado (`eer_ic_por_audio` no `global.json`), para comparar. |
| B = 1.000 réplicas (200 no 2021 DF; 100 nas faixas de duração do 2021 DF) | O 2021 DF tem 533 mil trials, e cada EER leva ~0,15 s. Com 100 a 200 réplicas, os percentis de 2,5% e 97,5% ainda são estáveis o bastante para o uso aqui. |
| **Matriz cruzada** (genuínos de X × falsos de Y) | É o teste mais direto, sem GPU, para "o genuíno se deslocou × o falso ficou parecido". É válida porque todas as pontuações saem do mesmo modelo e do mesmo pré-processamento. |
| Cllr/minCllr, além do limiar fixo | O erro num limiar fixo mistura ordenação e calibração. O minCllr isola a ordenação. É um framework da verificação de locutor que o ASVspoof também adota. |
| Calibração afim ajustada no 2021 DF | Simula o uso real, em que o limiar vem do domínio de treino. |
| Recalibração com k exemplos do alvo, 200 repetições | Responde se um pouco de dado do domínio corrige o problema, o que é relevante para a aplicação. |
| Taxas por locutor no limiar da EER global, e não a EER por locutor como medida principal | A EER de um locutor ignora o deslocamento entre locutores, que é justamente o fenômeno estudado. As duas foram gravadas. |
| Sensibilidade "sem os k piores locutores" | Mostra quanto a EER depende de poucos locutores. Está registrada como sensibilidade, não como correção. |
| Sorteio de 10 locutores entre os 62 do BRSpeech-DF, com as pontuações do AKCIT (zeros) | O teste oficial também usa zeros nas pontuações do AKCIT, então o pré-processamento é o mesmo nos dois lados. Responde se o teste é representativo. |
| Pareamento do BRSpeech-DF pelo mapa `ordenado` do Exp. 01 | É o mapa validado (correlação de 1,0000 com as pontuações do AKCIT). Os 6.587 falsos têm par. |
| Faixas de duração com limite em 4,04 s | Esse é o corte do modelo (64.600 amostras), que separa os áudios preenchidos dos cortados. |
| Fase 2: genuínos e falsos do 2021 DF de origem VCTK e sem codec | O ponto de partida é o caso mais "dentro do domínio" (EER de 1,27% e 0% de rejeição), de modo que qualquer deslocamento vem da perturbação. |
| Fase 2: RMS normalizado após os filtros de banda | Um fator por vez: o filtro de telefone também baixa o volume, e o volume é outro fator. |
| Fase 2: wav float32 nas saídas | Evita uma quantização extra, que seria mais um fator não controlado. |
| Fase 2: roda fora do container | Grava áudio na pasta de dados, que o container monta somente leitura (o mesmo motivo do `extrair_brspeech_hf.py`). Os codecs precisam do ffmpeg, que não está na imagem. |

## 5. Problemas e observações

1. **As chaves `DF-keys-stage-1` não têm metadados** (seção 2). Resolvido com a chave completa do
   AKCIT, conferida.
2. **Divergência no script de calibração.** A primeira execução falhou: numa amostra de 10 por
   classe, a regressão logística deu inclinação zero (divisão por zero no limiar). Agora essas
   amostras são contadas como "degeneradas" e puladas, e a contagem fica no JSON. O laço também
   ficou mais rápido: usa máscaras em vez de `setdiff1d` nos 533 mil trials.
3. **Pontuações do SLS e do AASIST no BRSpeech-DF parecem sem informação** (minCllr ≈ 1). Não
   foram usadas como evidência. Antes, é preciso conferir como foram geradas (taxa de 24 kHz?
   pré-processamento?) e compará-las com os números do artigo do BRSpeech-DF.
4. **Correção de uma frase do Experimento 01 (README do Exp. 01, seção 4.7).** Lá está escrito que,
   nos áudios curtos, as diferenças com e sem corte são "menores que 0,01" (ruído numérico). Na
   verdade, 861 dos 19.552 áudios curtos do In-the-Wild (4,4%) mudam mais de 0,01, e o máximo é
   0,237 (`26790.wav`). O percentil 99,9 é 0,064. Continua compatível com ruído numérico do tamanho
   de lote diferente (1 contra 14), mas a frase precisa ser corrigida. **Não alterei** o README do
   Experimento 01: fica para o pesquisador decidir.
5. **Recalibração com k = 200 não é melhor que com k = 10 no In-the-Wild** (12,8% contra 11,0%). O
   limiar com LLR = 0 da regressão logística minimiza a perda logística, não o erro balanceado, e
   pontuações saturadas pioram isso. A conclusão (recalibrar não resolve) não depende desse detalhe:
   nem o melhor limiar possível, a EER, fica perto do valor do domínio de treino.
6. **Ambiente local.** A Fase 1 rodou no PC (Windows, Python 3.13) e não no Docker, porque usa só
   numpy, scipy e scikit-learn sobre arquivos de texto. Os scripts foram escritos para Python 3.8
   (sem sintaxe nova), mas **ainda não foram rodados na imagem `residencia-exp01:pt181`**, e o
   scikit-learn não está no `requirements.txt` dela. Para rodar lá, é preciso acrescentá-lo.

## 6. Pendências

- Revisar a interpretação do README.
- Ouvir uma amostra dos genuínos dos 3 locutores "ruins" do BRSpeech-DF (5739, 13063 e 12626) e dos
  locutores com maior rejeição no In-the-Wild. É a checagem qualitativa mais barata da resposta 1.
- Confirmar as leituras da seção 4.3 do contexto: o silêncio como atalho (grupo do Müller) e o
  MLAAD. Conferir também a citação do Cllr (Brümmer e du Preez, 2006) e do plano de avaliação do
  ASVspoof 2021 (hidden track) antes de citá-los.
- Confirmar o significado da 13ª coluna da chave completa (provavelmente o idioma do alvo no VCC 2020).
- Investigar as pontuações do SLS e do AASIST no BRSpeech-DF (problema 3).
- Fase 2: copiar o `f2_perturbar.py` para o servidor (vai junto com o repositório), conferir se há
  ffmpeg no host, gerar as perturbações e rodar o `inferir.py` do Exp. 01 sobre a `lista.tsv` gerada.
- Fase 3: escrever a extração de representações por camada (precisa importar o modelo do Tak sem
  modificá-lo, com *hooks* do PyTorch).
