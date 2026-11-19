# Plano do Experimento 03 — Intervenção (Semana 10, última entrega)

Registrado em 18/11/2026. **Gate final: 19/11/2026, entrega até 12h.** Decisão do Matheus: rodar a **linha completa**, em etapas por prioridade; cada etapa termina com resultado e README atualizados. Regras de tempo: ~23h, se as Etapas 2 e 3 não fecharam, parar e documentar; Etapa 4 só com folga; manhã só para revisão e Termo.

## Pergunta
O Exp. 02 concluiu que a informação para separar real e falso existe nas camadas do XLS-R, mas a regra de decisão aprendida no ASVspoof não transfere ("o gargalo são os dados da regra"). O Exp. 03 testa essa conclusão intervindo nela:
1. Se a regra for aprendida com dados do domínio de origem que cubram os fatores identificados (reverberação, silêncio, volume, canal), ela generaliza melhor **sem nenhum dado do alvo**?
2. Com poucos exemplos do alvo, quanto se recupera **numa avaliação justa** (locutores e TTS fora)?
3. Combinando tudo, qual é a melhor "receita" barata, e quanto custa no domínio de origem?

## Etapas
- **0 — Pré-registro e métricas:** leitura prevista no README antes de rodar; minDCF (parâmetros do ASVspoof 5, conferidos ou marcados "a confirmar"); divisão fixa por locutor do df21_eval da Fase 3 em treino/validação/teste; camadas 5, 12, 24 e embedding do AASIST fixadas a priori; escolha de camada e C só na validação da origem.
- **1 — Linhas de base (sem GPU):** B0 detector completo; B1 detector + volume_padronizar; B2 probe limpo. EER e minDCF com IC em itw, brs, df21_hidden e df21 teste.
- **2 — I1, aumento nas duas classes:** conjuntos (a) +reverberação RT60 0,2–1,0 s; (b) +silêncio/volume; (c) +canal (telefone, MP3 16 kbps); (d) tudo. Extração com f3_extrair.py; probe limpo × limpo+(a..d); Δ pareado e custo no df21 teste. Variante: (d) sobre teste normalizado (volume_padronizar). Não usar perturbações da Fase 2 como teste (circular).
- **3 — I2, poucos exemplos em português:** k = 10, 50, 200; locutores e TTS disjuntos (3 TTS adaptação, 2 teste, rodízio); 20 sorteios; partindo do probe limpo e do (d).
- **4 — Divisões de treino/validação do BRSpeech-DF (opcional):** checar tamanho antes; fonte de adaptação com teste oficial intocado e segundo teste mais representativo.
- **Fechamento:** tabela B0/B1/B2/I1/I1+B1/I2/I1+I2 (EER e minDCF); nulos registrados; PROCESSO; Trabalho futuro (retreino completo com aumento; ASVspoof 5, MLAAD, AASIST puro; emoção/velocidade com RAVDESS e VC).

## Princípios
- Modelo do Tak congelado.
- **Camada e hiperparâmetros escolhidos sem olhar o teste.** Na Fase 3 do Exp. 02, a "melhor camada" foi escolhida no próprio teste — otimista; declarar isso.
- Avaliar só em dados reais fora do domínio.
- IC por bootstrap de locutores e diferença pareada.

## Leitura prevista
- I1 melhora In-the-Wild e hidden track, mas não BRSpeech-DF → a perda em português não é coberta por esses fatores.
- I1 melhora também o BRSpeech-DF → reforço causal de "o gargalo são os dados da regra".
- Ablação: qual fator (reverberação, silêncio/volume, canal) responde pelo ganho em cada cenário.
- I2 com TTS fora continua baixo → separação geral; sobe muito → artefato do processo de geração do dataset.
- Custo no 2021 DF acima do IC → registrar como custo.

## Entregas da Semana 10 (decididas em 18/11)
1. Experimento 03 (acima).
2. Documento técnico unificado sobre detecção de deepfakes de fala, servindo de baseline para o grupo de pesquisa (markdown em `docs/` do repositório).
3. Documento de resumo geral consolidado da Residência (documento editável, linkado no Termo).
4. Termo da Semana 10.
