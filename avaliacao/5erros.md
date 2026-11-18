# Análise de 5 erros reais

Erros tirados da avaliação do **sistema final** (RAG + LoRA + limiar) no `notebooks/04_avaliacao.ipynb`, sobre os 37 casos de `casos_teste.json`. As respostas são as que o modelo deu de verdade (`avaliacao/resultados/sistema_final_rag_lora_limiar.jsonl`). A lista completa de falhas, com a causa provável detectada automaticamente, está em `avaliacao/erros_candidatos.csv`: 7 falhas no sistema final, sendo 3 de resposta incompleta ou errada, 2 de citação errada e 2 de recuperação errada.

---

### Erro 1: número de telefone errado numa pergunta de emergência
- **Caso de teste:** nº 27, "Para qual número ligo para chamar a Defesa Civil numa emergência?"
- **Esperado:** 199 (Bombeiros: 193), trecho T24.
- **Resposta do modelo:** "Para o número 40199, ou pelo WhatsApp (61) 2034-4611." (citou T03)
- **Causa:** a busca colocou o T03 (como *receber alertas*, que tem "Defesa Civil" e números de contato) em 1º lugar e o T24 (telefones de emergência) em 2º. O modelo usou o trecho mais parecido, que tem um número verdadeiro mas para outra finalidade (cadastro por SMS). É o erro mais grave do placar, porque numa emergência real a pessoa ligaria para o número errado.
- **Correção proposta:** reescrever o título e o texto do T24 com as palavras que o produtor usa ("ligar", "telefone", "número de emergência"); adicionar ao `treino_lora.jsonl` perguntas sobre telefone com T03 e T24 juntos no contexto, para o modelo aprender a diferença; adotar busca híbrida. **Mitigação já existente:** toda resposta sai com o aviso fixo "Em emergência, ligue 199 (Defesa Civil) ou 193 (Bombeiros)".

### Erro 2: instrução inventada sobre irrigação e geada
- **Caso de teste:** nº 9, "Posso irrigar a lavoura no dia anterior à geada para prevenir o dano?"
- **Esperado:** "Não: irrigar no dia anterior pode aumentar o dano, porque o solo úmido esfria mais" (T07).
- **Resposta do modelo:** "Não. Irrigue no dia seguinte ou durante a geada para evitar o maior dano." (citou T07)
- **Causa:** a busca e a citação estavam certas, e o "não" também. Mas o modelo de 1,5B resumiu demais: tirou a justificativa e **inventou** "irrigue no dia seguinte", que não está no trecho e não protege a lavoura. Os exemplos de treino do LoRA são curtos, o que incentiva respostas compactas que perdem o porquê. O mesmo padrão apareceu no teste rápido do notebook 02 ("ligar o pivô ao menos duas horas antes da geada", que também não está na base).
- **Correção proposta:** exemplos de treino que mantenham a justificativa ("porque...") e que repitam só o que está no trecho; checagem automática de fidelidade (cada frase da resposta precisa ter sobreposição com o trecho citado, senão é marcada para revisão); avaliar um modelo maior (Qwen2.5-3B, que ainda cabe na T4 em 4 bits).

### Erro 3: trecho certo não recuperado, e o modelo respondeu outra coisa
- **Caso de teste:** nº 16, "Como diminuir a perda de água na cova da bananeira?"
- **Esperado:** plástico de 0,40 m x 0,40 m no fundo da cova (T13).
- **Resposta do modelo:** "Manter a superfície coberta com plantas vivas ou palha, reduzir o revolvimento do solo, incorporar matéria orgânica..." (citou T16)
- **Causa:** **falha de recuperação**. O T13 não veio entre os 3 primeiros (vieram T14, T23 e T16). O encoder `all-MiniLM-L6-v2` foi treinado em inglês e não deu peso a "bananeira" e "cova". Sem o trecho certo, o modelo devia recusar, mas preferiu uma prática parecida de outro trecho. A resposta não é falsa, mas não responde à pergunta.
- **Correção proposta:** busca híbrida (BM25 + vetorial), porque "bananeira" é um termo exato que a busca por palavras pega; colocar "cova da bananeira" no título do T13; incluir no treino casos em que o contexto tem um trecho *parecido mas não o certo* e a resposta esperada é "não consta".

### Erro 4: recusa indevida por falha de sinônimo na busca
- **Caso de teste:** nº 23, "Como evitar erosão na lavoura quando chove demais?"
- **Esperado:** semeadura em nível e terraços (T20 ou T19).
- **Resposta do modelo:** "Não consta na base de conhecimento."
- **Causa:** a busca trouxe T06, T21 e T11 (irrigação contra geada, granizo, método do tato) e nenhum trecho sobre chuva excessiva. O texto da base diz "excesso de chuva" e a pergunta diz "chove demais", um sinônimo que o encoder em inglês não aproxima. Dado esse contexto, o modelo **recusou corretamente**: o erro é só da busca, e o comportamento é seguro. É a única recusa indevida do sistema final (3,7%).
- **Correção proposta:** testar de novo o encoder multilíngue (no notebook 01, ele também separa melhor as perguntas fora da base) ou a busca híbrida; incluir variações de vocabulário nos títulos dos trechos ("chuva forte", "chove muito", "enxurrada").

### Erro 5: resposta certa com a citação do trecho vizinho
- **Caso de teste:** nº 11, "Qual método de cálculo da evapotranspiração é bem aceito no semiárido?"
- **Esperado:** Hargreaves (T09).
- **Resposta do modelo:** "O método Hargreaves é bem aceito no semiárido do norte de Minas Gerais e do Nordeste." (citou T08)
- **Causa:** a resposta copia o T09, mas o id citado foi o T08, da mesma fonte (EPAMIG) e do mesmo tema. O modelo confunde ids de trechos vizinhos, e a validação atual só confere se o id citado está entre os recuperados (e o T08 estava). Erro parecido apareceu no notebook 01 (resposta do pivô citando T04 em vez de T06). O cidadão veria a fonte certa (EPAMIG), mas o link de explicabilidade apontaria o trecho errado.
- **Correção proposta:** depois da geração, escolher automaticamente como fonte o trecho recuperado com **maior sobreposição de texto** com a resposta e, se for diferente do id citado, corrigir e registrar; mais exemplos de treino com trechos da mesma fonte lado a lado.

---

## Erros de engenharia corrigidos nesta versão

Não são erros do modelo, mas ficam registrados:

1. **O LoRA não era usado na interface:** o notebook 03 carregava o Qwen em float16 sem o adaptador. Agora ele carrega o adaptador em 4 bits, pelo mesmo código avaliado (`src/agrofacil.py`).
2. **O JSON tinha resposta fixa:** o campo `resposta` era uma frase escrita no código. Agora é o LLM que gera o JSON, e o sistema valida o formato e a citação.
3. **O teste de recusa usava uma pergunta com resposta:** a "pergunta armadilha" era `casos_teste[0]`. Agora a demonstração usa perguntas "não consta" de verdade.
4. **O filtro léxico deixava passar perguntas fora da base:** "preço do adubo" tinha a palavra "adubo" e score acima do limiar. O filtro foi removido, o limiar é escolhido em dados de validação e o LLM foi treinado para recusar. Na avaliação, o "preço do adubo" foi recusado pelo modelo.
5. **Incompatibilidade de versões no treino:** com o `transformers` atual, `warmup_ratio` não existe mais e o modelo é carregado em bfloat16, que a T4 não treina em fp16. A correção foi usar `warmup_steps`, carregar as partes não quantizadas em float32 e desligar o fp16.
