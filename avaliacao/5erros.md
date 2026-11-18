# Análise de 5 erros reais

> ⚠️ **PREENCHER depois de rodar `notebooks/04_avaliacao.ipynb`.** A última célula gera `avaliacao/erros_candidatos.csv` com todas as falhas do C3 e do sistema final e a causa provável de cada uma. Escolham **5 erros de causas diferentes** e copiem aqui **o que o modelo respondeu de verdade**. Não usem exemplos inventados.

Causas que o notebook detecta automaticamente:

| Causa | O que significa | Correção típica |
| :--- | :--- | :--- |
| `recuperacao_errada` | o trecho certo não veio entre os 3 primeiros | reescrever o título/texto do trecho, dividir trechos longos, testar outro encoder ou busca híbrida (BM25 + vetorial) |
| `recusa_indevida_por_limiar` | pergunta válida barrada pelo limiar | recalibrar o limiar com mais perguntas de validação; reformular o trecho para aproximar do vocabulário do produtor |
| `recusa_indevida_pelo_modelo` | o trecho estava no contexto, mas o LLM disse "não consta" | mais exemplos positivos parecidos no `treino_lora.jsonl`; equilibrar respostas e recusas |
| `alucinacao_fora_da_base` | respondeu algo que não está na base | mais exemplos de recusa no treino, principalmente do próprio tema agrícola (preço, dose de produto); regra de recusa para perguntas sobre "hoje/amanhã/preço" |
| `resposta_incompleta_ou_errada` | respondeu, mas sem a informação do gabarito | exemplos de treino que copiem números e passos do trecho; aumentar `max_new_tokens` |
| `citacao_errada` | resposta certa, mas citou outro trecho | exemplos de treino com o trecho correto em posições variadas do contexto (já embaralhamos) |
| `json_invalido` | a saída não é JSON no formato | mais épocas de LoRA; decodificação restrita a JSON |

---

### Erro 1: [título curto]
- **Configuração:** (C3 ou Sistema final)
- **Caso de teste:** nº __, "pergunta"
- **Esperado (gabarito):** ...
- **Resposta do modelo:** "..."
- **Causa:** (causa provável + explicação: score top-1, trechos recuperados etc.)
- **Correção proposta:** ...

### Erro 2: [título curto]
- **Configuração:**
- **Caso de teste:**
- **Esperado (gabarito):**
- **Resposta do modelo:**
- **Causa:**
- **Correção proposta:**

### Erro 3: [título curto]
- **Configuração:**
- **Caso de teste:**
- **Esperado (gabarito):**
- **Resposta do modelo:**
- **Causa:**
- **Correção proposta:**

### Erro 4: [título curto]
- **Configuração:**
- **Caso de teste:**
- **Esperado (gabarito):**
- **Resposta do modelo:**
- **Causa:**
- **Correção proposta:**

### Erro 5: [título curto]
- **Configuração:**
- **Caso de teste:**
- **Esperado (gabarito):**
- **Resposta do modelo:**
- **Causa:**
- **Correção proposta:**

---

## Erros de engenharia já corrigidos nesta versão

Não contam como "erros do modelo", mas ficam registrados:

1. **O LoRA não era usado na interface:** o notebook 03 carregava o Qwen em float16 sem o adaptador. Agora ele carrega o adaptador em 4 bits, pelo mesmo código avaliado (`src/agrofacil.py`).
2. **O JSON tinha resposta fixa:** o campo `resposta` era uma frase escrita no código. Agora é o LLM que gera o JSON, e o sistema valida o formato e a citação.
3. **O teste de recusa usava uma pergunta com resposta:** a "pergunta armadilha" era `casos_teste[0]`. Agora a demonstração usa perguntas "não consta" de verdade.
4. **O filtro léxico deixava passar perguntas fora da base:** "preço do adubo" tinha a palavra "adubo" e score 0,597, acima do limiar de 0,40. O filtro foi removido, o limiar agora é escolhido em dados de validação e o LLM foi treinado para recusar.
5. **Encoder em inglês para texto em português:** `all-MiniLM-L6-v2` foi trocado por `paraphrase-multilingual-MiniLM-L12-v2`; a comparação está em `avaliacao/recall.csv`.
