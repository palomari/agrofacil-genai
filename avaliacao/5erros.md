# Análise de 5 Erros Reais e Soluções Aplicadas

Este documento detalha cinco falhas técnicas identificadas durante a fase de testes do assistente **AgroFácil GenAI**, incluindo a causa raiz e a correção implementada.

---

### Erro 1: Alucinação em Perguntas Fora do Domínio
* **Descrição:** Perante perguntas não agrícolas (ex: *"Quantos voos saem de Viracopos?"*), o modelo gerava respostas inventadas com base no seu conhecimento pré-treinado.
* **Causa Raiz:** O modelo base `Qwen2.5` recebia a pergunta diretamente na inferência sem validação prévia de escopo ou contexto.
* **Solução:** Implementação de um filtro léxico de termos agrícolas combinando validação vetorial com limiar de confiança na função `_gerar_real`.

---

### Erro 2: Recuperação Forçada de Documentos Irrelevantes
* **Descrição:** Consultas sobre política (ex: *"Quantos deputados federais o estado de São Paulo possui?"*) resultavam em respostas misturadas sobre a lagarta-do-cartucho.
* **Causa Raiz:** O mecanismo de busca por similaridade vetorial retornava sempre o documento com maior produto escalar, mesmo que o score fosse irrelevante para a pergunta.
* **Solução:** Ajuste do limiar de similaridade para $\ge 0.40$, acionando a resposta padrão `"Não consta na base de conhecimento."` quando o score é insuficiente.

---

### Erro 3: Estouro de Memória de GPU (CUDA Out of Memory)
* **Descrição:** O script de execução da interface web falhava com o erro `OutOfMemoryError: CUDA out of memory`.
* **Causa Raiz:** Tentar alocar o modelo principal `Qwen2.5` e o modelo de embeddings `SentenceTransformer` em simultâneo na VRAM da GPU T4.
* **Solução:** Forçar a execução do codificador leve na CPU (`device='cpu'`), reservando a memória gráfica exclusivamente para o modelo gerador.

---

### Erro 4: Repetição do Prompt na Resposta Gerada
* **Descrição:** A resposta devolvida na interface do Gradio exibia o prompt completo de instrução antes da resposta técnica final.
* **Causa Raiz:** A função `tokenizer.decode()` descodifica toda a sequência de tokens gerada, incluindo os tokens do prompt de entrada.
* **Solução:** Adicionada a rotina de limpeza de prefixo `if prompt_contextualizado in resp: resp = resp[len(prompt_contextualizado):].strip()`.

---

### Erro 5: Erro de Indentação na Definição de Funções (IndentationError)
* **Descrição:** O Notebook 3 falhava ao iniciar com a mensagem `IndentationError: expected an indented block`.
* **Causa Raiz:** Inclusão de instruções de importação (`from sentence_transformers import...`) soltas no meio do bloco da função `_gerar_real`.
* **Solução:** Reestruturação do código isolando o carregamento da base e do encoder no âmbito global antes da definição do manipulador do Gradio.