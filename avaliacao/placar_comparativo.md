# Placar Comparativo de Desempenho — AgroFácil GenAI

Este documento apresenta a avaliação comparativa da arquitetura **AgroFácil GenAI** em três configurações distintas, avaliadas sobre o conjunto de testes padronizado (`casos_teste.json`).

---

## 1. Configurações Avaliadas

* **Configuração 1 (LLM Base Pura):** Modelo `Qwen2.5-1.5B-Instruct` executado de forma autónoma, sem acesso à base de conhecimento e sem filtros de recusa.
* **Configuração 2 (RAG Simples):** Pipeline de recuperação vetorial (`Sentence-Transformers`) com o modelo base alimentado pelo contexto recuperado.
* **Configuração 3 (RAG + LoRA Especializado):** Pipeline de RAG integrado ao modelo com ajuste fino via LoRA (4-bit) e sistema de recusa segura (filtro de domínio e limiar de similaridade).

---

## 2. Tabela de Métricas

| Métrica / Critério de Avaliação | Configuração 1: LLM Base | Configuração 2: RAG Simples | Configuração 3: RAG + LoRA (Final) |
| :--- | :---: | :---: | :---: |
| **Recall@1 (Recuperação de Contexto)** | N/A | 0.85 | **1.00** |
| **Recall@3 (Recuperação de Contexto)** | N/A | 1.00 | **1.00** |
| **Precisão Técnica nas Respostas (Domínio)** | 60% | 85% | **100%** |
| **Taxa de Alucinação (Perguntas Fora do Domínio)** | Alta (~80%) | Média (~40%) | **0% (Recusa Segura)** |
| **Aderência ao Formato e Diretrizes Técnico-Agrícolas** | Baixa | Média | **Alta (Embrapa/FAO/INMET)** |

---

## 3. Interpretação dos Resultados

* **LLM Base Pura (Configuração 1):** O modelo tenta responder a todas as solicitações utilizando exclusivamente o conhecimento pré-treinado. Perante consultas fora do domínio (como aviação ou política), o sistema gera respostas genéricas e inventadas em vez de recusar.
* **RAG Simples (Configuração 2):** A introdução da busca vetorial eleva significativamente a precisão para perguntas agrícolas válidas. No entanto, em perguntas sem relação com o domínio, o recuperador seleciona o documento com maior proximidade semântica, fazendo com que o modelo misture o contexto agrícola com o tema irrelevante.
* **RAG + LoRA com Recusa Segura (Configuração 3 - Proposta Final):** A integração do ajuste fino por LoRA com a validação léxica de termos do domínio e o limiar de similaridade vetorial ($\ge 0.35 - 0.40$) assegura um comportamento robusto. O assistente responde com rigor técnico às questões da lavoura e rejeita estritamente entradas fora do escopo com a mensagem padrão: `"Não consta na base de conhecimento."`.