# Placar comparativo: AgroFácil GenAI

> ⚠️ Este arquivo é **gerado automaticamente** pelo `notebooks/04_avaliacao.ipynb`.
> Rode os notebooks 01 → 02 → 04 no Colab (GPU T4) e faça o commit deste arquivo junto com `placar.csv`, `placar.png`, `detalhe_por_caso.csv`, `erros_candidatos.csv` e a pasta `resultados/`.

As configurações avaliadas sobre os 37 casos de `casos_teste.json` são:

| Configuração | Descrição |
| :--- | :--- |
| C1 · LLM base sem contexto | `Qwen2.5-1.5B-Instruct` (4 bits) sem acesso à base |
| C2 · RAG simples | busca dos 3 trechos mais parecidos + modelo base |
| C3 · RAG + LoRA | mesma busca + adaptador LoRA treinado no notebook 02 |
| Sistema final | C3 + limiar de recusa escolhido no notebook 01 |
