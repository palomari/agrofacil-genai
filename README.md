<div align="center">

# AgroFácil GenAI

### Assistente Técnico Agrícola Inteligente

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-1b3a2b?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Model Qwen2.5](https://img.shields.io/badge/Base_Model-Qwen2.5--1.5B--Instruct-2e8b57?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![LoRA 4-bit](https://img.shields.io/badge/Quantization-4--bit_NF4-1b3a2b?style=for-the-badge)](https://github.com/timdettmers/bitsandbytes)
[![Gradio UI](https://img.shields.io/badge/Frontend-Gradio_Custom_UI-2e8b57?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)

---

<table align="center" width="100%">
  <tr>
    <td align="center" style="background-color: #1b3a2b; color: #ffffff; padding: 12px; font-weight: bold; font-size: 16px;">
      ENGINNERING & GENAI PIPELINE • FIAP 2026
    </td>
  </tr>
</table>

</div>

---

## Integrantes da Equipe

<table width="100%">
  <thead>
    <tr style="background-color: #1b3a2b; color: #ffffff;">
      <th align="left">Nome do Integrante</th>
      <th align="center">Registro de Matrícula (RM)</th>
      <th align="center">Papel no Projeto</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff; color: #1b3a2b;">
      <td><b>Julia Ortiz</b></td>
      <td align="center"><code>RM550204</code></td>
      <td align="center">Engenharia de Prompt & RAG</td>
    </tr>
    <tr style="background-color: #f4fcf6; color: #1b3a2b;">
      <td><b>Julia Palomari</b></td>
      <td align="center"><code>RM551910</code></td>
      <td align="center">Frontend & Arquitetura Web</td>
    </tr>
    <tr style="background-color: #ffffff; color: #1b3a2b;">
      <td><b>Leticia Baptista</b></td>
      <td align="center"><code>RM550289</code></td>
      <td align="center">Dataset & Curadoria Técnica</td>
    </tr>
    <tr style="background-color: #f4fcf6; color: #1b3a2b;">
      <td><b>Vinicius Sobreira Borges</b></td>
      <td align="center"><code>RM97767</code></td>
      <td align="center">Fine-Tuning LoRA & Pipeline IA</td>
    </tr>
  </tbody>
</table>

---

## Visão Geral da Solução

O **AgroFácil GenAI** é um sistema de inteligência artificial generativa concebido para responder dúvidas técnicas relativas ao manejo agrícola, irrigação, nutrição de solo e manejo de pragas. 

A arquitetura combina a precisão da recuperação vetorial por **Retrieval-Augmented Generation (RAG)** com a eficiência de um **modelo especializado via LoRA (4-bit)** sobre a base `Qwen2.5-1.5B-Instruct`.

<table width="100%" style="border-collapse: collapse; margin: 15px 0;">
  <tr style="background-color: #e8f5ed; border: 1px solid #d4e5dc;">
    <td style="padding: 15px; color: #1b3a2b;">
      <h4 style="margin: 0 0 8px 0; color: #2e8b57;">PALETA DE CORES DA INTERFACE WEB</h4>
      <code>Base Principal: #1b3a2b (Verde Floresta)</code> | 
      <code>Destaque: #2e8b57 (Verde Esmeralda)</code><br>
      <code>Superfície: #e8f5ed (Menta Suave)</code> | 
      <code>Linha/Borda: #d4e5dc (Verde Menta)</code>
    </td>
  </tr>
</table>

### Principais Pilares de Projeto

* **Respostas Fundamentadas:** Utilização exclusiva de bases técnicas oficiais (Embrapa, FAO, INMET).
* **Recusa Segura (Safe Refusal):** Sistema dual-layer que identifica e bloqueia perguntas fora do domínio agrícola ou com baixa similaridade semântica.
* **Eficiência Computacional:** Quantização de 4-bit (NF4) via `bitsandbytes` permitindo a execução completa do pipeline em GPUs T4.

---

## Arquitetura do Pipeline

```text
                       ┌─────────────────────────┐
                       │   Entrada do Usuário    │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Filtro de Vocabulário  │
                       │     Léxico Agrícola     │
                       └────────────┬────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
         [ Fora do Escopo ]                 [ Termo Agrícola ]
                  │                                   │
                  ▼                                   ▼
      ┌───────────────────────┐           ┌───────────────────────┐
      │   Recusa Imediata     │           │  Recuperador Vetorial │
      │   "Não consta..."     │           │   all-MiniLM-L6-v2    │
      └───────────────────────┘           └───────────┬───────────┘
                                                      │
                                                      ▼
                                          ┌───────────────────────┐
                                          │   Score >= 0.40?      │
                                          └───────────┬───────────┘
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    │                                   │
                                 [ Não ]                             [ Sim ]
                                    │                                   │
                                    ▼                                   ▼
                        ┌───────────────────────┐           ┌───────────────────────┐
                        │   Recusa Segura       │           │ Prompt Contextualizado│
                        │   "Não consta..."     │           │ + Diretriz Técnica    │
                        └───────────────────────┘           └───────────┬───────────┘
                                                                        │
                                                                        ▼
                                                            ┌───────────────────────┐
                                                            │ Modelo Qwen2.5 1.5B   │
                                                            │ + Adaptador LoRA 4-bit│
                                                            └───────────┬───────────┘
                                                                        │
                                                                        ▼
                                                            ┌───────────────────────┐
                                                            │   Resposta Técnica    │
                                                            └───────────────────────┘
```

---

## Placar Comparativo de Desempenho

Resultados aferidos utilizando o dataset padronizado de teste (`avaliacao/casos_teste.json`):

| Métrica de Avaliação | Configuração 1: LLM Base | Configuração 2: RAG Simples | Configuração 3: RAG + LoRA (Proposta Final) |
| :--- | :---: | :---: | :---: |
| **Recall@1 (Recuperação)** | *Não Aplicável* | 0.85 | **1.00** |
| **Recall@3 (Recuperação)** | *Não Aplicável* | 1.00 | **1.00** |
| **Precisão Técnica (Domínio)** | 60.0% | 85.0% | **100.0%** |
| **Taxa de Alucinação (Fora do Domínio)** | Alta (~80.0%) | Média (~40.0%) | **0.0% (Safe Refusal)** |
| **Consumo Médio de VRAM** | ~6.2 GB | ~6.5 GB | **~3.8 GB (4-bit NF4)** |
| **Conformidade com Diretrizes** | Baixa | Média | **Alta (Embrapa / FAO / INMET)** |

---

## Matriz de Cobertura da Base de Dados

| Código | Fonte Institucional | Domínio de Conhecimento | Formato | Status de Validação |
| :---: | :--- | :--- | :---: | :---: |
| `DOC_001` | **Embrapa Solos e Irrigação** | Lâminas de Irrigação e Manejo de Milho | Texto Estruturado | Validado |
| `DOC_002` | **Manual de Pragas FAO** | Sintomas de Deficiência de Nitrogênio | Texto Estruturado | Validado |
| `DOC_003` | **Guia de Alertas INMET** | Mitigação de Geadas em Hortaliças | Texto Estruturado | Validado |

---

## Estrutura da Solução

```text
agrofacil-genai/
├── avaliacao/
│   ├── 5erros.md                  # Análise técnica dos 5 erros reais e correções
│   ├── casos_teste.json           # Conjunto de testes padronizado com gabarito
│   └── placar_comparativo.md      # Avaliação comparativa entre as 3 arquiteturas
├── base/
│   ├── documentos_tecnicos.txt    # Base de conhecimento fragmentada e indexada
│   └── tabela_base.md             # Tabela de metadados, licenças e fontes
├── notebooks/
│   ├── 01_rag_pipeline.ipynb      # Construção do RAG vetorial e cálculo de Recall
│   ├── 02_especializacao.ipynb    # Fine-tuning 4-bit LoRA do modelo Qwen2.5
│   └── 03_interface_web.ipynb     # Interface Gradio e lógica de recusa segura
└── README.md                      # Documentação técnica do projeto
```

---

## Instruções de Execução

### Pré-requisitos
* Python 3.10 ou superior
* Ambiente de execução acelerado por GPU (Recomendado: Google Colab - GPU Tesla T4)

### Instalação de Dependências
```bash
pip install -q transformers datasets accelerate peft bitsandbytes sentence-transformers gradio trl
```

### Execução dos Notebooks
1. **Pipeline RAG:** Abra `notebooks/01_rag_pipeline.ipynb` para executar o carregamento da base vetorial e calcular as métricas de `Recall@1` e `Recall@3`.
2. **Treinamento LoRA:** Execute `notebooks/02_especializacao.ipynb` para treinar os adaptadores de atenção e gerar o diretório `./agrofacil_adapter`.
3. **Interface Web:** Rode `notebooks/03_interface_web.ipynb` para iniciar a interface Gradio interativa.

---

## Licenciamento e Fontes

* **Embrapa Solos e Irrigação:** Dados Abertos Governamentais (Uso Livre).
* **FAO (Food and Agriculture Organization):** Conteúdo Educacional e Acadêmico.
* **INMET (Instituto Nacional de Meteorologia):** Dados Abertos de Domínio Público.
