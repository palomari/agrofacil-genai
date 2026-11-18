"""
AgroFácil GenAI - núcleo do pipeline (busca, geração fundamentada, recusa e avaliação).

Usado pelos notebooks 01 a 04, pela interface Gradio e pela API (api/app.py),
para que todas as partes rodem exatamente a mesma lógica.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_BASE = RAIZ / "base" / "trechos.jsonl"
CAMINHO_TREINO = RAIZ / "base" / "treino_lora.jsonl"
CAMINHO_TESTES = RAIZ / "avaliacao" / "casos_teste.json"
CAMINHO_CONFIG = RAIZ / "avaliacao" / "config_rag.json"
CAMINHO_ADAPTER = RAIZ / "modelos" / "agrofacil_adapter"

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
# Modelo de embeddings multilíngue (a base e as perguntas estão em português).
ENCODER_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K_PADRAO = 3

NAO_CONSTA = "Não consta na base de conhecimento."
AVISO_SEGURANCA = (
    "Orientação informativa baseada em fontes oficiais. Não substitui a Defesa Civil "
    "nem a assistência técnica. Em emergência, ligue 199 (Defesa Civil) ou 193 (Bombeiros)."
)

SYSTEM_RAG = (
    "Você é o AgroFácil, assistente de segurança climática para pequenos produtores rurais. "
    "Responda SOMENTE com base nos trechos fornecidos, em português simples e em no máximo 3 frases. "
    "Responda apenas com um objeto JSON válido, sem nenhum texto fora dele, no formato: "
    '{"resposta": "<texto>", "fonte_id": "<id do trecho usado, ex.: T05>", "status": "ok"}. '
    "Se os trechos não trazem a informação pedida, responda exatamente: "
    '{"resposta": "Não consta na base de conhecimento.", "fonte_id": null, "status": "nao_consta"}.'
)

SYSTEM_SEM_CONTEXTO = (
    "Você é o AgroFácil, assistente de segurança climática para pequenos produtores rurais. "
    "Responda em português simples e em no máximo 3 frases. "
    "Responda apenas com um objeto JSON válido, sem nenhum texto fora dele, no formato: "
    '{"resposta": "<texto>", "fonte_id": null, "status": "ok"}. '
    "Se você não souber a resposta com segurança, responda exatamente: "
    '{"resposta": "Não consta na base de conhecimento.", "fonte_id": null, "status": "nao_consta"}.'
)


# ---------------------------------------------------------------- dados
def carregar_jsonl(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def carregar_base(caminho: Path = CAMINHO_BASE) -> list[dict]:
    return carregar_jsonl(caminho)


def carregar_testes(caminho: Path = CAMINHO_TESTES) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def carregar_config() -> dict:
    """Limiar e encoder escolhidos no notebook 01 (com valores padrão se ainda não rodou)."""
    if CAMINHO_CONFIG.exists():
        with open(CAMINHO_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {"encoder": ENCODER_PADRAO, "limiar": 0.5, "k": K_PADRAO}


# ---------------------------------------------------------------- busca
class Buscador:
    """Busca vetorial por similaridade de cosseno sobre os trechos da base."""

    def __init__(self, trechos: list[dict], modelo: str = ENCODER_PADRAO, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.trechos = trechos
        self.modelo = modelo
        self.encoder = SentenceTransformer(modelo, device=device)
        textos = [f"{t['titulo']}. {t['texto']}" for t in trechos]
        self.embeddings = self.encoder.encode(textos, normalize_embeddings=True)

    def buscar(self, pergunta: str, k: int = K_PADRAO) -> list[dict]:
        v = self.encoder.encode([pergunta], normalize_embeddings=True)[0]
        scores = self.embeddings @ v
        ordem = np.argsort(-scores)[:k]
        return [{**self.trechos[i], "score": float(scores[i])} for i in ordem]


def montar_contexto(recuperados: list[dict]) -> str:
    return "\n\n".join(f"[{t['id']}] (Fonte: {t['fonte']}) {t['texto']}" for t in recuperados)


def mensagens_rag(pergunta: str, recuperados: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Trechos:\n{montar_contexto(recuperados)}\n\nPergunta: {pergunta}"},
    ]


def mensagens_sem_contexto(pergunta: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_SEM_CONTEXTO},
        {"role": "user", "content": f"Pergunta: {pergunta}"},
    ]


# ---------------------------------------------------------------- modelo
def carregar_modelo(adapter: str | Path | None = None, quant_4bit: bool = True):
    """Carrega o Qwen2.5-1.5B-Instruct (4 bits NF4 se houver GPU) e, opcionalmente, o adaptador LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    kwargs = {"device_map": "auto"}
    if quant_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODELO_BASE, **kwargs)
    if adapter:
        model = aplicar_adapter(model, adapter)
    model.eval()
    return model, tokenizer


def aplicar_adapter(model, adapter: str | Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter)).eval()


def gerar(model, tokenizer, mensagens: list[dict], max_new_tokens: int = 200) -> str:
    """Geração determinística (greedy) para que a avaliação seja reprodutível."""
    import torch

    texto = tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    entradas = tokenizer(texto, return_tensors="pt").to(model.device)
    with torch.no_grad():
        saida = model.generate(
            **entradas,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    novos = saida[0][entradas["input_ids"].shape[1]:]
    return tokenizer.decode(novos, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- JSON
def extrair_json(texto: str) -> dict | None:
    """Tenta ler o JSON gerado; aceita o objeto cercado por ```json ... ``` ou por texto."""
    if not texto:
        return None
    candidatos = [texto.strip()]
    bloco = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, flags=re.S)
    if bloco:
        candidatos.append(bloco.group(1))
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini != -1 and fim > ini:
        candidatos.append(texto[ini:fim + 1])
    for c in candidatos:
        try:
            dados = json.loads(c)
            if isinstance(dados, dict):
                return dados
        except json.JSONDecodeError:
            continue
    return None


def json_no_formato(dados: dict | None) -> bool:
    return (
        isinstance(dados, dict)
        and isinstance(dados.get("resposta"), str)
        and dados.get("status") in {"ok", "nao_consta"}
        and "fonte_id" in dados
    )


# ---------------------------------------------------------------- pipeline
def responder(
    pergunta: str,
    model,
    tokenizer,
    buscador: Buscador | None = None,
    modo: str = "rag",
    limiar: float | None = None,
    k: int = K_PADRAO,
) -> dict:
    """
    Pipeline completo. modo="rag" usa a base; modo="sem_contexto" pergunta direto ao modelo.
    Se `limiar` for informado, perguntas cujo trecho mais parecido tem score menor que o
    limiar são recusadas antes de chamar o modelo (primeira camada de recusa).
    """
    t0 = time.time()
    recuperados: list[dict] = []
    if modo == "rag":
        recuperados = buscador.buscar(pergunta, k=k)
    top1 = recuperados[0]["score"] if recuperados else None
    base_resultado = {
        "pergunta": pergunta,
        "modo": modo,
        "limiar": limiar,
        "score_top1": top1,
        "trechos_recuperados": [{"id": t["id"], "score": round(t["score"], 4)} for t in recuperados],
        "aviso": AVISO_SEGURANCA,
    }

    if modo == "rag" and limiar is not None and top1 < limiar:
        return {
            **base_resultado,
            "resposta": NAO_CONSTA, "status": "nao_consta", "fonte_id": None,
            "fonte": None, "url": None, "json_valido": True, "citacao_valida": None,
            "origem": "limiar", "bruto": None, "latencia_s": round(time.time() - t0, 3),
        }

    msgs = mensagens_rag(pergunta, recuperados) if modo == "rag" else mensagens_sem_contexto(pergunta)
    bruto = gerar(model, tokenizer, msgs)
    dados = extrair_json(bruto)
    valido = json_no_formato(dados)

    if not valido:
        resposta, status, fonte_id = bruto[:600], "json_invalido", None
    elif dados["status"] == "nao_consta":
        resposta, status, fonte_id = NAO_CONSTA, "nao_consta", None
    else:
        resposta, status = dados["resposta"].strip(), "ok"
        fonte_id = str(dados["fonte_id"]).strip().strip("[]") if dados.get("fonte_id") else None

    ids_recuperados = {t["id"] for t in recuperados}
    citacao_valida = (fonte_id in ids_recuperados) if (modo == "rag" and status == "ok") else None
    trecho = next((t for t in recuperados if t["id"] == fonte_id), None)
    return {
        **base_resultado,
        "resposta": resposta, "status": status, "fonte_id": fonte_id,
        "fonte": trecho["fonte"] if trecho else None, "url": trecho["url"] if trecho else None,
        "json_valido": valido, "citacao_valida": citacao_valida,
        "origem": "modelo", "bruto": bruto, "latencia_s": round(time.time() - t0, 3),
    }


def saida_publica(r: dict) -> dict:
    """Campos devolvidos ao cidadão pela API/interface (sem o texto bruto do modelo)."""
    chaves = ["resposta", "status", "fonte_id", "fonte", "url", "score_top1", "trechos_recuperados", "aviso"]
    return {c: r.get(c) for c in chaves}


# ---------------------------------------------------------------- avaliação
def normalizar(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def avaliar_caso(caso: dict, r: dict) -> dict:
    """Compara uma resposta com o gabarito. Cada grupo em `chaves` precisa ter ao menos um termo presente."""
    resposta = normalizar(r["resposta"])
    if caso["tipo"] == "nao_consta":
        recusou = r["status"] == "nao_consta"
        return {
            "acerto": recusou, "resposta_correta": recusou, "citacao_correta": None,
            "alucinou": r["status"] == "ok", "recusa_indevida": False,
        }
    contem = all(any(normalizar(t) in resposta for t in grupo) for grupo in caso["chaves"])
    resposta_correta = r["status"] == "ok" and contem
    citacao_correta = r.get("fonte_id") in caso["doc_ids"]
    return {
        "acerto": resposta_correta and citacao_correta,
        "resposta_correta": resposta_correta,
        "citacao_correta": citacao_correta,
        "alucinou": r["status"] == "ok" and not contem,
        "recusa_indevida": r["status"] == "nao_consta",
    }


def resumo_config(nome: str, casos: list[dict], resultados: list[dict]) -> dict:
    av = [avaliar_caso(c, r) for c, r in zip(casos, resultados)]
    esp = [(c, r, a) for c, r, a in zip(casos, resultados, av) if c["tipo"] == "esperado"]
    nc = [(c, r, a) for c, r, a in zip(casos, resultados, av) if c["tipo"] == "nao_consta"]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else 0.0
    return {
        "configuracao": nome,
        "json_valido_%": pct([r["json_valido"] for r in resultados]),
        "resposta_correta_%": pct([a["resposta_correta"] for _, _, a in esp]),
        "resposta_com_citacao_correta_%": pct([a["acerto"] for _, _, a in esp]),
        "recusa_correta_nao_consta_%": pct([a["acerto"] for _, _, a in nc]),
        "alucinacao_nao_consta_%": pct([a["alucinou"] for _, _, a in nc]),
        "resposta_errada_afirmada_%": pct([a["alucinou"] for _, _, a in esp]),
        "recusa_indevida_%": pct([a["recusa_indevida"] for _, _, a in esp]),
        "acerto_geral_%": pct([a["acerto"] for a in av]),
        "latencia_media_s": round(float(np.mean([r["latencia_s"] for r in resultados])), 2),
    }


def diagnosticar(caso: dict, r: dict) -> str:
    """Rótulo de causa provável para a análise dos erros."""
    a = avaliar_caso(caso, r)
    if a["acerto"]:
        return "ok"
    if r["status"] == "json_invalido":
        return "json_invalido"
    ids = [t["id"] for t in r.get("trechos_recuperados", [])]
    if caso["tipo"] == "nao_consta":
        return "alucinacao_fora_da_base"
    if r.get("origem") == "limiar":
        return "recusa_indevida_por_limiar"
    if r["modo"] == "rag" and not set(caso["doc_ids"]) & set(ids):
        return "recuperacao_errada"
    if r["status"] == "nao_consta":
        return "recusa_indevida_pelo_modelo"
    if not a["resposta_correta"]:
        return "resposta_incompleta_ou_errada"
    return "citacao_errada"


# ---------------------------------------------------------------- LGPD
_PADROES_PII = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
]


def mascarar_pii(texto: str) -> str:
    """Remove CPF, e-mail e telefone antes de qualquer registro em log (minimização de dados)."""
    for padrao, troca in _PADROES_PII:
        texto = padrao.sub(troca, texto)
    return texto
