"""
API do AgroFácil GenAI.

Executar na raiz do repositório:
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /saude      -> estado do serviço e configuração
    POST /perguntar  -> {"pergunta": "..."}  =>  {"resposta", "status", "fonte_id", "fonte", "url", ...}
"""
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agrofacil as af  

log = logging.getLogger("agrofacil.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ESTADO: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    cfg = af.carregar_config()
    adapter = af.CAMINHO_ADAPTER if af.CAMINHO_ADAPTER.exists() else None
    ESTADO["cfg"] = cfg
    ESTADO["lora"] = adapter is not None
    ESTADO["buscador"] = af.Buscador(af.carregar_base(), cfg["encoder"])
    ESTADO["model"], ESTADO["tok"] = af.carregar_modelo(adapter=adapter)
    log.info("Modelo carregado | LoRA=%s | limiar=%s", ESTADO["lora"], cfg["limiar"])
    yield
    ESTADO.clear()


app = FastAPI(title="AgroFácil GenAI", version="1.0.0", lifespan=ciclo_de_vida,
              description="Assistente de segurança climática para pequenos produtores rurais (RAG + LoRA local).")


class Pergunta(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500,
                          examples=["O que fazer quando sai aviso de geada?"])


@app.get("/saude")
def saude():
    return {"status": "ok" if "model" in ESTADO else "carregando",
            "modelo": af.MODELO_BASE, "lora": ESTADO.get("lora"), "config": ESTADO.get("cfg")}


@app.post("/perguntar")
def perguntar(p: Pergunta):
    if "model" not in ESTADO:
        raise HTTPException(503, "Modelo ainda carregando")
    cfg = ESTADO["cfg"]
    r = af.responder(p.pergunta, ESTADO["model"], ESTADO["tok"], buscador=ESTADO["buscador"],
                     modo="rag", limiar=cfg["limiar"], k=cfg.get("k", 3))
    log.info(json.dumps({"pergunta": af.mascarar_pii(p.pergunta), "status": r["status"],
                         "fonte_id": r["fonte_id"], "latencia_s": r["latencia_s"]}, ensure_ascii=False))
    return af.saida_publica(r)
