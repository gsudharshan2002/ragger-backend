from fastapi import APIRouter

from app.api.v1 import chat, documents, knowledge_bases, datasets, benchmark, health
from app.api.v1 import rag_config, traces, documents_sub, kb_documents
from app.api.v1.agent import agent_router

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(documents_sub.router, prefix="/documents", tags=["documents"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
api_router.include_router(kb_documents.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(benchmark.router, prefix="/benchmark", tags=["benchmark"])
api_router.include_router(rag_config.router, prefix="/rag", tags=["rag"])
api_router.include_router(traces.router, prefix="/traces", tags=["traces"])
