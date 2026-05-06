
"""
main.py
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard.conversation_router import router as conversations_router, get_conn
from dashboard.conversation_meta import init_meta_table, bootstrap_meta
from dashboard.feature_flags import load_flags,init_flags
from voice.ws_routes import router as ws_router
from whatsapp.webhook import router as whatsapp_router, start_workers
#from chat.chat_webhook import router as chat_router
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Start WhatsApp background workers
    start_workers()

    # 2. Ensure conversation_meta table exists and is populated
    try:
        conn = get_conn()
        init_meta_table(conn)   # creates table if not exists (idempotent)
        bootstrap_meta(conn)    # fills rows missing from meta (safe on every restart)
        init_flags(conn) 
        load_flags(conn) 
        conn.close()
        log.info("[STARTUP] conversation_meta ready")
    except Exception:
        log.exception("[STARTUP] Failed to initialise conversation_meta — continuing anyway")

    yield


app = FastAPI(title="Voice Agent", lifespan=lifespan)

app.include_router(conversations_router)
app.include_router(ws_router)
app.include_router(whatsapp_router)
#app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)