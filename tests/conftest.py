import uuid

import chromadb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from logisense.config import Settings
from logisense.db import Base
from logisense.services.rag.vector_store import ChromaVectorStore


@pytest.fixture()
def session():
    # StaticPool + check_same_thread=False: FastAPI's TestClient runs
    # requests in a worker thread, and a bare sqlite:///:memory: engine
    # opens a fresh (empty) DB per connection/thread otherwise.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def vector_store():
    # chromadb.EphemeralClient() shares collection state across instances
    # *within the same process* when the collection name matches -- each
    # test needs its own unique collection name to stay isolated.
    settings = Settings(
        database_url="sqlite:///:memory:",
        gemini_api_key="x",
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )
    return ChromaVectorStore(settings, client=chromadb.EphemeralClient())
