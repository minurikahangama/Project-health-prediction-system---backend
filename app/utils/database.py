"""
Database connection, session management, and base class.
All models import Base from here.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Copy .env.example to .env and fill in your PostgreSQL credentials."
    )

engine_options = {"pool_pre_ping": True}
# SQLite's SingletonThreadPool does not accept PostgreSQL queue-pool sizing
# options. Keeping them conditional lets isolated integration tests use the
# documented sqlite:///:memory: URL without changing production behaviour.
if not DATABASE_URL.startswith("sqlite"):
    engine_options.update(pool_size=10, max_overflow=20)

engine = create_engine(DATABASE_URL, **engine_options)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency — yields a database session per request,
    always closes it afterwards even if an exception occurs.

    Usage:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
