from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend/app')
from core.config_env import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()