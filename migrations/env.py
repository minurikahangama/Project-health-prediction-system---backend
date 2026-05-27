"""
Alembic environment configuration.

This file is used by `alembic revision --autogenerate` to detect
schema changes from your SQLAlchemy models.

IMPORTANT: Run alembic commands from inside the /backend folder:
    cd backend
    alembic revision --autogenerate -m "description"
    alembic upgrade head
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Alembic Config object
config = context.config

# Set the database URL from .env
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import ALL models so Alembic can detect them
from app.models.models import Base   # noqa: F401 — import all models via Base
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url                     = url,
        target_metadata         = target_metadata,
        literal_binds           = True,
        dialect_opts            = {"paramstyle": "named"},
        compare_type            = True,
        compare_server_default  = True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix  = "sqlalchemy.",
        poolclass = pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection              = connection,
            target_metadata         = target_metadata,
            compare_type            = True,
            compare_server_default  = True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
