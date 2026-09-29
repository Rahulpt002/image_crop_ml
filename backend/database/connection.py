"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config.settings import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite-specific
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and ensure columns exist."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        for col, col_type in [
            ("face_detected", "INTEGER DEFAULT 0"),
            ("face_box_x", "INTEGER"),
            ("face_box_y", "INTEGER"),
            ("face_box_width", "INTEGER"),
            ("face_box_height", "INTEGER"),
            ("face_confidence", "FLOAT"),
        ]:
            try:
                from sqlalchemy import text
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass
