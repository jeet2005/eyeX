"""
Database configuration and session management using SQLAlchemy async.
"""

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager

import config

# Create database URL
DATABASE_URL = f"sqlite+aiosqlite:///{config.DATA_DIR / 'attendance.db'}"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base for models
Base = declarative_base()


async def init_db():
    """Initialize database - create all tables"""
    # Import models to register them
    from database.models import Student, Attendance
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database initialized")


async def get_db():
    """Dependency for FastAPI - yields database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_session():
    """Context manager for manual session usage"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
