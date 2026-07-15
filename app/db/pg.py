from __future__ import annotations

import asyncpg
from app.config import settings

_pool: asyncpg.Pool | None = None


async def get_pg_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.postgres.dsn,
            min_size=1,
            max_size=5,
        )
    return _pool


async def init_pg_tables():
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_goals (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                gap_snapshot JSONB NOT NULL DEFAULT '[]',
                source_cv_submission_id TEXT,
                achieved_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_goal_steps (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES skill_goals(id) ON DELETE CASCADE,
                course_id TEXT NOT NULL,
                sequence_order INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'not_started',
                skills_addressed JSONB NOT NULL DEFAULT '[]',
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_goal_events (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES skill_goals(id) ON DELETE CASCADE,
                type TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_goals_user_status ON skill_goals(user_id, status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_goal_steps_goal ON skill_goal_steps(goal_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_goal_events_goal ON skill_goal_events(goal_id)"
        )
