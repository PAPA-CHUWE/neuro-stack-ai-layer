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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cv_extractions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                cv_text TEXT NOT NULL,
                skills JSONB NOT NULL DEFAULT '[]',
                model TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                validation JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'default',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                chunk_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'default',
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_id TEXT,
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
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cv_extractions_user ON cv_extractions(user_id, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_docs_tenant ON rag_documents(tenant_id, collection)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(document_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant ON rag_chunks(tenant_id, collection)"
        )
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mind_feedback (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence TEXT NOT NULL,
                sources_used JSONB NOT NULL DEFAULT '[]',
                rating TEXT NOT NULL,
                reason TEXT,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mind_feedback_tenant ON mind_feedback(tenant_id, review_status)"
        )

        # ── AI Response Traces ────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_response_traces (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT,
                user_intent TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT,
                knowledge_collection_codes TEXT[] DEFAULT '{}',
                retrieved_document_ids TEXT[] DEFAULT '{}',
                retrieved_chunk_ids TEXT[] DEFAULT '{}',
                tool_calls JSONB,
                tool_results_summary TEXT,
                latency_ms INTEGER,
                input_token_count INTEGER,
                output_token_count INTEGER,
                response_status TEXT NOT NULL DEFAULT 'success',
                grounding_status TEXT NOT NULL DEFAULT 'ungrounded',
                confidence_score REAL,
                citation_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_tenant ON ai_response_traces(tenant_id, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_grounding ON ai_response_traces(grounding_status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_conversation ON ai_response_traces(conversation_id)"
        )

        # ── Structured Feedback (expanded) ────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_feedback (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                message_id TEXT,
                response_trace_id TEXT,
                rating TEXT NOT NULL,
                numeric_score INTEGER,
                reason_codes TEXT[] DEFAULT '{}',
                comment TEXT,
                suggested_correction TEXT,
                status TEXT NOT NULL DEFAULT 'new',
                priority TEXT NOT NULL DEFAULT 'low',
                triage_category TEXT,
                triage_confidence REAL,
                triage_suggested_owner TEXT,
                triage_suggested_remediation TEXT,
                assigned_reviewer_id TEXT,
                review_decision TEXT,
                review_notes TEXT,
                reviewed_at TIMESTAMPTZ,
                linked_document_id TEXT,
                linked_prompt_version_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_tenant_status ON ai_feedback(tenant_id, status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_tenant_rating ON ai_feedback(tenant_id, rating)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_triage ON ai_feedback(triage_category)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_status_priority ON ai_feedback(status, priority)"
        )

        # ── Evaluation Cases ──────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_evaluation_cases (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                category TEXT NOT NULL,
                user_input TEXT NOT NULL,
                expected_intent TEXT,
                expected_tool TEXT,
                expected_knowledge_codes TEXT[] DEFAULT '{}',
                reference_answer TEXT,
                prohibited_claims TEXT[] DEFAULT '{}',
                style_requirements TEXT[] DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'draft',
                source_feedback_id TEXT,
                created_by TEXT NOT NULL,
                approved_by TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_cases_tenant ON ai_evaluation_cases(tenant_id, status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_cases_category ON ai_evaluation_cases(category)"
        )

        # ── Evaluation Runs ───────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_evaluation_runs (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                case_id TEXT NOT NULL,
                prompt_version TEXT,
                model TEXT NOT NULL,
                actual_output TEXT,
                correctness_score REAL,
                groundedness_score REAL,
                retrieval_precision REAL,
                citation_accuracy REAL,
                style_compliance REAL,
                hallucination_detected BOOLEAN DEFAULT FALSE,
                prohibited_claim_triggered BOOLEAN DEFAULT FALSE,
                latency_ms INTEGER,
                passed BOOLEAN,
                regression_of_run_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_runs_case ON ai_evaluation_runs(case_id)"
        )

        # ── Prompt Template Versions ──────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_template_versions (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                code TEXT NOT NULL,
                version INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                change_note TEXT,
                created_by TEXT NOT NULL,
                approved_by TEXT,
                activated_at TIMESTAMPTZ,
                retired_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(code, version, tenant_id)
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_tenant ON prompt_template_versions(tenant_id, code)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_status ON prompt_template_versions(status)"
        )

        # ── Mind Conversations ─────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mind_conversations (
                conversation_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT,
                has_introduced BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mind_conv_tenant ON mind_conversations(tenant_id, created_at DESC)"
        )

        # ── Mind Messages (persisted conversation transcript) ─
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mind_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mind_messages_conversation ON mind_messages(conversation_id, created_at)"
        )
