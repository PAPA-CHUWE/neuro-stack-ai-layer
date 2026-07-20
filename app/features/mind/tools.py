"""Mind feature — Page context tool handlers.

These are server-side tool handlers called by the orchestrator when pageContext
is present and the user's question refers to their current location.
"""

import json
import logging

from app.shared.database import get_pg_pool

logger = logging.getLogger(__name__)


async def get_page_context(route: str, tenant_id: str) -> dict:
    """Fetch structured metadata and live data for a given app route."""
    pool = await get_pg_pool()
    result: dict = {"route": route, "widgets": [], "description": ""}

    async with pool.acquire() as conn:
        if route.startswith("/learner") or route.startswith("/student"):
            result["description"] = "Learner dashboard with enrolled courses and progress."
            rows = await conn.fetch(
                """SELECT c.title, e.status
                   FROM enrollments e
                   JOIN courses c ON e."courseId" = c.id
                   WHERE e."userId" = (SELECT id FROM users WHERE "tenantId" = $1 LIMIT 1)
                   ORDER BY e."createdAt" DESC LIMIT 5""",
                tenant_id,
            )
            result["widgets"] = [
                {"label": r["title"], "value": r["status"]} for r in rows
            ] if rows else []

        elif route.startswith("/admin/courses"):
            result["description"] = "Course management — list, create, and edit courses."
            rows = await conn.fetch(
                'SELECT title, status, "createdAt" AS created_at FROM courses ORDER BY "createdAt" DESC LIMIT 10'
            )
            result["widgets"] = [
                {"label": r["title"], "value": r["status"]} for r in rows
            ]

        elif route.startswith("/admin/users"):
            result["description"] = "User management — invite, edit, and deactivate users."
            rows = await conn.fetch(
                """SELECT * FROM (
                     SELECT DISTINCT ON (u.id) u.id,
                       (u."firstName" || ' ' || u."lastName") AS name, u.email, u.status,
                       COALESCE(r.name, 'No Role') AS role, u."createdAt" AS created_at
                     FROM users u
                     LEFT JOIN user_roles ur ON ur."userId" = u.id
                     LEFT JOIN roles r ON r.id = ur."roleId"
                     ORDER BY u.id, u."createdAt" DESC
                   ) t ORDER BY created_at DESC LIMIT 10"""
            )
            result["widgets"] = [
                {"label": r["name"] or r["email"], "value": f"{r['role']} — {r['status']}"} for r in rows
            ]

        elif route.startswith("/admin/knowledge"):
            result["description"] = "Knowledge Base management — upload documents, configure Mind."
            docs = await conn.fetchval("SELECT COUNT(*) FROM rag_documents WHERE tenant_id = $1", tenant_id)
            chunks = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_chunks rc JOIN rag_documents rd ON rc.document_id = rd.id WHERE rd.tenant_id = $1",
                tenant_id,
            )
            result["widgets"] = [
                {"label": "Documents", "value": docs},
                {"label": "Chunks Indexed", "value": chunks},
            ]

        elif route.startswith("/admin/feedback"):
            result["description"] = "Feedback review — triage and resolve AI response feedback."
            new_count = await conn.fetchval(
                "SELECT COUNT(*) FROM ai_feedback WHERE tenant_id = $1 AND status = 'new'", tenant_id
            )
            critical = await conn.fetchval(
                "SELECT COUNT(*) FROM ai_feedback WHERE tenant_id = $1 AND status IN ('new','triaged') AND priority IN ('critical','high')",
                tenant_id,
            )
            result["widgets"] = [
                {"label": "New Feedback", "value": new_count},
                {"label": "Critical/High Unresolved", "value": critical},
            ]

        elif route.startswith("/admin/tenants"):
            result["description"] = "Tenant management — configure organization-level settings."
            tenants = await conn.fetchval("SELECT COUNT(*) FROM tenants")
            result["widgets"] = [{"label": "Tenants", "value": tenants}]

        elif route.startswith("/admin/roles"):
            result["description"] = "Role management — configure permissions and access control."

        elif route.startswith("/admin/permissions"):
            result["description"] = "Permission management — fine-grained access control for every role and module."
            permissions = await conn.fetchval("SELECT COUNT(*) FROM permissions")
            roles = await conn.fetchval("SELECT COUNT(*) FROM roles")
            result["widgets"] = [
                {"label": "Total Permissions", "value": permissions},
                {"label": "Active Roles", "value": roles},
            ]

        elif route.startswith("/admin/audit-logs"):
            result["description"] = "Audit logs — track all administrative actions."

        elif route.startswith("/admin/settings"):
            result["description"] = "Platform settings — authentication, integrations, notifications."

        elif route.startswith("/reports"):
            result["description"] = "Reporting dashboard — analytics and insights."

        elif route.startswith("/learning"):
            result["description"] = "Learning hub — browse and enroll in courses."

        elif route == "/admin":
            courses = await conn.fetchval("SELECT COUNT(*) FROM courses")
            published = await conn.fetchval("SELECT COUNT(*) FROM courses WHERE status = 'PUBLISHED'")
            users = await conn.fetchval("SELECT COUNT(*) FROM users")
            active_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE status = 'ACTIVE'")
            enrollments = await conn.fetchval("SELECT COUNT(*) FROM enrollments")
            departments = await conn.fetchval("SELECT COUNT(*) FROM departments")
            result["description"] = "Admin dashboard with platform management tools."
            result["widgets"] = [
                {"label": "Total Courses", "value": courses},
                {"label": "Published Courses", "value": published},
                {"label": "Total Users", "value": users},
                {"label": "Active Users", "value": active_users},
                {"label": "Total Enrollments", "value": enrollments},
                {"label": "Departments", "value": departments},
            ]

        else:
            result["description"] = f"Page at route {route}."

    return result


async def get_entity_context(entity_type: str, entity_id: str, tenant_id: str) -> dict:
    """Fetch details for a specific selected record by type and id."""
    pool = await get_pg_pool()
    result: dict = {"entityType": entity_type, "entityId": entity_id, "data": None}

    async with pool.acquire() as conn:
        if entity_type == "course":
            row = await conn.fetchrow(
                """SELECT id, title, description, status,
                   "createdAt" AS created_at, "updatedAt" AS updated_at FROM courses WHERE id = $1""", entity_id
            )
            if row:
                result["data"] = dict(row)
                result["data"]["created_at"] = str(row["created_at"])
                result["data"]["updated_at"] = str(row["updated_at"])
                count = await conn.fetchval(
                    'SELECT COUNT(*) FROM enrollments WHERE "courseId" = $1', entity_id
                )
                result["data"]["enrolled_learners"] = count

        elif entity_type == "user":
            row = await conn.fetchrow(
                """SELECT u.id, (u."firstName" || ' ' || u."lastName") AS name, u.email,
                   COALESCE(r.name, 'No Role') AS role, u.status, u."tenantId" AS tenant_id,
                   u."createdAt" AS created_at
                   FROM users u
                   LEFT JOIN user_roles ur ON ur."userId" = u.id
                   LEFT JOIN roles r ON r.id = ur."roleId"
                   WHERE u.id = $1 LIMIT 1""", entity_id
            )
            if row:
                result["data"] = dict(row)
                result["data"]["created_at"] = str(row["created_at"])

        elif entity_type == "tenant":
            row = await conn.fetchrow(
                'SELECT id, name, slug, "isActive", "createdAt" AS created_at FROM tenants WHERE id = $1', entity_id
            )
            if row:
                result["data"] = dict(row)
                result["data"]["created_at"] = str(row["created_at"])

    return result
