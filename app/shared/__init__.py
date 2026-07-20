"""Shared infrastructure for the NeuroStack AI Layer.

This package contains cross-cutting concerns:
- database.py: PostgreSQL connection pool and DDL initialization
- providers/: LLM and vector store abstractions
- utils/: Deduplicated helper functions
- middleware/: Request context middleware
- prompts/: Shared prompt templates
- parsers/: Document parsing utilities
"""
