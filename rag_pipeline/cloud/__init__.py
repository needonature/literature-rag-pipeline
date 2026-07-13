"""GCP cloud-native backends for the RAG pipeline (Vertex AI + BigQuery).

These modules are import-safe with no credentials: they only touch GCP when you
actually call their functions. See DEPLOY_GCP.md for the end-to-end runbook.
"""
