"""AWS path #3 — Amazon OpenSearch (Serverless) k-NN: cloud-native managed vector
search. The AWS analog of Vertex AI Vector Search (cloud/gcp_vertex.py).

Requests are SigV4-signed via boto3 credentials; opensearch-py talks to the
collection. Needs an OpenSearch Serverless collection (or a managed domain) plus
a data-access policy granting your principal. boto3/opensearch-py are imported
lazily, so importing this module never needs AWS.

    create_index(dim)                 -> create the knn index
    index_documents(chunks, vectors)  -> bulk-load embedded chunks
    knn_search(query_vec, k)          -> [(source_dict, score), ...]

See cloud/DEPLOY_AWS.md for the collection + IAM setup.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import config


def _client():
    import boto3  # lazy
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

    creds = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(creds, config.AWS_REGION, config.OPENSEARCH_SERVICE)
    if not config.OPENSEARCH_HOST:
        raise RuntimeError("RAG_OPENSEARCH_HOST not set (the collection endpoint, no https://).")
    return OpenSearch(
        hosts=[{"host": config.OPENSEARCH_HOST, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def create_index(dim: int, client=None) -> str:
    client = client or _client()
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": int(dim),
                    "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
                },
                "body": {"type": "text"},
                "pmid": {"type": "keyword"},
                "title": {"type": "text"},
                "biomarker": {"type": "keyword"},
                "biofluid": {"type": "keyword"},
            }
        },
    }
    if not client.indices.exists(config.OPENSEARCH_INDEX):
        client.indices.create(config.OPENSEARCH_INDEX, body=body)
    return config.OPENSEARCH_INDEX


def index_documents(chunks: Sequence[Any], vectors, client=None) -> int:
    from opensearchpy.helpers import bulk

    client = client or _client()
    actions = [
        {
            "_index": config.OPENSEARCH_INDEX,
            "_source": {
                "embedding": vectors[i].tolist(),
                "body": c.text,
                "pmid": c.metadata.get("pmid"),
                "title": c.metadata.get("title"),
                "biomarker": c.metadata.get("biomarker"),
                "biofluid": c.metadata.get("biofluid"),
            },
        }
        for i, c in enumerate(chunks)
    ]
    bulk(client, actions)
    client.indices.refresh(config.OPENSEARCH_INDEX)
    return len(actions)


def knn_search(query_vec: Sequence[float], k: int = 6, client=None) -> List[Tuple[Dict[str, Any], float]]:
    client = client or _client()
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": list(query_vec), "k": k}}},
        "_source": ["pmid", "title", "biomarker", "biofluid", "body"],
    }
    res = client.search(index=config.OPENSEARCH_INDEX, body=body)
    return [(h["_source"], float(h["_score"])) for h in res["hits"]["hits"]]


if __name__ == "__main__":
    print("aws_opensearch.py import OK.")
    print("  set RAG_OPENSEARCH_HOST to your collection endpoint, then build + query.")
