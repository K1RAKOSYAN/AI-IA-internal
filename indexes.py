
from pinecone import Pinecone, ServerlessSpec


import pinecone
from typing import Literal



def create_index_if_not_exists(
    index_name: str,
    index_metric: Literal["cosine", "euclidean", "dotproduct"],
    spec: ServerlessSpec,
    pc:  Pinecone,
    dimension=3072,
):
    try:
        
        if index_name not in pc.list_indexes():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=index_metric,
                spec=spec,
            )
        print(f"Index {index_name} created")
    except pinecone.exceptions.PineconeException as e:
        print(f"Error: {e.reason}")