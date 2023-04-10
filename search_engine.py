import time
from dataclasses import dataclass
from typing import List, Dict, Any
from pinecone import Index
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
from pinecone_text.dense.base_dense_ecoder import BaseDenseEncoder
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone import SparseValues


@dataclass
class SearchResult:

    doc_id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class TopKSearchResults:

    results: List[SearchResult]


class SearchEngine:

    def __init__(self,
                 pinecone_index: Index,
                 dense_encoder: BaseDenseEncoder,
                 sparse_encoder: BaseSparseEncoder):
        self._index = pinecone_index
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder

    def search(self, query_text: str, top_k: int, alpha: float) -> TopKSearchResults:

        # we cap alpha at 0.999 to always able to extract the sparse score from the total score
        d_vector, s_vector = hybrid_convex_scale(self._dense_encoder.encode_queries(query_text),
                                                 self._sparse_encoder.encode_queries(query_text),
                                                 alpha)
        if len(s_vector["indices"]):
            query_result = self._index.query(top_k=top_k,
                                             values=d_vector,
                                             sparse_values=SparseValues(indices=s_vector["indices"],
                                                                        values=s_vector["values"]),
                                             include_metadata=True)
        else:
            query_result = self._index.query(top_k=top_k,
                                             values=d_vector,
                                             include_metadata=True)

        results: List[SearchResult] = []
        for res in query_result:
            # if score is zero the match is completely arbitrary, so we skip it (might happen for alpha==0.0)
            if res.score <= 0.000001:
                continue

            result_obj = SearchResult(doc_id=res.id,
                                      score=round(res.score, 2),
                                      metadata=res.metadata)
            results.append(result_obj)
        return TopKSearchResults(results)
