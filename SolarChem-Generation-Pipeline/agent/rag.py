import json
import math
from typing import Any, Dict, List, Optional

from langchain_text_splitters  import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBEDDING_MODEL = "avr/sfr-embedding-mistral:latest"
DEFAULT_RERANK_MODEL = "rank-T5-flan"
DEFAULT_HYBRID_WEIGHTS = (0.5, 0.5)

CHUNK_TYPES = ("Naive", "Recursive", "Semantic")
RANK_TYPES = ("Naive", "Hybrid", "Rerank")


class Ragger:

    def __init__(
        self,
        chunk_type: str = "Naive",
        rank_type: str = "Naive",
        chunk_size: int = 1024,
        overlap: int = 128,
        top_k: int = 5,
        paper_sections: Optional[List[str]] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        hybrid_weights: tuple = DEFAULT_HYBRID_WEIGHTS,
    ) -> None:
        if chunk_type not in CHUNK_TYPES:
            raise ValueError(f"chunk_type must be one of {CHUNK_TYPES}, got {chunk_type!r}")
        if rank_type not in RANK_TYPES:
            raise ValueError(f"rank_type must be one of {RANK_TYPES}, got {rank_type!r}")

        self.chunk_type = chunk_type
        self.rank_type = rank_type
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.top_k = int(top_k)
        self.paper_sections = list(paper_sections or [])
        self.rerank_model = rerank_model
        self.hybrid_weights = list(hybrid_weights)
        self.embeddings = OpenAIEmbeddings(
            model="qwen/qwen3-embedding-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-xxx",
            check_embedding_ctx_length=False,
        )

        self.paper_index: Optional[str] = None
        self.input_data: str = ""

    def build_for_paper(
        self,
        paper_json: List[Dict[str, Any]],
        paper_index: Optional[str] = None,
    ) -> "Ragger":
        self.paper_index = paper_index
        self.input_data = self._join_sections(paper_json)
        return self

    @classmethod
    def from_path(cls, input_path: str, **kwargs) -> "Ragger":
        with open(input_path, "rb") as f:
            data = json.load(f)
        paper_index = input_path.split("_")[-1].split(".")[0]
        return cls(**kwargs).build_for_paper(data, paper_index=paper_index)

    def _join_sections(self, data: List[Dict[str, Any]]) -> str:
        text = ""
        for item in data:
            title = item.get("title")
            if title not in self.paper_sections:
                continue
            content = item.get("content") or ""
            text += f"{title}:\n{content}\n"
        return text

    def get_chunks(self) -> List[Document]:
        if self.chunk_type == "Naive":
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separator="",
                strip_whitespace=False,
            )
        elif self.chunk_type == "Recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                length_function=len,
                is_separator_regex=False,
            )
        else:
            number_of_chunks = max(
                1, int(math.ceil(len(self.input_data) / self.chunk_size))
            )
            min_chunk_size = int(math.ceil(self.chunk_size / 2))
            text_splitter = SemanticChunker(
                self.embeddings,
                number_of_chunks=number_of_chunks,
                min_chunk_size=min_chunk_size,
            )

        texts = text_splitter.split_text(self.input_data)
        return [Document(page_content=x) for x in texts]

    def ranking(self, chunks: List[Document], query: str) -> Dict[int, str]:
        if not chunks:
            return {}

        retriever = FAISS.from_documents(chunks, self.embeddings).as_retriever(
            search_kwargs={"k": len(chunks)},
        )

        if self.rank_type == "Naive":
            retri_docs = retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]
        elif self.rank_type == "Rerank":
            compressor = FlashrankRerank(model=self.rerank_model, top_n=len(chunks))
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever,
            )
            compressed_docs = compression_retriever.invoke(query)
            retri_text = [x.page_content for x in compressed_docs]
        else:
            keyword_retriever = BM25Retriever.from_documents(chunks)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever, keyword_retriever],
                weights=self.hybrid_weights,
            )
            retri_docs = ensemble_retriever.invoke(query)
            retri_text = [x.page_content for x in retri_docs]

        return {k: retri_text[k] for k in range(len(retri_text))}

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        k = self.top_k if top_k is None else top_k
        chunks = self.get_chunks()
        retri_index = self.ranking(chunks, query)
        if not retri_index:
            return []
        k = min(k, len(retri_index))
        return [retri_index[i] for i in range(k)]
