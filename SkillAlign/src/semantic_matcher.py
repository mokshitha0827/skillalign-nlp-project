"""Semantic similarity: pretrained Sentence Transformer (preferred) with a documented TF-IDF fallback.

* ``embeddings`` method - ``all-MiniLM-L6-v2`` sentence vectors compared with cosine similarity.
* ``tfidf`` fallback    - lexical TF-IDF vectors (skill mentions normalised to canonical tokens, light stemming).

Cosine similarity is a *relative closeness score*, not a probability and not proof of competence.
The two methods produce differently-scaled scores, so each has its own configurable threshold.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .skill_extractor import find_skills_in_text

DEFAULT_MODEL = "all-MiniLM-L6-v2"
METHOD_EMBEDDINGS = "embeddings"
METHOD_TFIDF = "tfidf"

_MODEL_CACHE: dict[str, tuple[object | None, str | None]] = {}


def _default_loader(model_name: str):
    from sentence_transformers import SentenceTransformer  # imported lazily: heavy dependency
    return SentenceTransformer(model_name)


def reset_model_cache() -> None:
    """Forget cached load results (e.g. before retrying a failed download)."""
    _MODEL_CACHE.clear()


# ------------------------------------------------------------------ TF-IDF helpers
_TOKEN = re.compile(r"[a-z0-9_+#]+")
_STOP = set(ENGLISH_STOP_WORDS) | {"experience", "using", "used", "use", "work", "worked", "working"}


def _stem(tok: str) -> str:
    for suf in ("ations", "ation", "ing", "ers", "er", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tfidf_analyzer(text: str) -> list[str]:
    """Tokenise for TF-IDF: skill mentions become canonical tokens so aliases line up (sklearn == scikit-learn)."""
    pieces, last = [], 0
    for skill, _m, start, end in find_skills_in_text(text):
        pieces.append(text[last:start])
        pieces.append(" skill_" + re.sub(r"[^a-z0-9]+", "_", skill.lower()) + " ")
        last = end
    pieces.append(text[last:])
    toks = _TOKEN.findall(" ".join(pieces).lower())
    return [t if t.startswith("skill_") else _stem(t) for t in toks if t not in _STOP and len(t) > 1]


class SemanticMatcher:
    """Compute a ``(n_queries, n_sentences)`` cosine-similarity matrix and remember which method was used.

    Parameters
    ----------
    preference : ``"auto"`` / ``"embeddings"`` try the pretrained model and fall back to TF-IDF on any failure;
                 ``"tfidf"`` skips the model entirely.
    loader     : optional callable ``model_name -> object with .encode(list[str])`` (used by tests/offline demos).
    """

    def __init__(self, preference: str = "auto", model_name: str = DEFAULT_MODEL,
                 loader: Callable[[str], object] | None = None):
        if preference not in {"auto", "embeddings", "tfidf"}:
            raise ValueError("preference must be 'auto', 'embeddings' or 'tfidf'")
        self.preference = preference
        self.model_name = model_name
        self._loader = loader
        self._model = None
        self.method: str | None = None
        self.note: str = ""

    # -- loading
    def load(self) -> str:
        if self.method is not None:
            return self.method
        if self.preference == "tfidf":
            self.method, self.note = METHOD_TFIDF, "TF-IDF selected by the user (embedding model not used)."
            return self.method
        model, err = self._get_model()
        if model is None:
            self.method = METHOD_TFIDF
            self.note = (f"Embedding model '{self.model_name}' could not be loaded ({err}). "
                         "Using the TF-IDF fallback instead.")
        else:
            self._model, self.method = model, METHOD_EMBEDDINGS
            self.note = f"Pretrained Sentence Transformer '{self.model_name}' with cosine similarity."
        return self.method

    def _get_model(self):
        if self._loader is not None:
            try:
                return self._loader(self.model_name), None
            except Exception as exc:
                return None, f"{type(exc).__name__}: {str(exc)[:120]}"
        if self.model_name in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_name]
        try:
            result = (_default_loader(self.model_name), None)
        except Exception as exc:  # ImportError, OSError (offline), etc.
            result = (None, f"{type(exc).__name__}: {str(exc)[:120]}")
        _MODEL_CACHE[self.model_name] = result
        return result

    # -- similarity
    def similarity(self, queries: list[str], sentences: list[str]) -> np.ndarray:
        method = self.load()
        if not queries or not sentences:
            return np.zeros((len(queries), len(sentences)))
        if method == METHOD_EMBEDDINGS:
            try:
                return self._embedding_similarity(queries, sentences)
            except Exception as exc:  # runtime failure -> degrade gracefully
                self.method = METHOD_TFIDF
                self.note = f"Embedding encoding failed ({type(exc).__name__}); switched to the TF-IDF fallback."
        return self._tfidf_similarity(queries, sentences)

    def _embedding_similarity(self, queries, sentences) -> np.ndarray:
        q = np.asarray(self._model.encode(list(queries), normalize_embeddings=True, show_progress_bar=False))
        s = np.asarray(self._model.encode(list(sentences), normalize_embeddings=True, show_progress_bar=False))
        return np.clip(cosine_similarity(q, s), -1.0, 1.0)

    @staticmethod
    def _tfidf_similarity(queries, sentences) -> np.ndarray:
        vec = TfidfVectorizer(analyzer=tfidf_analyzer, sublinear_tf=True)
        try:
            matrix = vec.fit_transform(list(sentences) + list(queries))
        except ValueError:  # empty vocabulary
            return np.zeros((len(queries), len(sentences)))
        s, q = matrix[: len(sentences)], matrix[len(sentences):]
        return np.clip(cosine_similarity(q, s), 0.0, 1.0)

    def info(self) -> dict:
        self.load()
        return {"method": self.method, "model_name": self.model_name if self.method == METHOD_EMBEDDINGS else None,
                "note": self.note}


if __name__ == "__main__":  # python -m src.semantic_matcher --download
    import sys
    if "--download" in sys.argv:
        m = SemanticMatcher("embeddings")
        print(m.load(), "-", m.note)
        sys.exit(0 if m.method == METHOD_EMBEDDINGS else 1)
    print("Usage: python -m src.semantic_matcher --download")
