import numpy as np
import pytest

from src.semantic_matcher import METHOD_EMBEDDINGS, METHOD_TFIDF, SemanticMatcher, reset_model_cache, tfidf_analyzer


class FakeEncoder:
    """Deterministic stand-in for a Sentence Transformer (tests the code path, NOT model quality)."""
    VOCAB = ["python", "sql", "docker", "model", "team"]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = np.array([[t.lower().count(w) for w in self.VOCAB] + [0.1] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_embedding_path_with_injected_encoder():
    m = SemanticMatcher("auto", loader=lambda name: FakeEncoder())
    sim = m.similarity(["python scripts"], ["I write python code", "We eat lunch"])
    assert m.method == METHOD_EMBEDDINGS and sim.shape == (1, 2)
    assert sim[0, 0] > sim[0, 1]
    assert np.all(sim <= 1.0) and np.all(sim >= -1.0)


def test_fallback_when_model_cannot_load():
    def broken(name):
        raise OSError("no network")
    m = SemanticMatcher("auto", loader=broken)
    sim = m.similarity(["docker containers"], ["Packaged apps with Docker containers", "Played football"])
    assert m.method == METHOD_TFIDF and "could not be loaded" in m.note and "OSError" in m.note
    assert sim[0, 0] > sim[0, 1]


def test_fallback_on_runtime_encoding_failure():
    class Boom:
        def encode(self, *a, **k):
            raise RuntimeError("boom")
    m = SemanticMatcher("auto", loader=lambda n: Boom())
    m.similarity(["a skill"], ["a sentence"])
    assert m.method == METHOD_TFIDF and "failed" in m.note


def test_missing_sentence_transformers_package_falls_back(monkeypatch):
    reset_model_cache()
    import src.semantic_matcher as sm
    monkeypatch.setattr(sm, "_default_loader", lambda n: (_ for _ in ()).throw(ImportError("no module")))
    m = SemanticMatcher("embeddings", model_name="definitely-not-a-model")
    assert m.load() == METHOD_TFIDF
    reset_model_cache()


def test_tfidf_preference_skips_model():
    m = SemanticMatcher("tfidf", loader=lambda n: pytest.fail("loader must not be called"))
    assert m.load() == METHOD_TFIDF


def test_invalid_preference_and_empty_inputs():
    with pytest.raises(ValueError):
        SemanticMatcher("magic")
    m = SemanticMatcher("tfidf")
    assert m.similarity([], ["x"]).shape == (0, 1)
    assert m.similarity(["x"], []).shape == (1, 0)


def test_tfidf_alias_tokens_align():
    assert tfidf_analyzer("used sklearn") == tfidf_analyzer("used scikit-learn")
