"""Inference service. Loads saved artifacts and serves predictions."""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.config import MODEL_DIR, TOP_K, HYBRID_ALPHA


class Recommender:
    def __init__(self):
        self.loaded = False

    def load(self):
        """Load all artifacts from disk (called once on API startup)."""
        svd_path = os.path.join(MODEL_DIR, "svd.pkl")
        art_path = os.path.join(MODEL_DIR, "artifacts.pkl")
        mov_path = os.path.join(MODEL_DIR, "movies.csv")

        if not os.path.exists(art_path):
            raise FileNotFoundError(
                f"Artifacts not found at {art_path}. Run `python train.py` first."
            )

        self.artifacts = joblib.load(art_path)
        self.svd = joblib.load(svd_path) if os.path.exists(svd_path) else None
        self.movies = pd.read_csv(mov_path)

        # Unpack
        self.mu               = self.artifacts["mu"]
        self.bu               = self.artifacts["bu"]
        self.bi               = self.artifacts["bi"]
        self.G_NORM           = self.artifacts["G_NORM"]
        self.g_index          = self.artifacts["g_index"]
        self.g_m2row          = self.artifacts["g_m2row"]
        self.tag_tfidf        = self.artifacts["tag_tfidf"]
        self.tag_ids          = self.artifacts["tag_ids"]
        self.user_high_rated  = self.artifacts["user_high_rated"]
        self.u2i              = self.artifacts["u2i"]
        self.m2i              = self.artifacts["m2i"]
        self.n_users          = self.artifacts["n_users"]
        self.n_items          = self.artifacts["n_items"]
        self.svd_type         = self.artifacts["svd_type"]

        # NumPy MF fallback
        if self.svd_type == "numpy":
            mf_path = os.path.join(MODEL_DIR, "mf_numpy.pkl")
            self.mf = joblib.load(mf_path)

        self.loaded = True

    # ---------- Core predictors ----------
    def predict_baseline(self, user_id: int, movie_id: int) -> float:
        return self.mu + self.bu.get(user_id, 0.0) + self.bi.get(movie_id, 0.0)

    def predict_svd(self, user_id: int, movie_id: int) -> float:
        if self.svd_type == "surprise" and self.svd is not None:
            return float(self.svd.predict(user_id, movie_id).est)
        # NumPy fallback
        if user_id in self.mf["u2i"] and movie_id in self.mf["m2i"]:
            ui, mi = self.mf["u2i"][user_id], self.mf["m2i"][movie_id]
            return float(
                self.mf["mu"] + self.mf["bu"][ui] + self.mf["bi"][mi]
                + np.dot(self.mf["P"][ui], self.mf["Q"][mi])
            )
        return self.mu

    def user_content_score(self, user_id: int, movie_id: int, k: int = 50) -> float:
        if user_id not in self.user_high_rated or movie_id not in self.g_m2row:
            return 0.0
        liked = self.user_high_rated[user_id][:k]
        rows = [self.g_m2row[m] for m in liked if m in self.g_m2row]
        if not rows:
            return 0.0
        profile = self.G_NORM[rows].mean(axis=0)
        profile /= (np.linalg.norm(profile) + 1e-9)
        return float(np.dot(profile, self.G_NORM[self.g_m2row[movie_id]]))

    def predict_content(self, user_id: int, movie_id: int) -> float:
        raw = self.user_content_score(user_id, movie_id)
        return float(np.clip(self.mu + 1.5 * raw, 0.5, 5.0))

    def predict_hybrid(self, user_id: int, movie_id: int,
                       alpha: float = HYBRID_ALPHA) -> float:
        cf = self.predict_svd(user_id, movie_id)
        cb = self.predict_content(user_id, movie_id)
        return float(np.clip(alpha * cf + (1 - alpha) * cb, 0.5, 5.0))

    def predict(self, user_id: int, movie_id: int, model: str = "hybrid") -> float:
        model = model.lower()
        if model == "baseline":
            return self.predict_baseline(user_id, movie_id)
        if model == "svd":
            return self.predict_svd(user_id, movie_id)
        if model == "content":
            return self.predict_content(user_id, movie_id)
        return self.predict_hybrid(user_id, movie_id)

    # ---------- Top-N ----------
    def recommend(self, user_id: int, top_n: int = TOP_K,
                  model: str = "hybrid") -> list:
        if user_id not in self.user_high_rated:
            # Unknown user → fall back to popular movies
            return self._popular_fallback(top_n)

        seen = set(self.user_high_rated.get(user_id, []))

        # Candidate pool: popular movies not seen
        popular = (
            self.movies.sort_values("movieId")
            .head(3000)["movieId"].tolist()
        )
        candidates = [m for m in popular if m not in seen]

        # Add content-similar movies
        for liked in self.user_high_rated[user_id][:5]:
            for mid, _ in self.movie_content_similarity(liked, top_n=50):
                if mid not in seen and mid not in candidates:
                    candidates.append(mid)

        scored = [(m, self.predict(user_id, m, model)) for m in candidates]
        scored.sort(key=lambda x: -x[1])
        top = scored[:top_n]

        df = pd.DataFrame(top, columns=["movieId", "predicted_rating"])
        df = df.merge(self.movies, on="movieId", how="left")
        return df[["movieId", "title", "genres", "predicted_rating"]].to_dict("records")

    def _popular_fallback(self, top_n: int) -> list:
        df = self.movies.head(top_n).copy()
        df["predicted_rating"] = self.mu
        return df[["movieId", "title", "genres", "predicted_rating"]].to_dict("records")

    # ---------- Similar movies ----------
    def movie_content_similarity(self, movie_id: int, top_n: int = 10) -> list:
        results = {}
        if movie_id in self.g_m2row:
            row = self.g_m2row[movie_id]
            g_sim = np.dot(self.G_NORM, self.G_NORM[row])
            for i, mid in enumerate(self.g_index):
                results[mid] = results.get(mid, 0.0) + 0.7 * g_sim[i]
        if movie_id in set(self.tag_ids):
            i = int(np.where(self.tag_ids == movie_id)[0][0])
            t_sim = cosine_similarity(self.tag_tfidf[i], self.tag_tfidf)[0]
            for mid, s in zip(self.tag_ids, t_sim):
                results[mid] = results.get(mid, 0.0) + 0.3 * s
        results.pop(movie_id, None)
        return sorted(results.items(), key=lambda x: -x[1])[:top_n]

    def similar_movies(self, movie_id: int, top_n: int = 10) -> list:
        sims = self.movie_content_similarity(movie_id, top_n)
        df = pd.DataFrame(sims, columns=["movieId", "similarity"])
        df = df.merge(self.movies, on="movieId", how="left")
        df["predicted_rating"] = df["similarity"]
        return df[["movieId", "title", "genres", "predicted_rating"]].to_dict("records")


# Singleton
recommender = Recommender()