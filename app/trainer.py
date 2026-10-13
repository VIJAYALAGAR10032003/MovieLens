"""
Training pipeline. Trains all models ONCE and saves artifacts to disk.
Run: python train.py
"""
import os
import time
import warnings
import numpy as np
import pandas as pd                       # ✅ TOP-LEVEL import (fixes UnboundLocalError)
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.config import (
    DATA_DIR, OUT_DIR, MODEL_DIR,
    MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
    TEST_YEAR_FRACTION, RANDOM_STATE,
)
from app.metrics import rmse, mae, regression_accuracy, classification_metrics

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _user_content_score(user_id, movie_id, uhr, G_NORM, g_m2row, k=50):
    """Cosine similarity between a movie's genome vector and the user's liked-movie profile."""
    if user_id not in uhr or movie_id not in g_m2row:
        return 0.0
    liked = uhr[user_id][:k]
    rows = [g_m2row[m] for m in liked if m in g_m2row]
    if not rows:
        return 0.0
    profile = G_NORM[rows].mean(axis=0)
    profile /= (np.linalg.norm(profile) + 1e-9)
    return float(np.dot(profile, G_NORM[g_m2row[movie_id]]))


def train_and_save():
    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------
    log("Loading CSVs ...")
    ratings  = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies   = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    tags     = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    links    = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    g_scores = pd.read_csv(os.path.join(DATA_DIR, "genome-scores.csv"))
    g_tags   = pd.read_csv(os.path.join(DATA_DIR, "genome-tags.csv"))

    log(f"  ratings      : {ratings.shape}")
    log(f"  movies       : {movies.shape}")
    log(f"  tags         : {tags.shape}")
    log(f"  links        : {links.shape}")
    log(f"  genome-scores: {g_scores.shape}")
    log(f"  genome-tags  : {g_tags.shape}")

    ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.year

    # ---------------------------------------------------------
    # Filter + split
    # ---------------------------------------------------------
    log("Filtering + time-based split ...")
    uc = ratings["userId"].value_counts()
    mc = ratings["movieId"].value_counts()
    ratings = ratings[ratings["userId"].isin(uc[uc >= MIN_USER_RATINGS].index)]
    ratings = ratings[ratings["movieId"].isin(mc[mc >= MIN_MOVIE_RATINGS].index)]
    log(f"  After filtering: {ratings.shape}")

    cutoff = ratings["timestamp"].quantile(1 - TEST_YEAR_FRACTION)
    train = ratings[ratings["timestamp"] < cutoff].copy()
    test  = ratings[ratings["timestamp"] >= cutoff].copy()

    train_users = set(train["userId"].unique())
    train_items = set(train["movieId"].unique())
    test = test[test["userId"].isin(train_users) & test["movieId"].isin(train_items)]
    log(f"  Train: {len(train):,}   Test: {len(test):,}")

    user_ids = sorted(train["userId"].unique())
    item_ids = sorted(train["movieId"].unique())
    u2i = {u: i for i, u in enumerate(user_ids)}
    i2u = {i: u for u, i in u2i.items()}
    m2i = {m: i for i, m in enumerate(item_ids)}
    i2m = {i: m for m, i in m2i.items()}
    n_users, n_items = len(user_ids), len(item_ids)
    log(f"  n_users={n_users:,}  n_items={n_items:,}")

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------
    log("Training baseline ...")
    mu = float(train["rating"].mean())
    bu = (train.groupby("userId")["rating"].mean() - mu).to_dict()
    bi = (train.groupby("movieId")["rating"].mean() - mu).to_dict()

    # ---------------------------------------------------------
    # SVD (Surprise) or NumPy MF fallback
    # ---------------------------------------------------------
    log("Training Collaborative Filtering (SVD) ...")

    use_surprise = False
    try:
        from surprise import SVD, Dataset, Reader
        use_surprise = True
    except ImportError:
        pass

    svd_model = None
    mf_numpy  = None

    if use_surprise:
        log("  Using scikit-surprise SVD")
        reader = Reader(rating_scale=(0.5, 5.0))
        train_data = Dataset.load_from_df(
            train[["userId", "movieId", "rating"]], reader
        )
        trainset = train_data.build_full_trainset()
        svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02,
                        random_state=RANDOM_STATE, verbose=False)
        svd_model.fit(trainset)
        joblib.dump(svd_model, os.path.join(MODEL_DIR, "svd.pkl"))
        svd_type = "surprise"
    else:
        log("  surprise not installed → NumPy biased-MF")
        tr_u = train["userId"].map(u2i).values
        tr_m = train["movieId"].map(m2i).values
        tr_r = train["rating"].values.astype(np.float32)

        K = 32
        P = np.random.normal(0, 0.1, (n_users, K)).astype(np.float32)
        Q = np.random.normal(0, 0.1, (n_items, K)).astype(np.float32)
        bu_ = np.zeros(n_users, dtype=np.float32)
        bi_ = np.zeros(n_items, dtype=np.float32)
        mu_ = float(mu)

        lr, reg, EPOCHS, BATCH = 0.005, 0.02, 15, 8192
        n = len(tr_u)
        idx = np.arange(n)

        for ep in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, n, BATCH):
                b = idx[s:s + BATCH]
                uu, mm, rr = tr_u[b], tr_m[b], tr_r[b]
                pred = mu_ + bu_[uu] + bi_[mm] + np.sum(P[uu] * Q[mm], axis=1)
                err = rr - pred
                bu_[uu] += lr * (err - reg * bu_[uu])
                bi_[mm] += lr * (err - reg * bi_[mm])
                Pu = P[uu].copy()
                P[uu] += lr * (err[:, None] * Q[mm] - reg * P[uu])
                Q[mm] += lr * (err[:, None] * Pu - reg * Q[mm])
            log(f"    epoch {ep+1}/{EPOCHS}")

        mf_numpy = {
            "P": P, "Q": Q, "bu": bu_, "bi": bi_, "mu": mu_,
            "u2i": u2i, "m2i": m2i,
        }
        joblib.dump(mf_numpy, os.path.join(MODEL_DIR, "mf_numpy.pkl"))
        svd_type = "numpy"

    # ---------------------------------------------------------
    # Content-Based features
    # ---------------------------------------------------------
    log("Building content-based features ...")

    genome_mat = g_scores.pivot(index="movieId", columns="tagId",
                                values="relevance").fillna(0.0)
    genome_mat = genome_mat.loc[genome_mat.index.isin(movies["movieId"])]
    log(f"  Genome matrix: {genome_mat.shape}")

    G_NP   = genome_mat.values.astype(np.float32)
    G_NORM = normalize(G_NP)
    g_index = list(genome_mat.index)
    g_m2row = {m: i for i, m in enumerate(g_index)}

    tag_corpus = (
        tags.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x.astype(str)))
            .rename("tag_text")
            .reset_index()
    )
    vec = TfidfVectorizer(max_features=2000, stop_words="english",
                          token_pattern=r"(?u)\b\w[\w'-]+\b")
    tag_tfidf = vec.fit_transform(tag_corpus["tag_text"])
    tag_ids = tag_corpus["movieId"].values
    log(f"  Tag TF-IDF: {tag_tfidf.shape}")

    user_high_rated = (
        train[train["rating"] >= 4.0]
        .groupby("userId")["movieId"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Save all artifacts
    # ---------------------------------------------------------
    joblib.dump(
        {
            "mu": mu, "bu": bu, "bi": bi,
            "G_NP": G_NP, "G_NORM": G_NORM,
            "g_index": g_index, "g_m2row": g_m2row,
            "tag_tfidf": tag_tfidf, "tag_ids": tag_ids,
            "user_high_rated": user_high_rated,
            "u2i": u2i, "m2i": m2i,
            "n_users": n_users, "n_items": n_items,
            "svd_type": svd_type,
        },
        os.path.join(MODEL_DIR, "artifacts.pkl"),
        compress=3,
    )
    movies.to_csv(os.path.join(MODEL_DIR, "movies.csv"), index=False)
    log(f"  Artifacts saved → {MODEL_DIR}")

    # ---------------------------------------------------------
    # Quick evaluation on a 10k sample
    # ---------------------------------------------------------
    log("Quick evaluation on 10k test sample ...")
    sample = test.sample(min(10_000, len(test)), random_state=RANDOM_STATE)
    y_true = sample["rating"].values

    # Baseline preds
    base = np.array([mu + bu.get(u, 0.0) + bi.get(m, 0.0)
                     for u, m in zip(sample["userId"], sample["movieId"])])

    # CF preds
    if svd_type == "surprise":
        cf = np.array([svd_model.predict(u, m).est
                       for u, m in zip(sample["userId"], sample["movieId"])])
    else:
        P_   = mf_numpy["P"]
        Q_   = mf_numpy["Q"]
        bu_  = mf_numpy["bu"]
        bi_  = mf_numpy["bi"]
        mu_  = mf_numpy["mu"]
        u2i_ = mf_numpy["u2i"]
        m2i_ = mf_numpy["m2i"]
        cf = []
        for u, m in zip(sample["userId"], sample["movieId"]):
            if u in u2i_ and m in m2i_:
                ui, mi = u2i_[u], m2i_[m]
                cf.append(mu_ + bu_[ui] + bi_[mi] + float(np.dot(P_[ui], Q_[mi])))
            else:
                cf.append(mu_)
        cf = np.array(cf)

    # Content preds (min-max scaled)
    cb_raw = np.array([
        _user_content_score(u, m, user_high_rated, G_NORM, g_m2row)
        for u, m in zip(sample["userId"], sample["movieId"])
    ])
    cb_min, cb_max = cb_raw.min(), cb_raw.max()
    if cb_max > cb_min:
        cb_scaled = 0.5 + 4.5 * (cb_raw - cb_min) / (cb_max - cb_min)
    else:
        cb_scaled = np.full_like(cb_raw, mu)

    # Hybrid (alpha=0.85)
    hy = 0.85 * cf + 0.15 * cb_scaled

    # Metrics table
    rows = []
    for name, preds in [("Baseline", base), ("SVD", cf),
                        ("Content", cb_scaled), ("Hybrid", hy)]:
        p = np.clip(preds, 0.5, 5.0)
        cm = classification_metrics(y_true, p)
        rows.append({
            "model":         name,
            "RMSE":          round(rmse(y_true, p), 4),
            "MAE":           round(mae(y_true, p), 4),
            "Reg_Acc_0.5":   round(regression_accuracy(y_true, p, 0.5), 4),
            "Reg_Acc_1.0":   round(regression_accuracy(y_true, p, 1.0), 4),
            "Cls_Accuracy":  round(cm["accuracy"], 4),
            "Precision":     round(cm["precision"], 4),
            "Recall":        round(cm["recall"], 4),
            "F1":            round(cm["f1"], 4),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print("\n==== METRICS ====")
    print(metrics_df.to_string(index=False))

    log("Training complete.")


if __name__ == "__main__":
    train_and_save()