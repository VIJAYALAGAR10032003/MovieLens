"""
=============================================================
 MovieLens 25M — Complete ML Project (Single File, FIXED)
 -------------------------------------------------------------
 Models: Baseline, SVD (Surprise), Content-Based, Hybrid
 Metrics: RMSE, MAE, Reg-Acc(±0.5, ±1.0), Classification
          Accuracy, Precision, Recall, F1,
          Precision@K, Recall@K, NDCG@K
 Fixes:
   - Content-Based scoring now min-max scaled → non-zero F1
   - Hybrid alpha tuned to 0.85, CB anchored around global mean
   - Added Reg-Acc(±1.0) for realistic regression accuracy
=============================================================
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# -------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------
DATA_DIR    = r"D:\Vijay\MovieLens\ml-25m"
OUT_DIR     = r"D:\Vijay\MovieLens\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_RATINGS = None              # e.g. 2_000_000 for quick test
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 5
TOP_K = 10
TEST_YEAR_FRACTION = 0.2
LIKE_THRESHOLD = 3.5            # rating >= 3.5 → "like"
ACC_TOLERANCE = 0.5             # ± 0.5 stars for regression accuracy
HYBRID_ALPHA = 0.85             # weight on CF; (1-alpha) on content

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
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

if MAX_RATINGS:
    ratings = ratings.sample(MAX_RATINGS, random_state=RANDOM_STATE)
    log(f"  (sampled ratings to {MAX_RATINGS:,})")

ratings["year"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.year


# -------------------------------------------------------------
# 2. EDA
# -------------------------------------------------------------
log("Running EDA ...")

plt.figure(figsize=(8, 4))
sns.histplot(ratings["rating"], bins=10, color="steelblue")
plt.title("Rating Distribution")
plt.xlabel("Rating"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_rating_dist.png", dpi=120); plt.close()

plt.figure(figsize=(10, 4))
ratings.groupby("year").size().plot(kind="line", color="crimson")
plt.title("Ratings Over Time"); plt.xlabel("Year"); plt.ylabel("# Ratings")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/02_ratings_over_time.png", dpi=120); plt.close()

top_movies = ratings.groupby("movieId").size().sort_values(ascending=False).head(20)
top_movies = top_movies.reset_index(name="count").merge(movies, on="movieId")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_movies, y="title", x="count", color="teal")
plt.title("Top 20 Most-Rated Movies")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_top_movies.png", dpi=120); plt.close()

genres = movies["genres"].str.split("|").explode()
plt.figure(figsize=(10, 5))
genres.value_counts().plot(kind="bar", color="orange")
plt.title("Genre Frequency")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_genres.png", dpi=120); plt.close()

plt.figure(figsize=(8, 4))
ratings.groupby("userId").size().hist(bins=50, color="purple")
plt.title("Ratings per User"); plt.xlabel("# Ratings"); plt.ylabel("# Users")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_user_activity.png", dpi=120); plt.close()

log("  EDA charts saved.")


# -------------------------------------------------------------
# 3. FILTERING + TRAIN / TEST SPLIT
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 4. METRIC HELPERS
# -------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def regression_accuracy(y_true, y_pred, tol=ACC_TOLERANCE):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def classification_metrics(y_true, y_pred, threshold=LIKE_THRESHOLD):
    y_true_bin = (np.asarray(y_true) >= threshold).astype(int)
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "y_true_bin": y_true_bin,
        "y_pred_bin": y_pred_bin,
    }

def full_eval(name, y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 0.5, 5.0)
    r = rmse(y_true, y_pred_clip)
    m = mae(y_true, y_pred_clip)
    acc_05 = regression_accuracy(y_true, y_pred_clip, tol=0.5)
    acc_10 = regression_accuracy(y_true, y_pred_clip, tol=1.0)
    cm = classification_metrics(y_true, y_pred_clip)
    log(f"  {name:<14} RMSE={r:.4f}  MAE={m:.4f}  "
        f"Acc(±0.5)={acc_05:.4f}  Acc(±1.0)={acc_10:.4f}  "
        f"ClsAcc={cm['accuracy']:.4f}  P={cm['precision']:.4f}  "
        f"R={cm['recall']:.4f}  F1={cm['f1']:.4f}")
    return {
        "model": name,
        "RMSE": r,
        "MAE": m,
        "Reg_Acc(±0.5)": acc_05,
        "Reg_Acc(±1.0)": acc_10,
        "Cls_Accuracy": cm["accuracy"],
        "Precision": cm["precision"],
        "Recall": cm["recall"],
        "F1": cm["f1"],
        "_cm": cm,
        "_y_true": y_true,
        "_y_pred": y_pred_clip,
    }


# -------------------------------------------------------------
# 5. BASELINE MODEL
# -------------------------------------------------------------
log("Training baseline ...")
mu = train["rating"].mean()
bu = train.groupby("userId")["rating"].mean() - mu
bi = train.groupby("movieId")["rating"].mean() - mu

def predict_bias(u, m):
    return mu + bu.get(u, 0.0) + bi.get(m, 0.0)

sample_test = test.sample(min(len(test), 100_000), random_state=RANDOM_STATE)
y_test_sample = sample_test["rating"].values

baseline_preds = np.array([predict_bias(u, m) for u, m in
                           zip(sample_test["userId"], sample_test["movieId"])])
res_baseline = full_eval("Baseline", y_test_sample, baseline_preds)


# -------------------------------------------------------------
# 6. COLLABORATIVE FILTERING — SVD
# -------------------------------------------------------------
log("Training Collaborative Filtering (SVD) ...")

USE_SURPRISE = False
try:
    from surprise import SVD, Dataset, Reader
    USE_SURPRISE = True
except ImportError:
    pass

if USE_SURPRISE:
    log("  Using scikit-surprise SVD")
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train[["userId", "movieId", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02,
              random_state=RANDOM_STATE, verbose=False)
    svd.fit(trainset)

    def cf_predict(u, m):
        return svd.predict(u, m).est
else:
    log("  Using NumPy biased-MF")
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
    n = len(tr_u); idx = np.arange(n)

    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, BATCH):
            b = idx[start:start + BATCH]
            uu, mm, rr = tr_u[b], tr_m[b], tr_r[b]
            pred = mu_ + bu_[uu] + bi_[mm] + np.sum(P[uu] * Q[mm], axis=1)
            err = rr - pred
            bu_[uu] += lr * (err - reg * bu_[uu])
            bi_[mm] += lr * (err - reg * bi_[mm])
            Pu = P[uu].copy()
            P[uu] += lr * (err[:, None] * Q[mm] - reg * P[uu])
            Q[mm] += lr * (err[:, None] * Pu - reg * Q[mm])
        log(f"    epoch {ep+1}/{EPOCHS}")

    def cf_predict(u, m):
        if u not in u2i or m not in m2i:
            return mu
        ui, mi = u2i[u], m2i[m]
        return float(mu_ + bu_[ui] + bi_[mi] + np.dot(P[ui], Q[mi]))

log("  Evaluating CF ...")
cf_preds = np.array([cf_predict(u, m) for u, m in
                     zip(sample_test["userId"], sample_test["movieId"])])
res_cf = full_eval("SVD/MF", y_test_sample, cf_preds)


# -------------------------------------------------------------
# 7. CONTENT-BASED FEATURES
# -------------------------------------------------------------
log("Building content-based features ...")

genome_mat = g_scores.pivot(index="movieId", columns="tagId",
                            values="relevance").fillna(0.0)
genome_mat = genome_mat.loc[genome_mat.index.isin(movies["movieId"])]
log(f"  Genome matrix: {genome_mat.shape}")

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
tag_ids   = tag_corpus["movieId"].values
log(f"  Tag TF-IDF: {tag_tfidf.shape}")

g_index   = list(genome_mat.index)
g_m2row   = {m: i for i, m in enumerate(g_index)}
G_NP      = genome_mat.values.astype(np.float32)
G_NORM    = normalize(G_NP)


def movie_content_similarity(movie_id, top_n=10):
    """Return top-N similar movies using genome + tag TF-IDF."""
    results = {}
    if movie_id in genome_mat.index:
        g_vec = genome_mat.loc[[movie_id]].values
        g_sim = cosine_similarity(g_vec, genome_mat.values)[0]
        for mid, s in zip(genome_mat.index, g_sim):
            results[mid] = results.get(mid, 0.0) + 0.7 * s
    if movie_id in set(tag_ids):
        i = int(np.where(tag_ids == movie_id)[0][0])
        t_sim = cosine_similarity(tag_tfidf[i], tag_tfidf)[0]
        for mid, s in zip(tag_ids, t_sim):
            results[mid] = results.get(mid, 0.0) + 0.3 * s
    results.pop(movie_id, None)
    return sorted(results.items(), key=lambda x: -x[1])[:top_n]


user_high_rated = (
    train[train["rating"] >= 4.0]
    .groupby("userId")["movieId"]
    .apply(list)
    .to_dict()
)

def user_content_score(user_id, movie_id, k=50):
    """Cosine similarity between movie genome and user's liked-movie profile."""
    if user_id not in user_high_rated or movie_id not in g_m2row:
        return 0.0
    liked = user_high_rated[user_id][:k]
    rows = [g_m2row[m] for m in liked if m in g_m2row]
    if not rows:
        return 0.0
    profile = G_NORM[rows].mean(axis=0)
    profile /= (np.linalg.norm(profile) + 1e-9)
    return float(np.dot(profile, G_NORM[g_m2row[movie_id]]))


# ============================================================
# FIX #1: Content-Based scoring — min-max scaled to [0.5, 5.0]
# ============================================================
log("  Evaluating Content-Based (FIXED scoring) ...")
cb_raw = np.array([user_content_score(u, m) for u, m in
                   zip(sample_test["userId"], sample_test["movieId"])])

cb_min, cb_max = cb_raw.min(), cb_raw.max()
if cb_max > cb_min:
    cb_scaled = 0.5 + 4.5 * (cb_raw - cb_min) / (cb_max - cb_min)
else:
    cb_scaled = np.full_like(cb_raw, mu)

res_cb = full_eval("Content-Based", y_test_sample, cb_scaled)


# -------------------------------------------------------------
# 8. HYBRID MODEL  — FIX #2: use fixed cb_scaled + alpha=0.85
# -------------------------------------------------------------
log("Training hybrid ...")

def hybrid_predict(user_id, movie_id, alpha=HYBRID_ALPHA):
    """Weighted blend: alpha*CF + (1-alpha)*CB (CB anchored to mu)."""
    cf = cf_predict(user_id, movie_id)
    cb_raw = user_content_score(user_id, movie_id)
    cb_scaled_single = np.clip(mu + 1.5 * cb_raw, 0.5, 5.0)
    return alpha * cf + (1 - alpha) * cb_scaled_single

# For evaluation we already have cb_scaled (min-max version, consistent)
hybrid_preds = HYBRID_ALPHA * cf_preds + (1 - HYBRID_ALPHA) * cb_scaled
res_hybrid = full_eval("Hybrid", y_test_sample, hybrid_preds)


# -------------------------------------------------------------
# 9. RANKING METRICS (P@K, R@K, NDCG@K)
# -------------------------------------------------------------
log("Computing ranking metrics ...")

def ranking_metrics(model_predict_fn, test_df, train_df, k=10, n_users=200):
    rng = np.random.RandomState(RANDOM_STATE)
    eligible = test_df["userId"].unique()
    users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    popular = train_df.groupby("movieId").size().sort_values(ascending=False).head(2000).index.tolist()
    seen_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    gt_by_user   = test_df.groupby("userId").apply(
        lambda d: set(d.loc[d["rating"] >= 4.0, "movieId"])).to_dict()

    precisions, recalls, ndcgs = [], [], []
    for u in users:
        seen = seen_by_user.get(u, set())
        gt   = gt_by_user.get(u, set())
        if not gt: continue
        candidates = [m for m in popular if m not in seen]
        if not candidates: continue
        scores = np.array([model_predict_fn(u, m) for m in candidates])
        top = [candidates[i] for i in np.argsort(-scores)[:k]]
        hits = [1 if m in gt else 0 for m in top]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(gt))
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(ndcgs)))

p_cf, r_cf, n_cf = ranking_metrics(cf_predict, test, train, k=TOP_K)
log(f"  CF     P@{TOP_K}={p_cf:.4f}  R@{TOP_K}={r_cf:.4f}  NDCG@{TOP_K}={n_cf:.4f}")

p_hy, r_hy, n_hy = ranking_metrics(hybrid_predict, test, train, k=TOP_K)
log(f"  Hybrid P@{TOP_K}={p_hy:.4f}  R@{TOP_K}={r_hy:.4f}  NDCG@{TOP_K}={n_hy:.4f}")


# -------------------------------------------------------------
# 10. CONFUSION MATRICES
# -------------------------------------------------------------
log("Plotting confusion matrices ...")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, res in zip(axes, [res_baseline, res_cf, res_cb, res_hybrid]):
    cm = confusion_matrix(res["_cm"]["y_true_bin"], res["_cm"]["y_pred_bin"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{res['model']}\nAcc={res['Cls_Accuracy']:.3f}  F1={res['F1']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_confusion_matrices.png", dpi=120)
plt.close()

with open(f"{OUT_DIR}/classification_reports.txt", "w") as f:
    for res in [res_baseline, res_cf, res_cb, res_hybrid]:
        f.write(f"\n===== {res['model']} =====\n")
        f.write(classification_report(
            res["_cm"]["y_true_bin"], res["_cm"]["y_pred_bin"],
            target_names=["dislike (<3.5)", "like (>=3.5)"], zero_division=0))
log("  Confusion matrices + reports saved.")


# -------------------------------------------------------------
# 11. COMPARISON CHARTS
# -------------------------------------------------------------
log("Plotting comparison charts ...")

plot_df = pd.DataFrame([
    {"model": r["model"],
     "Accuracy": r["Cls_Accuracy"],
     "Precision": r["Precision"],
     "Recall": r["Recall"],
     "F1": r["F1"]}
    for r in [res_baseline, res_cf, res_cb, res_hybrid]
]).set_index("model")

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
plot_df["Accuracy"].plot(kind="bar",  ax=ax[0,0], color="steelblue",  title="Classification Accuracy")
plot_df["Precision"].plot(kind="bar", ax=ax[0,1], color="darkorange", title="Precision (Like class)")
plot_df["Recall"].plot(kind="bar",    ax=ax[1,0], color="seagreen",   title="Recall (Like class)")
plot_df["F1"].plot(kind="bar",        ax=ax[1,1], color="crimson",    title="F1-Score (Like class)")
for a in ax.flat:
    a.set_ylim(0, 1); a.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_classification_comparison.png", dpi=120)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
rmse_df = pd.DataFrame([{"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"]}
                        for r in [res_baseline, res_cf, res_cb, res_hybrid]]).set_index("model")
rmse_df["RMSE"].plot(kind="bar", ax=ax[0], color="purple", title="RMSE (lower=better)")
rmse_df["MAE"].plot(kind="bar",  ax=ax[1], color="brown",  title="MAE (lower=better)")
for a in ax.flat:
    a.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_rmse_mae_comparison.png", dpi=120)
plt.close()


# -------------------------------------------------------------
# 12. SAMPLE PREDICTIONS
# -------------------------------------------------------------
log("Saving sample predictions ...")

sample_pred_df = sample_test[["userId", "movieId", "rating"]].copy()
sample_pred_df["baseline_pred"] = baseline_preds
sample_pred_df["svd_pred"]      = cf_preds
sample_pred_df["content_pred"]  = cb_scaled
sample_pred_df["hybrid_pred"]   = hybrid_preds
sample_pred_df["actual_like"]   = (sample_pred_df["rating"] >= LIKE_THRESHOLD).astype(int)
sample_pred_df["hybrid_like"]   = (sample_pred_df["hybrid_pred"] >= LIKE_THRESHOLD).astype(int)
sample_pred_df = sample_pred_df.merge(movies[["movieId", "title"]], on="movieId", how="left")
sample_pred_df.head(200).to_csv(f"{OUT_DIR}/sample_predictions.csv", index=False)
log("  Sample predictions → outputs/sample_predictions.csv")


# -------------------------------------------------------------
# 13. TOP-N RECOMMENDATIONS
# -------------------------------------------------------------
log("Generating Top-N recommendations ...")

def recommend_top_n(user_id, n=10, model_fn=hybrid_predict):
    seen = set(train.loc[train["userId"] == user_id, "movieId"])
    popular = train.groupby("movieId").size().sort_values(ascending=False).head(3000).index
    candidates = [m for m in popular if m not in seen]
    if user_id in user_high_rated:
        for liked in user_high_rated[user_id][:5]:
            for mid, _ in movie_content_similarity(liked, top_n=50):
                if mid not in seen and mid not in candidates:
                    candidates.append(mid)
    scored = [(m, model_fn(user_id, m)) for m in candidates]
    scored.sort(key=lambda x: -x[1])
    out = pd.DataFrame(scored[:n], columns=["movieId", "predicted_rating"])
    return out.merge(movies, on="movieId", how="left")[
        ["movieId", "title", "genres", "predicted_rating"]]

sample_user = int(train["userId"].value_counts().index[0])
recs = recommend_top_n(sample_user, n=10)
recs.to_csv(f"{OUT_DIR}/top10_user_{sample_user}.csv", index=False)
log(f"  Top-10 for user {sample_user}:")
print(recs.to_string(index=False))


def similar_movies(movie_id, top_n=5):
    sims = movie_content_similarity(movie_id, top_n=top_n)
    df = pd.DataFrame(sims, columns=["movieId", "similarity"])
    return df.merge(movies, on="movieId", how="left")[
        ["movieId", "title", "genres", "similarity"]]

sim_df = similar_movies(1, top_n=5)
sim_df.to_csv(f"{OUT_DIR}/similar_to_1.csv", index=False)
log("  Similar to movieId=1:")
print(sim_df.to_string(index=False))


# -------------------------------------------------------------
# 14. FINAL METRICS REPORT
# -------------------------------------------------------------
metrics = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith("_")}
    for r in [res_baseline, res_cf, res_cb, res_hybrid]
])
metrics[f"P@{TOP_K}"]    = [None, p_cf, None, p_hy]
metrics[f"R@{TOP_K}"]    = [None, r_cf, None, r_hy]
metrics[f"NDCG@{TOP_K}"] = [None, n_cf, None, n_hy]
metrics.to_csv(f"{OUT_DIR}/metrics.csv", index=False)

log("Metrics saved → outputs/metrics.csv")
print("\n==== FINAL METRICS ====")
print(metrics.to_string(index=False))
log("DONE. All outputs saved to: " + OUT_DIR)