"""Giai đoạn 3: phân nhóm kiểu nhu cầu (RQ1): quy tắc Syntetos-Boylan ("nhóm") so với K-means ("cụm").

Chạy:  python scripts/03_cluster.py

Đọc clean/vn1_long.parquet, ghi:
  clean/clusters/assignments.parquet        unique_id, origin, eligibility, label_A, label_B
  clean/clusters/features_origin{1..4}.parquet
  reports/cluster/*.png, cluster_numbers.json, cluster_summary.md

Quy tắc chống rò rỉ: đặc trưng và nhãn tại mốc o chỉ dùng các tuần < o.
Bốn mốc: sau tuần 131, 144, 157, 170 (mỗi mốc dự báo 13 tuần tiếp theo). Mô tả chính dùng mốc 1 (hết 2023-01-02).
K-means được khớp MỘT lần ở mốc 1; các mốc sau gán theo tâm cụm gần nhất với cùng bộ chuẩn hóa,
để định nghĩa cụm không đổi theo thời gian.

Chuỗi đủ điều kiện tại mốc: >= 26 tuần kể từ lần bán đầu, có bán trong 26 tuần gần nhất, có giá hợp lệ.
Chuỗi khác nhận nhãn cố định: new (dưới 26 tuần lịch sử, gồm cả chưa ra mắt), dormant, no_price.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "clean")
OUT_DATA = os.path.join(CLEAN, "clusters")
OUT = os.path.join(ROOT, "reports", "cluster")
ORIGINS = [131, 144, 157, 170]
K = 5                      # chọn sau khi xem bảng silhouette / Davies-Bouldin và hồ sơ cụm (xem báo cáo)
SEED = 0
FEATURES = ["zero_share", "log_adi", "log_cv2", "log_mean_nz", "log_rev52", "seasonal_strength", "trend",
            "log_recency", "log_price", "price_cv", "q4_share"]
FEATURE_VI = {"zero_share": "tỷ lệ tuần bằng 0", "log_adi": "ADI (log)", "log_cv2": "CV² (log)",
              "log_mean_nz": "lượng bán khi có bán (log)", "log_rev52": "doanh thu tuần, 52 tuần cuối (log)",
              "seasonal_strength": "độ mạnh mùa vụ", "trend": "xu hướng (log tỷ lệ)", "log_recency": "số tuần ngừng bán (log)",
              "log_price": "mức giá (log)", "price_cv": "biến động giá (CV)", "q4_share": "tỷ trọng doanh thu quý 4"}
SB = ["smooth", "erratic", "intermittent", "lumpy"]

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY, INK, MUTED, GRID, SURF = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlesize": 11, "figure.dpi": 130, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def features_at(o, D):
    """Đặc trưng của mọi chuỗi tại mốc o, chỉ từ các tuần < o."""
    Q, R, Praw, first_idx, month = D["Q"][:, :o], D["R"][:, :o], D["Praw"][:, :o], D["first_idx"], D["month"][:o]
    n = len(Q)
    exists = first_idx < o
    age = np.where(exists, o - first_idx, 0)
    active = np.arange(o)[None, :] >= first_idx[:, None]
    nz = Q > 0
    n_nz = nz.sum(axis=1)
    last_sale = np.where(n_nz > 0, o - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    zero_run = np.where(exists, np.minimum(o - 1 - last_sale, age), 0)
    price_ok = (Praw > 0) & nz
    has_price = price_ok.any(axis=1)
    elig = np.where(~exists | (age < 26), "new", np.where(zero_run >= 26, "dormant", np.where(~has_price, "no_price", "eligible")))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_nz = Q.sum(axis=1) / n_nz
        cv2 = ((np.where(nz, Q, 0) ** 2).sum(axis=1) / n_nz - mean_nz ** 2) / mean_nz ** 2
        adi = age / n_nz
        zero_share = 1 - n_nz / age
        w52 = np.minimum(age, 52)
        rev52 = R[:, o - 52:].sum(axis=1) / np.maximum(w52, 1)
        prev_n = age - 52
        rev_prev = (R.sum(axis=1) - R[:, o - 52:].sum(axis=1)) / np.maximum(prev_n, 1)
        trend = np.where(prev_n >= 13, np.log((rev52 + 1) / (rev_prev + 1)), np.nan)
        # Độ mạnh mùa vụ: phần phương sai của log1p(doanh thu tuần) do trung bình theo tháng giải thích (cần >= 52 tuần)
        L = np.where(active, np.log1p(R), np.nan)
        tot_mean = np.nanmean(L, axis=1)
        ss_tot = np.nansum((L - tot_mean[:, None]) ** 2, axis=1)
        ss_between = np.zeros(n)
        for m in range(1, 13):
            cols = month == m
            cnt = active[:, cols].sum(axis=1)
            mm = np.where(cnt > 0, np.nansum(L[:, cols], axis=1) / np.maximum(cnt, 1), tot_mean)
            ss_between += cnt * (mm - tot_mean) ** 2
        seasonal = np.where((age >= 52) & (ss_tot > 0), ss_between / ss_tot, np.nan)
        q4 = np.isin(D["month"][o - 52:o], [10, 11, 12])
        r52 = R[:, o - 52:]
        q4_share = np.where((age >= 52) & (r52.sum(axis=1) > 0), r52[:, q4].sum(axis=1) / r52.sum(axis=1), np.nan)
        pm = np.where(price_ok, Praw, np.nan)
        price_med = np.nanmedian(pm, axis=1)
        price_cv = np.nanstd(pm, axis=1) / np.nanmean(pm, axis=1)
    sb = np.where(adi < 1.32, np.where(cv2 < 0.49, 0, 1), np.where(cv2 < 0.49, 2, 3))
    f = pd.DataFrame({
        "unique_id": D["ids"], "origin": o, "origin_week": D["weeks"][o - 1], "eligibility": elig, "age": age, "zero_run": zero_run,
        "adi": adi, "cv2": cv2, "mean_nz": mean_nz, "rev52": rev52, "price_median": price_med,
        "zero_share": zero_share, "log_adi": np.log(adi), "log_cv2": np.log1p(cv2), "log_mean_nz": np.log(mean_nz),
        "log_rev52": np.log1p(rev52), "seasonal_strength": seasonal, "trend": trend, "log_recency": np.log1p(zero_run),
        "log_price": np.log(price_med), "price_cv": price_cv, "q4_share": q4_share,
        "label_A": np.where(elig == "eligible", np.array(SB)[sb], elig),
    })
    return f


class Scaler:
    """Điền thiếu bằng trung vị, cắt ở phân vị 1-99, chuẩn hóa z; các tham số lấy từ mốc 1."""
    def fit(self, X):
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.lo, self.hi = Xf.quantile(0.01), Xf.quantile(0.99)
        Xc = Xf.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = Xc.mean(), Xc.std().replace(0, 1)
        return self

    def transform(self, X):
        return ((X.fillna(self.med).clip(self.lo, self.hi, axis=1) - self.mu) / self.sd).to_numpy()


def name_clusters(profile):
    """Đặt tên tiếng Việt theo hồ sơ trung vị của cụm (không phụ thuộc số thứ tự cụm)."""
    names = {}
    for c, r in profile.iterrows():
        if r["price_cv"] >= 0.5:
            base = "Giá biến động bất thường"
        elif r["q4_share"] >= 0.40:
            base = "Mùa vụ cuối năm"
        elif r["price_median"] >= 5 * profile["price_median"].median():
            base = "Sản phẩm giá cao"
        elif r["zero_share"] >= 0.6:
            base = "Bán thưa, lượng nhỏ"
        elif r["zero_share"] >= 0.25:
            base = "Bán ngắt quãng, quy mô nhỏ"
        else:
            base = "Chủ lực bán đều"
        names[c] = base
    # tránh trùng tên
    seen = {}
    for c in sorted(names, key=lambda c: -profile.loc[c, "rev52"]):
        b = names[c]
        seen[b] = seen.get(b, 0) + 1
        if seen[b] > 1:
            names[c] = f"{b} ({seen[b]})"
    return names


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)
    df = pd.read_parquet(os.path.join(CLEAN, "vn1_long.parquet"))
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    all_ids = np.sort(df["unique_id"].unique())
    n_flat_rows, n_flat_series = int(df["is_flat_period"].sum()), int(df.loc[df["is_flat_period"], "unique_id"].nunique())
    df = df[~df["is_flat_period"]]      # cắt lịch sử: bỏ giai đoạn phẳng, lịch sử bắt đầu từ history_start_week
    R = df.pivot(index="unique_id", columns="week", values="revenue").reindex(index=all_ids, columns=weeks)
    ids = R.index
    piv = lambda col: df.pivot(index="unique_id", columns="week", values=col).reindex(index=ids, columns=weeks)
    first = df.groupby("unique_id")["history_start_week"].first().loc[ids]
    D = {"R": R.fillna(0.0).to_numpy(), "Q": piv("sales").fillna(0.0).to_numpy(), "Praw": piv("price_raw").to_numpy(),
         "first_idx": np.searchsorted(weeks.values, first.values), "month": weeks.month.to_numpy(), "weeks": weeks, "ids": ids.to_numpy()}
    NUM = {"origins": {i + 1: {"index": o, "last_week_used": str(weeks[o - 1].date()), "forecast_from": str(weeks[o].date()) if o < len(weeks) else None}
                       for i, o in enumerate(ORIGINS)}}

    NUM["flat_period_cut"] = {"rows_removed_from_history": n_flat_rows, "series_affected": n_flat_series}
    feats = {i + 1: features_at(o, D) for i, o in enumerate(ORIGINS)}
    f1 = feats[1]
    e1 = f1[f1["eligibility"] == "eligible"].reset_index(drop=True)
    scaler = Scaler().fit(e1[FEATURES])
    X1 = scaler.transform(e1[FEATURES])
    NUM["imputed_share_origin1"] = {c: float(e1[c].isna().mean()) for c in FEATURES if e1[c].isna().any()}

    # ---- Chọn k ----
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(X1), min(6000, len(X1)), replace=False)
    ksel = []
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X1)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X1)
        ksel.append({"k": k, "kmeans_silhouette": float(silhouette_score(X1[samp], km.labels_[samp])),
                     "kmeans_davies_bouldin": float(davies_bouldin_score(X1, km.labels_)),
                     "kmeans_min_cluster_share": float(np.bincount(km.labels_).min() / len(X1)),
                     "ward_silhouette": float(silhouette_score(X1[samp], ward.labels_[samp])),
                     "ward_davies_bouldin": float(davies_bouldin_score(X1, ward.labels_)),
                     "ari_kmeans_vs_ward": float(adjusted_rand_score(km.labels_, ward.labels_))})
    NUM["k_selection"] = ksel
    sb_codes = pd.Categorical(e1["label_A"], categories=SB).codes
    NUM["sb_silhouette_in_feature_space"] = float(silhouette_score(X1[samp], sb_codes[samp]))
    NUM["sb_davies_bouldin"] = float(davies_bouldin_score(X1, sb_codes))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ks = [r["k"] for r in ksel]
    for ax, key, title in ((axes[0], "silhouette", "Silhouette (cao hơn là tốt hơn)"), (axes[1], "davies_bouldin", "Davies-Bouldin (thấp hơn là tốt hơn)")):
        ax.plot(ks, [r[f"kmeans_{key}"] for r in ksel], color=BLUE, marker="o", label="K-means")
        ax.plot(ks, [r[f"ward_{key}"] for r in ksel], color=ORANGE, marker="o", label="Ward")
        ax.axvline(K, color=GRAY, lw=1, ls="--"); ax.set_title(title); ax.set_xlabel("số cụm k"); ax.legend()
    save(fig, "fig01_chon_k.png")

    # ---- Mô hình cuối ----
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X1)
    raw_lab = km.labels_
    rev_rank = pd.Series(e1["rev52"].to_numpy()).groupby(raw_lab).sum().sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(rev_rank)}      # B1 = cụm doanh thu lớn nhất
    centers = km.cluster_centers_[rev_rank]
    assign = lambda X: np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    e1["cluster"] = np.vectorize(remap.get)(raw_lab)
    prof_cols = ["zero_share", "adi", "cv2", "mean_nz", "rev52", "seasonal_strength", "trend", "zero_run", "price_median", "price_cv", "q4_share", "age"]
    profile = e1.groupby("cluster")[prof_cols].median()
    names = name_clusters(profile)
    code = {c: f"B{c + 1}" for c in range(K)}
    full = {c: f"{code[c]} {names[c]}" for c in range(K)}
    size = e1.groupby("cluster").size()
    rev = e1.groupby("cluster")["rev52"].sum()
    NUM["clusters_origin1"] = {full[c]: {"n_series": int(size[c]), "share_series": float(size[c] / len(e1)),
                                          "share_revenue_last52": float(rev[c] / rev.sum()),
                                          "median": {k: float(v) for k, v in profile.loc[c].items()}} for c in range(K)}
    NUM["final_k"] = K
    NUM["final_silhouette"] = float(silhouette_score(X1[samp], e1["cluster"].to_numpy()[samp]))

    # ---- Gán nhãn cho cả 4 mốc ----
    out = []
    for i, f in feats.items():
        el = f["eligibility"] == "eligible"
        lab = pd.Series(f["eligibility"].to_numpy(), index=f.index, dtype=object)
        lab[el] = [full[c] for c in assign(scaler.transform(f.loc[el, FEATURES]))]
        f["label_B"] = lab
        # Hai cờ bổ sung cho phương án "A tinh chỉnh" (đề xuất dùng khi mô hình hóa)
        f["flag_q4_seasonal"] = el & (f["q4_share"] >= 0.40)
        f["flag_price_noisy"] = el & (f["price_cv"] >= 0.5)
        f.to_parquet(os.path.join(OUT_DATA, f"features_origin{i}.parquet"), index=False)
        out.append(f[["unique_id", "origin", "origin_week", "eligibility", "label_A", "label_B", "flag_q4_seasonal", "flag_price_noisy"]].assign(origin_no=i))
    assignments = pd.concat(out, ignore_index=True)
    assignments.to_parquet(os.path.join(OUT_DATA, "assignments.parquet"), index=False)
    f1 = feats[1]
    NUM["flags_by_origin"] = {i: {"q4_seasonal": int(f["flag_q4_seasonal"].sum()), "price_noisy": int(f["flag_price_noisy"].sum())} for i, f in feats.items()}
    NUM["flag_q4_by_label_A_origin1"] = {a: int((f1["flag_q4_seasonal"] & (f1["label_A"] == a)).sum()) for a in SB}
    NUM["eligibility_by_origin"] = {i: {k: int(v) for k, v in f["eligibility"].value_counts().items()} for i, f in feats.items()}
    o1 = ORIGINS[0]
    fut = D["R"][:, o1:o1 + 52].sum(axis=1)
    NUM["test_revenue_share_by_eligibility_origin1"] = {k: float(fut[(f1["eligibility"] == k).to_numpy()].sum() / fut.sum()) for k in ("eligible", "new", "dormant", "no_price")}

    # ---- A so với B ----
    el1 = f1[f1["eligibility"] == "eligible"]
    ct = pd.crosstab(el1["label_A"], el1["label_B"]).reindex(index=SB, columns=[full[c] for c in range(K)], fill_value=0)
    NUM["crosstab_A_vs_B_origin1"] = {a: {b: int(v) for b, v in row.items()} for a, row in ct.iterrows()}
    NUM["ari_A_vs_B_origin1"] = float(adjusted_rand_score(el1["label_A"], el1["label_B"]))
    fut_s = pd.Series(fut, index=f1["unique_id"].to_numpy())
    el_fut = fut_s.loc[el1["unique_id"]].to_numpy()
    NUM["A_origin1"] = {a: {"n_series": int((el1["label_A"] == a).sum()), "share_series": float((el1["label_A"] == a).mean()),
                            "share_revenue_last52": float(el1.loc[el1["label_A"] == a, "rev52"].sum() / el1["rev52"].sum()),
                            "share_test_revenue": float(el_fut[(el1["label_A"] == a).to_numpy()].sum() / el_fut.sum())} for a in SB}
    for c in range(K):
        NUM["clusters_origin1"][full[c]]["share_test_revenue"] = float(el_fut[(el1["label_B"] == full[c]).to_numpy()].sum() / el_fut.sum())
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    share = ct.div(ct.sum(axis=1), axis=0).to_numpy()
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K), [full[c].replace(" ", "\n", 1) for c in range(K)], fontsize=8); ax.set_yticks(range(4), SB); ax.grid(False)
    for i in range(4):
        for j in range(K):
            ax.text(j, i, f"{ct.iloc[i, j]:,}".replace(",", "."), ha="center", va="center", color="white" if share[i, j] > 0.5 else INK, fontsize=9)
    ax.set_title("Syntetos-Boylan (hàng) so với K-means (cột), số chuỗi tại mốc 1")
    save(fig, "fig04_bang_cheo_A_B.png")

    # ---- Độ ổn định ----
    stab = {}
    for i in range(1, 4):
        a, b = feats[i].set_index("unique_id"), feats[i + 1].set_index("unique_id")
        both = a.index[(a["eligibility"] == "eligible") & (b.loc[a.index, "eligibility"] == "eligible")]
        stab[f"{i}->{i + 1}"] = {
            "n_series_eligible_in_both": int(len(both)),
            "ari_A": float(adjusted_rand_score(a.loc[both, "label_A"], b.loc[both, "label_A"])),
            "ari_B": float(adjusted_rand_score(a.loc[both, "label_B"], b.loc[both, "label_B"])),
            "switch_share_A": float((a.loc[both, "label_A"] != b.loc[both, "label_A"]).mean()),
            "switch_share_B": float((a.loc[both, "label_B"] != b.loc[both, "label_B"]).mean()),
            "eligible_then_dormant": int(((a["eligibility"] == "eligible") & (b.loc[a.index, "eligibility"] == "dormant")).sum()),
            "new_then_eligible": int(((a["eligibility"] == "new") & (b.loc[a.index, "eligibility"] == "eligible")).sum()),
        }
    NUM["stability"] = stab
    a, b = feats[1].set_index("unique_id"), feats[2].set_index("unique_id")
    both = a.index[(a["eligibility"] == "eligible") & (b.loc[a.index, "eligibility"] == "eligible")]
    for key, labels in (("A", SB), ("B", [full[c] for c in range(K)])):
        tr = pd.crosstab(a.loc[both, f"label_{key}"], b.loc[both, f"label_{key}"], normalize="index").reindex(index=labels, columns=labels, fill_value=0)
        NUM[f"stay_rate_{key}_1->2"] = {l: float(tr.loc[l, l]) for l in labels}

    # ---- Hồ sơ cụm ----
    Z = pd.DataFrame(X1, columns=FEATURES).groupby(e1["cluster"]).mean()
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.imshow(Z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(FEATURES)), [FEATURE_VI[c] for c in FEATURES], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(K), [full[c] for c in range(K)], fontsize=9)
    for i in range(K):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{Z.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(Z.iloc[i, j]) > 1.2 else INK)
    ax.set_title("Hồ sơ cụm (z-score: đỏ = cao, xanh = thấp)")
    save(fig, "fig03_ho_so_cum.png")

    # ---- PCA ----
    pca = PCA(n_components=2, random_state=SEED).fit(X1)
    Y = pca.transform(X1)
    NUM["pca_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    NUM["pca_loadings"] = {f"PC{i + 1}": {FEATURES[j]: float(pca.components_[i, j]) for j in range(len(FEATURES))} for i in range(2)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for c in range(K):
        m = (e1["cluster"] == c).to_numpy()
        axes[0].scatter(Y[m, 0], Y[m, 1], s=4, color=PALETTE[c], alpha=0.35, lw=0, label=full[c])
    for j, a_ in enumerate(SB):
        m = (e1["label_A"] == a_).to_numpy()
        axes[1].scatter(Y[m, 0], Y[m, 1], s=4, color=PALETTE[j], alpha=0.35, lw=0, label=a_)
    for ax, t in zip(axes, ("Tô màu theo K-means", "Tô màu theo Syntetos-Boylan")):
        ax.set_title(t); ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} phương sai)")
        lg = ax.legend(markerscale=4, fontsize=8, loc="best")
        for h in lg.legend_handles:
            h.set_alpha(1)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    save(fig, "fig02_pca.png")

    # ---- Ví dụ từng cụm ----
    id_pos = pd.Series(np.arange(len(ids)), index=ids)
    fig, axes = plt.subplots(K, 7, figsize=(17, 2.1 * K), gridspec_kw={"width_ratios": [1.6] + [1] * 6})
    fig.suptitle("Cột đầu: tổng doanh thu tuần của cụm (triệu). Sáu cột sau: số lượng bán hàng tuần của 6 chuỗi ngẫu nhiên.", x=0.01, ha="left", fontsize=11)
    x = weeks[:o1]
    for c in range(K):
        members = e1.loc[e1["cluster"] == c, "unique_id"].to_numpy()
        pos = id_pos.loc[members].to_numpy()
        ax = axes[c, 0]
        ax.plot(x, D["R"][pos, :o1].sum(axis=0) / 1e6, color=PALETTE[c], lw=1.6)
        ax.set_title(full[c], fontsize=9); ax.set_ylim(0); ax.tick_params(labelsize=7); ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        for j, p in enumerate(rng.choice(pos, 6, replace=False)):
            ax = axes[c, j + 1]
            ax.plot(x, D["Q"][p, :o1], color=PALETTE[c], lw=1)
            ax.set_title(str(ids[p]), fontsize=8, fontweight="normal"); ax.tick_params(labelsize=6); ax.set_xticks([]); ax.set_ylim(0)
    save(fig, "fig05_vi_du_theo_cum.png")

    # ---- Ổn định: hình ----
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs = np.arange(3)
    ax.bar(xs - 0.2, [stab[k]["switch_share_A"] * 100 for k in stab], width=0.38, color=BLUE, label="Syntetos-Boylan")
    ax.bar(xs + 0.2, [stab[k]["switch_share_B"] * 100 for k in stab], width=0.38, color=ORANGE, label="K-means")
    ax.set_xticks(xs, [f"mốc {k.replace('->', ' sang ')}" for k in stab]); ax.set_ylabel("% chuỗi đổi nhãn sau 13 tuần"); ax.legend()
    ax.set_title("Độ ổn định của nhãn giữa hai mốc liên tiếp")
    save(fig, "fig06_on_dinh.png")

    with open(os.path.join(OUT, "cluster_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(NUM, f, ensure_ascii=False, indent=1)

    pct = lambda v: f"{v * 100:.1f}%".replace(".", ",")
    L = ["# Tóm tắt phân nhóm kiểu nhu cầu (VN1)", "",
         f"Mốc 1 dùng dữ liệu đến {NUM['origins'][1]['last_week_used']}. Chuỗi đủ điều kiện: {len(e1)}; "
         f"new: {NUM['eligibility_by_origin'][1].get('new', 0)}; dormant: {NUM['eligibility_by_origin'][1].get('dormant', 0)}; "
         f"no_price: {NUM['eligibility_by_origin'][1].get('no_price', 0)}.", "",
         "## Chọn k", "", "| k | Silhouette K-means | Davies-Bouldin K-means | Cụm nhỏ nhất | Silhouette Ward | ARI K-means/Ward |", "|---:|---:|---:|---:|---:|---:|"]
    for r in ksel:
        L.append(f"| {r['k']} | {r['kmeans_silhouette']:.3f} | {r['kmeans_davies_bouldin']:.2f} | {pct(r['kmeans_min_cluster_share'])} | {r['ward_silhouette']:.3f} | {r['ari_kmeans_vs_ward']:.2f} |")
    L += ["", f"Chọn k = {K}. Silhouette của nhãn Syntetos-Boylan trong cùng không gian đặc trưng: {NUM['sb_silhouette_in_feature_space']:.3f}.", "",
          "## Cụm K-means tại mốc 1", "", "| Cụm | Số chuỗi | % chuỗi | % doanh thu 52 tuần cuối | % tuần bằng 0 | ADI | CV² | Doanh thu tuần | Giá | Tỷ trọng Q4 | Xu hướng |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in range(K):
        d = NUM["clusters_origin1"][full[c]]; m = d["median"]
        L.append(f"| {full[c]} | {d['n_series']} | {pct(d['share_series'])} | {pct(d['share_revenue_last52'])} | {pct(m['zero_share'])} | {m['adi']:.2f} | {m['cv2']:.2f} | {m['rev52']:.0f} | {m['price_median']:.1f} | {pct(m['q4_share'])} | {m['trend']:+.2f} |")
    L += ["", "## Syntetos-Boylan tại mốc 1", "", "| Nhóm | Số chuỗi | % chuỗi | % doanh thu 52 tuần cuối |", "|---|---:|---:|---:|"]
    for a_ in SB:
        d = NUM["A_origin1"][a_]
        L.append(f"| {a_} | {d['n_series']} | {pct(d['share_series'])} | {pct(d['share_revenue_last52'])} |")
    L += ["", f"ARI giữa hai cách chia: {NUM['ari_A_vs_B_origin1']:.2f}.", "", "## Độ ổn định giữa các mốc", "",
          "| Cặp mốc | Chuỗi đủ điều kiện ở cả hai | % đổi nhãn A | % đổi nhãn B | ARI A | ARI B |", "|---|---:|---:|---:|---:|---:|"]
    for k, s in stab.items():
        L.append(f"| {k} | {s['n_series_eligible_in_both']} | {pct(s['switch_share_A'])} | {pct(s['switch_share_B'])} | {s['ari_A']:.2f} | {s['ari_B']:.2f} |")
    with open(os.path.join(OUT, "cluster_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps({k: NUM[k] for k in ("imputed_share_origin1", "eligibility_by_origin", "test_revenue_share_by_eligibility_origin1", "crosstab_A_vs_B_origin1",
                                          "stay_rate_A_1->2", "stay_rate_B_1->2", "pca_explained_variance", "A_origin1", "sb_davies_bouldin", "final_silhouette")}, ensure_ascii=True, indent=1))
    print(json.dumps({k: v.get("share_test_revenue") for k, v in NUM["clusters_origin1"].items()}, ensure_ascii=True))
    print(profile.round(2).to_string())


if __name__ == "__main__":
    main()
