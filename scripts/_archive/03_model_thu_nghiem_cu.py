"""Giai đoạn 3: dự báo doanh thu tuần, tầm 13 tuần, theo cụm kiểu nhu cầu.

Chạy:  python scripts/03_model.py

Đọc clean/ (không sửa), ghi vào reports/model/:
  results.json                   mọi con số
  fig01_wmape_theo_phuong_phap.png, fig02_wmape_theo_cum.png
  forecasts_test.parquet         dự báo của phương pháp được chọn trên 4 mốc kiểm tra

Quy trình (không dùng tập kiểm tra để chọn mô hình):
  - Mốc XÁC THỰC: 4 mốc trong 52 tuần cuối của giai đoạn huấn luyện (tuần 79, 92, 105, 118).
  - Mốc KIỂM TRA: 4 mốc trong 52 tuần cuối của dữ liệu (tuần 131, 144, 157, 170).
  - Tại mỗi mốc o: chỉ dùng dữ liệu của các tuần < o; mô hình học máy chỉ học từ các mốc t có t + 13 <= o.
  - Cụm (smooth / erratic / intermittent / lumpy / short / dormant) được gán lại tại từng mốc từ lịch sử trước mốc.
  - Chọn phương pháp tốt nhất cho từng cụm trên mốc xác thực, rồi áp nguyên lựa chọn đó cho mốc kiểm tra.
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "reports", "model")
H = 13
VAL_ORIGINS = [79, 92, 105, 118]
TEST_ORIGINS = [131, 144, 157, 170]
TRAIN_ORIGIN_MIN, TRAIN_ORIGIN_STEP = 30, 3
MAX_TRAIN_ROWS = 1_200_000
CLASSES = ["smooth", "erratic", "intermittent", "lumpy", "short", "dormant"]
ML_VARIANTS = [("revenue", "squared_error"), ("revenue", "poisson"), ("units", "squared_error"), ("units", "poisson")]
SEED = 0


def window_mean(M, o, w, age):
    """Trung bình w tuần cuối trước mốc o, chỉ tính các tuần chuỗi đã ra mắt."""
    n = np.minimum(w, age)
    return M[:, max(0, o - w):o].sum(axis=1) / np.maximum(n, 1)


def series_state(o, Q, first_idx):
    """Thông tin của từng chuỗi tại mốc o, chỉ từ các tuần < o."""
    exists = first_idx < o
    age = np.where(exists, o - first_idx, 0)
    hist = Q[:, :o]
    nz = hist > 0
    n_nz = nz.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_nz = hist.sum(axis=1) / n_nz
        var_nz = (np.where(nz, hist, 0) ** 2).sum(axis=1) / n_nz - mean_nz ** 2
        cv2 = var_nz / mean_nz ** 2
        adi = age / n_nz
    last_sale = np.where(n_nz > 0, o - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    zero_run = np.where(exists, o - 1 - last_sale, 0)
    klass = np.where(adi < 1.32, np.where(cv2 < 0.49, 0, 1), np.where(cv2 < 0.49, 2, 3))
    klass = np.where(age < 26, 4, klass)
    klass = np.where(zero_run >= 26, 5, klass)
    return exists, age, zero_run, klass, adi, cv2


def client_season(o, client_rev):
    """Hệ số mùa vụ cấp nhà bán hàng cho 13 tuần sau mốc o: doanh thu cùng tuần năm trước / mức nền 13 tuần trước đó.
    Co về hệ số toàn bộ khi nhà bán hàng nhỏ. NaN nếu chưa đủ 65 tuần lịch sử."""
    n_clients = client_rev.shape[0]
    if o - 65 < 0:
        return np.full((n_clients, H), np.nan), np.full(H, np.nan)
    base_c = client_rev[:, o - 65:o - 52].mean(axis=1)
    tot = client_rev.sum(axis=0)
    g = tot[o - 52:o - 52 + H] / tot[o - 65:o - 52].mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        c = client_rev[:, o - 52:o - 52 + H] / base_c[:, None]
    c = np.where(np.isfinite(c) & (base_c[:, None] > 0), c, g[None, :])
    c = 0.5 * c + 0.5 * g[None, :]
    return np.clip(c, 0.5, 3.0), g


def build_features(o, M, D, with_target):
    """Một dòng cho mỗi (chuỗi, tầm h). M là ma trận mục tiêu (doanh thu hoặc số lượng)."""
    Q, R, P, first_idx, client_code, client_rev, week_of_year = (D[k] for k in ("Q", "R", "P", "first_idx", "client_code", "client_rev", "woy"))
    exists, age, zero_run, klass, adi, cv2 = series_state(o, Q, first_idx)
    m13 = window_mean(M, o, 13, age)
    m52 = window_mean(M, o, 52, age)
    scale = np.where(m13 > 0, m13, m52)
    use = exists & (scale > 0) & (klass != 5)
    idx = np.nonzero(use)[0]
    s, a = scale[idx], age[idx]
    Mi = M[idx]
    f = {
        "age": np.minimum(a, 156), "zero_run": zero_run[idx], "klass": klass[idx], "client": client_code[idx],
        "log_scale": np.log1p(s),
        "last": Mi[:, o - 1] / s,
        "m4": window_mean(Mi, o, 4, a) / s, "m13": window_mean(Mi, o, 13, a) / s,
        "m26": window_mean(Mi, o, 26, a) / s, "m52": window_mean(Mi, o, 52, a) / s,
        "max13": Mi[:, o - 13:o].max(axis=1) / s,
        "nz4": window_mean((Mi > 0).astype(float), o, 4, a), "nz13": window_mean((Mi > 0).astype(float), o, 13, a),
        "nz52": window_mean((Mi > 0).astype(float), o, 52, a),
        "adi": np.minimum(adi[idx], 60), "cv2": np.minimum(np.nan_to_num(cv2[idx]), 20),
    }
    price_last = P[idx, o - 1]
    with np.errstate(invalid="ignore", all="ignore"):
        price_med = np.nanmedian(np.where(Q[idx, :o] > 0, P[idx, :o], np.nan), axis=1)
    f["log_price"] = np.log(np.where(price_last > 0, price_last, np.nan))
    f["price_rel"] = np.clip(price_last / price_med, 1 / 3, 3)
    season_c, season_g = client_season(o, client_rev)
    n = len(idx)
    X = pd.DataFrame({k: np.repeat(v, H) for k, v in f.items()})
    hh = np.tile(np.arange(1, H + 1), n)
    X["h"] = hh
    tw = (o + hh - 1) % len(week_of_year) if o + H > len(week_of_year) else o + hh - 1
    X["woy"] = week_of_year[tw]
    X["season_client"] = season_c[client_code[idx]].ravel()
    X["season_global"] = np.tile(season_g, n)
    y = None
    if with_target:
        y = (Mi[:, o:o + H] / s[:, None]).ravel()
    return X.astype(np.float32), y, idx, s


def fit_predict(o, M, D, loss):
    """Huấn luyện trên các mốc t <= o - 13 rồi dự báo 13 tuần sau mốc o. Trả về ma trận (số chuỗi x 13) theo đơn vị của M."""
    rng = np.random.default_rng(SEED)
    Xs, ys = [], []
    for t in range(TRAIN_ORIGIN_MIN, o - H + 1, TRAIN_ORIGIN_STEP):
        X, y, _, _ = build_features(t, M, D, True)
        Xs.append(X); ys.append(y)
    X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    if len(X) > MAX_TRAIN_ROWS:
        keep = rng.choice(len(X), MAX_TRAIN_ROWS, replace=False)
        X, y = X.iloc[keep], y[keep]
    cat = [X.columns.get_loc(c) for c in ("klass", "client")]
    model = HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.07, max_leaf_nodes=63,
                                          min_samples_leaf=100, l2_regularization=1.0, categorical_features=cat,
                                          random_state=SEED)
    model.fit(X, np.clip(y, 0, 30))
    Xo, _, idx, s = build_features(o, M, D, False)
    pred = np.clip(model.predict(Xo), 0, None).reshape(len(idx), H) * s[:, None]
    out = np.zeros((M.shape[0], H))
    out[idx] = pred
    return out, len(X)


def simple_forecasts(o, D):
    """Các phương pháp không cần huấn luyện, đều theo doanh thu. Chuỗi chưa ra mắt và chuỗi ngủ đông: 0 (trừ naive/mean nguyên bản)."""
    R, first_idx, client_code = D["R"], D["first_idx"], D["client_code"]
    exists, age, zero_run, klass, _, _ = series_state(o, D["Q"], first_idx)
    rep = lambda v: np.repeat(v[:, None], H, axis=1)
    m13, m26, m52 = (window_mean(R, o, w, age) for w in (13, 26, 52))
    season_c, _ = client_season(o, D["client_rev"])
    season = season_c[client_code]
    out = {"naive": rep(R[:, o - 1]), "mean13": rep(m13), "mean26": rep(m26), "mean52": rep(m52),
           "zero": np.zeros((len(R), H))}
    if o - 52 >= 0:
        out["seasonal_naive"] = R[:, o - 52:o - 52 + H].copy()
    if not np.isnan(season).any():
        out["mean13_x_season"] = rep(m13) * season
        out["mean26_x_season"] = rep(m26) * season
    return out, klass, exists


def evaluate(fc, actual, klass, exists):
    """Tổng |sai số|, tổng thực tế, tổng sai lệch: toàn bộ và theo cụm (cụm 6 = chưa ra mắt tại mốc)."""
    k = np.where(exists, klass, 6)
    err = fc - actual
    rows = {}
    for c in range(7):
        m = k == c
        rows[c] = (float(np.abs(err[m]).sum()), float(actual[m].sum()), float(err[m].sum()), int(m.sum()))
    tot = (float(np.abs(err.sum(axis=0)).sum()), float(actual.sum()))
    return rows, tot


def run_origins(origins, D, label):
    acc, totals, n_train = {}, {}, {}
    store = {}
    for o in origins:
        t0 = time.time()
        actual = D["R"][:, o:o + H]
        fcs, klass, exists = simple_forecasts(o, D)
        for target, loss in ML_VARIANTS:
            M = D["R"] if target == "revenue" else D["Q"]
            pred, n = fit_predict(o, M, D, loss)
            if target == "units":
                pred = pred * np.nan_to_num(D["P"][:, o - 1])[:, None]
            fcs[f"gbm_{target}_{loss}"] = pred
            n_train[f"{o}_{target}_{loss}"] = n
        for name, fc in fcs.items():
            fc = np.where(exists[:, None], fc, 0.0)
            rows, tot = evaluate(fc, actual, klass, exists)
            a = acc.setdefault(name, {c: [0.0, 0.0, 0.0, 0] for c in range(7)})
            for c, r in rows.items():
                for j in range(4):
                    a[c][j] += r[j]
            t = totals.setdefault(name, [0.0, 0.0])
            t[0] += tot[0]; t[1] += tot[1]
            store[(o, name)] = fc
        store[(o, "_klass")] = np.where(exists, klass, 6)
        print(f"[{label}] origin {o} done in {time.time() - t0:.0f}s", flush=True)
    return acc, totals, store, n_train


def summarize(acc, totals):
    out = {}
    for name, a in acc.items():
        ab = sum(v[0] for v in a.values()); act = sum(v[1] for v in a.values()); bias = sum(v[2] for v in a.values())
        ex_ab = sum(a[c][0] for c in range(6)); ex_act = sum(a[c][1] for c in range(6))
        out[name] = {"wmape_series_all": ab / act, "bias_all": bias / act, "wmape_series_existing": ex_ab / ex_act,
                     "wmape_total_level": totals[name][0] / totals[name][1],
                     "by_class": {CLASSES[c]: {"wmape": a[c][0] / a[c][1] if a[c][1] else None, "abs_err": a[c][0],
                                               "actual": a[c][1], "series_origin_count": a[c][3]} for c in range(6)}}
    return out


def combine(store, origins, choice):
    """Ghép dự báo: mỗi cụm dùng phương pháp đã chọn."""
    res = {}
    for o in origins:
        k = store[(o, "_klass")]
        fc = np.zeros_like(store[(o, "naive")])
        for c, name in choice.items():
            fc[k == c] = store[(o, name)][k == c]
        res[o] = fc
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(CLEAN, "vn1_long.parquet"))
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    R = df.pivot(index="unique_id", columns="week", values="revenue").reindex(columns=weeks)
    ids = R.index
    Q = df.pivot(index="unique_id", columns="week", values="sales").reindex(index=ids, columns=weeks).fillna(0.0).to_numpy()
    P = df.pivot(index="unique_id", columns="week", values="price").reindex(index=ids, columns=weeks).to_numpy()
    meta = df.groupby("unique_id").agg(client=("client", "first"), first=("first_sale_week", "first")).loc[ids]
    client_code = pd.factorize(meta["client"])[0]
    R = R.fillna(0.0).to_numpy()
    client_rev = np.zeros((client_code.max() + 1, len(weeks)))
    np.add.at(client_rev, client_code, R)
    D = {"R": R, "Q": Q, "P": P, "first_idx": np.searchsorted(weeks.values, meta["first"].values),
         "client_code": client_code, "client_rev": client_rev, "woy": weeks.isocalendar().week.to_numpy().astype(int)}

    val_acc, val_tot, val_store, _ = run_origins(VAL_ORIGINS, D, "val")
    val = summarize(val_acc, val_tot)
    methods = [m for m in val if all((o, m) in val_store for o in VAL_ORIGINS)]
    # Chọn theo cụm trên mốc xác thực
    choice = {}
    for c in range(6):
        choice[c] = min(methods, key=lambda m: val_acc[m][c][0])
    best_single = min(methods, key=lambda m: val[m]["wmape_series_all"])
    best_gbm = min((m for m in methods if m.startswith("gbm")), key=lambda m: val[m]["wmape_series_all"])

    test_acc, test_tot, test_store, n_train = run_origins(TEST_ORIGINS, D, "test")
    test = summarize(test_acc, test_tot)

    def score_combo(store, origins):
        fc = combine(store, origins, choice)
        acc = {c: [0.0, 0.0, 0.0, 0] for c in range(7)}
        tot = [0.0, 0.0]
        for o in origins:
            k = store[(o, "_klass")]
            rows, t = evaluate(fc[o], D["R"][:, o:o + H], np.where(k == 6, 0, k), k != 6)
            for c, r in rows.items():
                for j in range(4):
                    acc[c][j] += r[j]
            tot[0] += t[0]; tot[1] += t[1]
        return summarize({"combo": acc}, {"combo": tot})["combo"], fc

    val["chon_theo_cum"], _ = score_combo(val_store, VAL_ORIGINS)
    test["chon_theo_cum"], combo_fc = score_combo(test_store, TEST_ORIGINS)

    # Tỷ trọng doanh thu thực tế theo cụm trên mốc kiểm tra
    act = {CLASSES[c]: test_acc["naive"][c][1] for c in range(6)}
    act["chua_ra_mat"] = test_acc["naive"][6][1]
    tot_act = sum(act.values())
    results = {
        "setup": {"horizon": H, "val_origins": [str(weeks[o].date()) for o in VAL_ORIGINS],
                  "test_origins": [str(weeks[o].date()) for o in TEST_ORIGINS],
                  "metric": "WMAPE = tổng |sai số| / tổng doanh thu thực tế, gộp 4 mốc x 13 tuần",
                  "train_rows_per_fit": n_train},
        "choice_by_class_from_validation": {CLASSES[c]: m for c, m in choice.items()},
        "best_single_method_on_validation": best_single, "best_gbm_on_validation": best_gbm,
        "test_revenue_share_by_class": {k: v / tot_act for k, v in act.items()},
        "test_series_origin_count_by_class": {CLASSES[c]: test_acc["naive"][c][3] for c in range(6)} | {"chua_ra_mat": test_acc["naive"][6][3]},
        "validation": val, "test": test,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # Lưu dự báo của phương án chọn theo cụm
    rows = []
    for o in TEST_ORIGINS:
        k = test_store[(o, "_klass")]
        n = len(ids)
        rows.append(pd.DataFrame({
            "unique_id": np.repeat(ids.to_numpy(), H), "origin": weeks[o], "h": np.tile(np.arange(1, H + 1), n),
            "week": np.tile(weeks[o:o + H].values, n), "cluster": np.repeat(np.array(CLASSES + ["chua_ra_mat"])[k], H),
            "forecast": combo_fc[o].ravel(), "actual": D["R"][:, o:o + H].ravel()}))
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(OUT, "forecasts_test.parquet"), index=False)

    # Hình
    BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                         "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "figure.dpi": 130, "legend.frameon": False})
    order = sorted(test, key=lambda m: test[m]["wmape_series_all"])
    order = [m for m in order if m != "zero"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    vals = [test[m]["wmape_series_all"] * 100 for m in order]
    colors = [AQUA if m == "chon_theo_cum" else BLUE if m.startswith("gbm") else GRAY for m in order]
    bars = ax.barh(range(len(order)), vals, color=colors, height=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), order); ax.invert_yaxis(); ax.set_xlabel("WMAPE cấp chuỗi trên 4 mốc kiểm tra (%)")
    ax.set_title("Sai số theo phương pháp (xám: đơn giản, xanh: học máy, lục: chọn theo cụm)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_wmape_theo_phuong_phap.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(6)
    for j, (m, color) in enumerate((("naive", GRAY), (best_gbm, BLUE), ("chon_theo_cum", AQUA))):
        v = [(test[m]["by_class"][c]["wmape"] or 0) * 100 for c in CLASSES]
        ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=m)
    ax.set_xticks(x, [f"{c}\n{results['test_revenue_share_by_class'][c]:.0%} doanh thu" for c in CLASSES])
    ax.set_ylim(0, 250); ax.set_ylabel("WMAPE (%)"); ax.legend(); ax.set_title("Sai số theo cụm trên mốc kiểm tra (cắt ở 250%)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_wmape_theo_cum.png")); plt.close(fig)

    brief = {"choice": results["choice_by_class_from_validation"], "best_single_val": best_single, "best_gbm_val": best_gbm,
             "rev_share": results["test_revenue_share_by_class"],
             "val": {m: {k: round(v, 4) for k, v in d.items() if k != "by_class"} for m, d in val.items()},
             "test": {m: {k: round(v, 4) for k, v in d.items() if k != "by_class"} for m, d in test.items()},
             "test_by_class": {m: {c: (round(d["by_class"][c]["wmape"], 3) if d["by_class"][c]["wmape"] is not None else None) for c in CLASSES}
                               for m, d in test.items()}}
    print(json.dumps(brief, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
