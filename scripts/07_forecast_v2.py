"""Mô hình hóa, bản cải tiến (v2): mô hình học máy mạnh hơn và trộn theo nhóm.

Chạy:  python scripts/07_forecast_v2.py                  (khoảng 40-60 phút)
       python scripts/07_forecast_v2.py --figures-only   (chỉ vẽ lại hình từ results.json)

Đọc clean/ (không sửa), dùng lại các hàm của scripts/06_forecast.py, ghi vào reports/forecast_v2/:
  results.json, forecasts_test_v2.parquet, fig01..fig04 *.png

Quy trình (mọi lựa chọn chỉ dựa trên 4 mốc XÁC THỰC; 4 mốc KIỂM TRA chỉ chấm một lần ở cuối)
  Bước 1  Trọng số huấn luyện: thử trọng số = (mức bán của chuỗi)^p với p = 0; 0,5; 1.
  Bước 2  Đặc trưng mới: mùa vụ của chính chuỗi, hệ số mùa vụ trung bình hai năm, đà của nhà bán hàng / kho / sản phẩm,
          mức bán của cùng sản phẩm ở kho khác, từng tuần gần nhất, độ dao động, thay đổi giá.
  Bước 3  Mô hình lớn hơn (nhiều vòng lặp hơn, cây lớn hơn).
  Bước 4  Thêm mô hình dự báo trung vị (hàm mất mát sai số tuyệt đối).
  Bước 5  Trộn theo nhóm: với mỗi nhóm kiểu nhu cầu, chọn tỷ lệ trộn giữa mô hình trung bình, mô hình trung vị,
          naive, trung bình 13 tuần và TSB.
  Tiêu chí chọn ở mọi bước: ĐIỂM VN1 trên mốc xác thực. Điểm VN1 = MAE% + |Bias%|, tính riêng cho từng kỳ 13 tuần
  (như cuộc thi chấm một kỳ) rồi lấy trung bình các kỳ. Không gộp các kỳ trước khi tính độ lệch, vì lệch dương của kỳ này
  sẽ bù cho lệch âm của kỳ khác.
"""
import importlib.util
import itertools
import json
import os
import shutil
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "forecast_v2")
CACHE = os.path.join(OUT, "_cache")
_spec = importlib.util.spec_from_file_location("forecast_base", os.path.join(ROOT, "scripts", "06_forecast.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
H, GROUPS, VAL_ORIGINS, TEST_ORIGINS = base.H, base.GROUPS, base.VAL_ORIGINS, base.TEST_ORIGINS
TRAIN_MIN, TRAIN_STEP, MAX_ROWS, SEED = 30, 3, 1_200_000, 0
BLEND_STEP = 0.25
warnings.filterwarnings("ignore", category=RuntimeWarning)

OLD = ["age", "zero_run", "group", "client", "log_scale", "last", "m4", "m13", "m26", "m52", "max13", "nz4", "nz13", "nz52",
       "adi", "cv2", "q4_share", "log_price", "price_rel", "h", "woy", "season_global", "season_client"]
ALL = None       # điền khi dựng đặc trưng lần đầu


# ---------------------------------------------------------------- đặc trưng
def features(o, D, with_target):
    """Một dòng cho mỗi (chuỗi, tầm h). Mọi đại lượng chỉ dùng các tuần < o và được chia cho mức bán gần đây của chuỗi."""
    st = base.series_state(o, D)
    age, g, R = st["age"], st["group"], D["Rh"]
    m13_all, m52_all = base.wmean(R, o, 13, age), base.wmean(R, o, 52, age)
    scale = np.where(m13_all > 0, m13_all, m52_all)
    idx = np.flatnonzero(st["exists"] & (scale > 0) & (g != 5))
    s, a, Ri = scale[idx], age[idx], R[idx]
    nzf = (Ri > 0).astype(float)
    wm = lambda M, w: base.wmean(M, o, w, a)
    f = {"age": np.minimum(a, 156), "zero_run": st["zero_run"][idx], "group": g[idx], "client": D["client"][idx],
         "log_scale": np.log1p(s), "last": Ri[:, o - 1] / s, "m4": wm(Ri, 4) / s, "m13": wm(Ri, 13) / s, "m26": wm(Ri, 26) / s,
         "m52": wm(Ri, 52) / s, "max13": Ri[:, o - 13:o].max(axis=1) / s, "nz4": wm(nzf, 4), "nz13": wm(nzf, 13), "nz52": wm(nzf, 52),
         "adi": np.minimum(st["adi"][idx], 60), "cv2": np.minimum(np.nan_to_num(st["cv2"][idx]), 20), "q4_share": st["q4_share"][idx]}
    P = D["P"][idx]
    price_last = P[:, o - 1]
    price_med = np.nanmedian(np.where(D["Qh"][idx, :o] > 0, P[:, :o], np.nan), axis=1)
    f["log_price"] = np.log(np.where(price_last > 0, price_last, np.nan))
    f["price_rel"] = np.clip(price_last / price_med, 1 / 3, 3)
    # ----- đặc trưng mới -----
    for k in (2, 3, 4):
        f[f"lag{k}"] = np.where(a >= k, Ri[:, o - k], np.nan) / s
    f["m8"] = wm(Ri, 8) / s
    f["std13"] = Ri[:, o - 13:o].std(axis=1) / s
    f["max52"] = Ri[:, max(0, o - 52):o].max(axis=1) / s
    f["trend_4_13"] = np.log((f["m4"] + 0.1) / (f["m13"] + 0.1))
    f["trend_13_52"] = np.log((f["m13"] + 0.1) / (f["m52"] + 0.1))
    f["nz26"] = wm(nzf, 26)
    f["price_chg"] = np.clip(np.log(price_last / np.nanmean(P[:, o - 13:o], axis=1)), -1, 1)
    f["flag_q4"] = st["flag_q4"][idx].astype(float)
    ex = st["exists"].astype(float)
    for name in ("client", "wh", "prod"):                       # đà và quy mô của nhà bán hàng, kho, sản phẩm
        codes = D[name]
        n = codes.max() + 1
        t4 = np.bincount(codes, weights=R[:, o - 4:o].sum(axis=1), minlength=n) / 4
        t13 = np.bincount(codes, weights=R[:, o - 13:o].sum(axis=1), minlength=n) / 13
        t52 = np.bincount(codes, weights=R[:, max(0, o - 52):o].sum(axis=1), minlength=n) / min(52, o)
        cnt = np.bincount(codes, weights=ex, minlength=n)
        c = codes[idx]
        f[name + "_log13"] = np.log1p(t13[c])
        f[name + "_mom_4_13"] = np.log((t4[c] + 1) / (t13[c] + 1))
        f[name + "_mom_13_52"] = np.log((t13[c] + 1) / (t52[c] + 1))
        f[name + "_n"] = cnt[c]
        f[name + "_share"] = m13_all[idx] / (t13[c] + 1e-9)
        if name == "prod":                                       # mức bán của cùng sản phẩm ở các kho khác
            f["prod_sibling_level"] = np.log1p(np.maximum(t13[c] - m13_all[idx], 0) / np.maximum(cnt[c] - 1, 1) / s)
    n = len(idx)
    hh = np.arange(1, H + 1)
    tw = o + hh - 1
    per_h = {"h": np.tile(hh, n), "woy": np.tile(D["woy_ext"][tw], n),
             "is_bf": np.tile(np.isin(D["woy_ext"][tw], [47, 48]).astype(float), n)}
    nan_h = np.full(n * H, np.nan)
    if o >= 65:
        sg, sc = base.season_ratios(o, D)
        per_h["season_global"] = np.tile(sg, n)
        per_h["season_client"] = sc[D["client"][idx]].ravel()
        base_lvl = Ri[:, o - 65:o - 52].mean(axis=1)
        ok = (a >= 65) & (base_lvl > 0)
        safe = np.where(base_lvl > 0, base_lvl, 1.0)
        own = np.stack([Ri[:, o - 52 + h - 2:o - 52 + h + 1].mean(axis=1) for h in hh], axis=1)   # 3 tuần quanh tuần cùng kỳ năm trước
        per_h["season_own"] = np.where(ok[:, None], np.clip(own / safe[:, None], 0, 10), np.nan).ravel()
        f["season_own_h13"] = np.where(ok, np.clip(Ri[:, o - 52:o - 39].mean(axis=1) / safe, 0, 10), np.nan)
        sg2 = base.season_ratios(o - 52, D)[0] if o - 52 >= 65 else sg
        per_h["season_global_2y"] = np.tile((sg + sg2) / 2, n)
    else:
        per_h["season_global"] = nan_h; per_h["season_client"] = nan_h; per_h["season_own"] = nan_h
        f["season_own_h13"] = np.full(n, np.nan); per_h["season_global_2y"] = nan_h
    cols = list(f) + list(per_h)
    X = np.empty((n * H, len(cols)), dtype=np.float32)
    for j, k in enumerate(f):
        X[:, j] = np.repeat(np.asarray(f[k], dtype=np.float64), H)
    for j, k in enumerate(per_h):
        X[:, len(f) + j] = per_h[k]
    y = (Ri[:, o:o + H] / s[:, None]).ravel().astype(np.float32) if with_target else None
    return cols, X, y, idx, s


def cache_origin(o, D, with_target):
    global ALL
    cols, X, y, idx, s = features(o, D, with_target)
    if ALL is None:
        ALL = cols
    assert cols == ALL
    np.save(os.path.join(CACHE, f"X_{o}.npy"), X)
    np.save(os.path.join(CACHE, f"s_{o}.npy"), s)
    np.save(os.path.join(CACHE, f"idx_{o}.npy"), idx)
    if with_target:
        np.save(os.path.join(CACHE, f"y_{o}.npy"), y)


def training_set(o, seed):
    rng = np.random.default_rng(seed + o)
    ts = list(range(TRAIN_MIN, o - H + 1, TRAIN_STEP))
    sizes = [np.load(os.path.join(CACHE, f"y_{t}.npy"), mmap_mode="r").shape[0] for t in ts]
    frac = min(1.0, MAX_ROWS / sum(sizes))
    Xs, ys, ws = [], [], []
    for t, m in zip(ts, sizes):
        keep = np.flatnonzero(rng.random(m) < frac) if frac < 1 else np.arange(m)
        Xs.append(np.load(os.path.join(CACHE, f"X_{t}.npy"), mmap_mode="r")[keep])
        ys.append(np.load(os.path.join(CACHE, f"y_{t}.npy"), mmap_mode="r")[keep])
        ws.append(np.repeat(np.load(os.path.join(CACHE, f"s_{t}.npy")), H)[keep])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def fit_predict(o, colnames, cfg, n_series):
    X, y, w = training_set(o, SEED)
    ci = [ALL.index(c) for c in colnames]
    cat = [colnames.index(c) for c in ("group", "client")]
    weight = w ** cfg["p"]
    model = HistGradientBoostingRegressor(loss=cfg["loss"], max_iter=cfg["iters"], learning_rate=cfg["lr"], max_leaf_nodes=cfg["leaves"],
                                          min_samples_leaf=cfg["min_leaf"], l2_regularization=1.0, categorical_features=cat, random_state=SEED)
    model.fit(X[:, ci], np.clip(y, 0, 30), sample_weight=weight / weight.mean())
    del X, y, w
    Xo = np.load(os.path.join(CACHE, f"X_{o}.npy"))
    idx, s = np.load(os.path.join(CACHE, f"idx_{o}.npy")), np.load(os.path.join(CACHE, f"s_{o}.npy"))
    out = np.zeros((n_series, H))
    out[idx] = np.clip(model.predict(Xo[:, ci]), 0, None).reshape(len(idx), H) * s[:, None]
    return out


def init_store(o, D):
    """Cùng cấu trúc với scripts/06_forecast.py để dùng lại hàm chấm điểm; kèm các dự báo đơn giản."""
    st = base.series_state(o, D)
    age, exists = st["age"], st["exists"]
    rep = lambda v: np.repeat(np.asarray(v, dtype=float)[:, None], H, axis=1)
    price = np.nan_to_num(D["P"][:, o - 1])
    fc = {"naive": rep(D["Rh"][:, o - 1]), "mean13": rep(base.wmean(D["Rh"], o, 13, age)), "zero": np.zeros((len(age), H)),
          "tsb_a0.2": rep(base.smoothing_family(o, D, 0.2)["tsb"] * price)}
    fc = {k: np.where(exists[:, None], v, 0.0) for k, v in fc.items()}
    diff2 = (D["Rh"][:, 1:o] - D["Rh"][:, :o - 1]) ** 2
    ok = np.arange(o - 1)[None, :] >= D["start"][:, None]
    n_diff = ok.sum(axis=1)
    return {"scale": np.where(n_diff > 0, (diff2 * ok).sum(axis=1) / np.maximum(n_diff, 1), 0.0), "weight": D["Rh"][:, o - 13:o].sum(axis=1),
            "fc": fc, "group": st["group"], "flag_q4": st["flag_q4"], "actual": D["R"][:, o:o + H], "valid": ~D["F"][:, o:o + H]}


def vn1(store, origins, pick, mask=None):
    """Điểm VN1 = MAE% + |Bias%| của từng kỳ 13 tuần, lấy trung bình các kỳ (bỏ kỳ không có doanh thu thực tế)."""
    vals = []
    for o in origins:
        sc = base.score(store, [o], pick, mask)
        if sc["wmape"] is not None:
            vals.append(sc["wmape"] + abs(sc["bias"]))
    return float(np.mean(vals)) if vals else None


def summarize(store, origins, pick, mask=None):
    sc = base.score(store, origins, pick, mask)
    out = {k: sc[k] for k in ("wmape", "bias", "wmape_total_level", "wrmsse", "rmsse_median", "rmse")}
    out["diem_gop"] = sc["wmape"] + abs(sc["bias"]) if sc["wmape"] is not None else None     # cách cũ: gộp các kỳ rồi mới tính
    out["diem_vn1"] = vn1(store, origins, pick, mask)
    return out


def blend_weights(store, origins, comps):
    """Với mỗi nhóm, tìm tỷ lệ trộn (bước BLEND_STEP, tổng bằng 1) tối thiểu hóa điểm VN1 của nhóm:
    trung bình theo kỳ của (tổng |sai số| + |tổng sai số|) / tổng thực tế."""
    k = int(round(1 / BLEND_STEP))
    grid = [np.array(c) / k for c in itertools.product(range(k + 1), repeat=len(comps)) if sum(c) == k]
    out = {}
    for code in range(5):
        parts, acts = [], []
        for o in origins:
            s = store[o]
            m = s["group"] == code
            v = s["valid"][m]
            parts.append(np.stack([np.where(v, s["fc"][c][m], 0.0) for c in comps]))
            acts.append(np.where(v, s["actual"][m], 0.0))
        best = None
        for w in grid:
            per = []
            for p_, a_ in zip(parts, acts):
                if a_.sum() > 0:
                    e = np.tensordot(w, p_, axes=1) - a_
                    per.append((np.abs(e).sum() + abs(e.sum())) / a_.sum())
            val = float(np.mean(per))
            if best is None or val < best[0]:
                best = (val, w)
        out[code] = {c: float(x) for c, x in zip(comps, best[1])}
    return out


def blended(store, o, weights):
    s = store[o]
    out = np.zeros_like(s["actual"])
    for code, w in weights.items():
        m = s["group"] == code
        out[m] = sum(x * s["fc"][c][m] for c, x in w.items() if x > 0)
    return out


def full_scores(store, origins, pick, seller0, weeks):
    return {"toan_bo": summarize(store, origins, pick),
            "nha_ban_hang_0": summarize(store, origins, pick, lambda o: seller0),
            "con_lai": summarize(store, origins, pick, lambda o: ~seller0),
            "co_mua_vu_q4": summarize(store, origins, pick, lambda o: store[o]["flag_q4"]),
            "theo_nhom": {GROUPS[c]: summarize(store, origins, pick, lambda o, c=c: store[o]["group"] == c) for c in range(5)},
            "theo_moc": {str(weeks[o].date()): summarize(store, [o], pick) for o in origins}}


LABELS = {"v2_tron_theo_nhom": "v2: trộn theo nhóm", "v2_mo_hinh_trung_binh": "v2: một mô hình (trung bình)", "v2_mo_hinh_trung_vi": "v2: một mô hình (trung vị)",
          "v1_moi_nhom_tu_chon": "v1: mỗi nhóm tự chọn", "v1_gbm_doanh_thu": "v1: GBM doanh thu trực tiếp", "naive": "Naive", "mean13": "Trung bình 13 tuần",
          "tsb_a0.2": "TSB (alpha 0,2)"}
STUDY_LABELS = {"cu": "Cấu hình của bản đầu", "cu_p0.5": "Trọng số mũ 0,5", "cu_p1": "Trọng số mũ 1", "moi": "Trọng số tốt nhất + đặc trưng mới",
                "lon": "Trọng số tốt nhất + mô hình lớn hơn", "trung_vi": "Mô hình trung vị", "tron": "Trộn theo nhóm (được chọn)"}


def make_figures(out):
    BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10, "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 130, "legend.frameon": False})
    study, T = out["validation_study"], out["test"]
    names = [k for k in STUDY_LABELS if k in study]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    v = [study[k]["diem_vn1"] * 100 for k in names]
    bars = ax.barh(range(len(names)), v, color=[AQUA if k == "tron" else BLUE for k in names], height=0.7)
    for b_, x_ in zip(bars, v):
        ax.text(x_ + 0.5, b_.get_y() + b_.get_height() / 2, f"{x_:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), [STUDY_LABELS[k] for k in names]); ax.invert_yaxis(); ax.set_xlim(0, max(v) * 1.12)
    ax.set_xlabel("điểm VN1, trung bình 4 kỳ xác thực"); ax.set_title("Các phương án thử trên mốc xác thực (thấp hơn là tốt hơn)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_tung_buoc_cai_tien.png")); plt.close(fig)

    trio = [("naive", GRAY), ("v1_moi_nhom_tu_chon", BLUE), ("v2_tron_theo_nhom", AQUA)]

    def grouped(ax, cats, getter, fmt="{:.0f}", dy=1.0):
        x = np.arange(len(cats))
        for j, (m, color) in enumerate(trio):
            v = [getter(m, c) for c in cats]
            bars = ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=LABELS[m])
            for b, val_ in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val_ + (dy if val_ >= 0 else -3 * dy), fmt.format(val_), ha="center", fontsize=8)
        return x

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    cats = GROUPS[:5]
    x = grouped(ax, cats, lambda m, c: T[m]["theo_nhom"][c]["diem_vn1"] * 100)
    ax.set_xticks(x, [f"{c}\n{out['test_composition'][c]:.0%} doanh thu" for c in cats]); ax.set_ylim(0, 125); ax.set_ylabel("điểm VN1")
    ax.legend(loc="upper left", ncol=3, fontsize=9); ax.set_title("Điểm VN1 theo nhóm trên mốc kiểm tra (trung bình 4 kỳ)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_theo_nhom.png")); plt.close(fig)

    origins = list(T["naive"]["theo_moc"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = grouped(axes[0], origins, lambda m, c: T[m]["theo_moc"][c]["diem_vn1"] * 100)
    axes[0].set_ylim(0, 100); axes[0].legend(loc="upper center", ncol=3, fontsize=8); axes[0].set_title("Điểm VN1 của từng kỳ 13 tuần")
    grouped(axes[1], origins, lambda m, c: T[m]["theo_moc"][c]["bias"] * 100)
    axes[1].set_ylim(-26, 24); axes[1].axhline(0, color=MUTED, lw=1); axes[1].set_title("Độ lệch theo từng mốc (%, dương = dự báo cao)")
    for ax in axes:
        ax.set_xticks(x, [f"mốc {pd.Timestamp(o).strftime('%d/%m/%Y')}" for o in origins], fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_theo_tung_moc.png")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    order = sorted((m for m in ("v2_tron_theo_nhom", "v2_mo_hinh_trung_binh", "v2_mo_hinh_trung_vi", "v1_moi_nhom_tu_chon", "v1_gbm_doanh_thu", "naive", "mean13", "tsb_a0.2") if m in T),
                   key=lambda m: T[m]["toan_bo"]["diem_vn1"])
    for ax, key, title, mult, fmt in ((axes[0], "diem_vn1", "Điểm VN1 (trung bình 4 kỳ)", 100, "{:.1f}"), (axes[1], "wmape", "MAE% gộp 4 kỳ", 100, "{:.1f}")):
        v = [T[m]["toan_bo"][key] * mult for m in order]
        bars = ax.barh(range(len(order)), v, color=[AQUA if m.startswith("v2") else BLUE if m.startswith("v1") else GRAY for m in order], height=0.7)
        for b, val_ in zip(bars, v):
            ax.text(val_ + max(v) * 0.01, b.get_y() + b.get_height() / 2, fmt.format(val_), va="center", fontsize=8)
        ax.set_title(title); ax.set_xlim(0, max(v) * 1.15)
    axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order]); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_so_sanh_chung.png")); plt.close(fig)


def main():
    os.makedirs(CACHE, exist_ok=True)
    D, ids, weeks, seller0 = base.load_data()
    n_series = len(ids)
    t0 = time.time()
    train_origins = list(range(TRAIN_MIN, max(VAL_ORIGINS + TEST_ORIGINS) - H + 1, TRAIN_STEP))
    for t in sorted(set(train_origins) | set(VAL_ORIGINS) | set(TEST_ORIGINS)):
        cache_origin(t, D, True)
    print(f"đặc trưng: {len(ALL)} cột, {time.time() - t0:.0f}s", flush=True)

    val = {o: init_store(o, D) for o in VAL_ORIGINS}
    study, cfgs = {}, {}

    def add(name, cols, cfg):
        t1 = time.time()
        for o in VAL_ORIGINS:
            val[o]["fc"][name] = fit_predict(o, cols, cfg, n_series)
        study[name] = summarize(val, VAL_ORIGINS, lambda o: val[o]["fc"][name])
        cfgs[name] = {"cols": cols, "cfg": cfg}
        print(f"[xác thực] {name:10s} điểm VN1 {study[name]['diem_vn1']:.4f} (gộp {study[name]['diem_gop']:.4f}) wmape {study[name]['wmape']:.4f} "
              f"bias {study[name]['bias']:+.4f} ({time.time() - t1:.0f}s)", flush=True)

    c0 = {"loss": "poisson", "iters": 250, "lr": 0.07, "leaves": 63, "min_leaf": 100, "p": 0.0}
    add("cu", OLD, c0)
    add("cu_p0.5", OLD, c0 | {"p": 0.5})
    add("cu_p1", OLD, c0 | {"p": 1.0})
    best_w = min(("cu", "cu_p0.5", "cu_p1"), key=lambda k: study[k]["diem_vn1"])
    p = cfgs[best_w]["cfg"]["p"]
    add("moi", ALL, c0 | {"p": p})
    cols = ALL if study["moi"]["diem_vn1"] < study[best_w]["diem_vn1"] else OLD
    add("lon", cols, c0 | {"p": p, "iters": 500, "lr": 0.05, "leaves": 127, "min_leaf": 50})
    best_mean = min(("cu", "cu_p0.5", "cu_p1", "moi", "lon"), key=lambda k: study[k]["diem_vn1"])
    add("trung_vi", cfgs[best_mean]["cols"], cfgs[best_mean]["cfg"] | {"loss": "absolute_error"})

    comps = [best_mean, "trung_vi", "naive", "mean13", "tsb_a0.2"]
    weights = blend_weights(val, VAL_ORIGINS, comps)
    study["tron"] = summarize(val, VAL_ORIGINS, lambda o: blended(val, o, weights))
    for k in ("naive", "mean13", "tsb_a0.2"):
        study[k] = summarize(val, VAL_ORIGINS, lambda o, k=k: val[o]["fc"][k])
    print(f"[xác thực] trộn: điểm VN1 {study['tron']['diem_vn1']:.4f} wmape {study['tron']['wmape']:.4f} bias {study['tron']['bias']:+.4f}", flush=True)
    print("tỷ lệ trộn:", json.dumps({GROUPS[c]: {k: v for k, v in w.items() if v > 0} for c, w in weights.items()}, ensure_ascii=False), flush=True)
    study_groups = {k: {GROUPS[c]: summarize(val, VAL_ORIGINS, (lambda o, k=k: val[o]["fc"][k]) if k != "tron" else (lambda o: blended(val, o, weights)),
                                             lambda o, c=c: val[o]["group"] == c) for c in range(5)}
                    for k in ("cu", best_mean, "trung_vi", "naive", "mean13", "tsb_a0.2", "tron")}
    del val

    # ---- chấm một lần trên mốc kiểm tra ----
    test = {o: init_store(o, D) for o in TEST_ORIGINS}
    for o in TEST_ORIGINS:
        t1 = time.time()
        test[o]["fc"][best_mean] = fit_predict(o, cfgs[best_mean]["cols"], cfgs[best_mean]["cfg"], n_series)
        test[o]["fc"]["trung_vi"] = fit_predict(o, cfgs["trung_vi"]["cols"], cfgs["trung_vi"]["cfg"], n_series)
        print(f"[kiểm tra] mốc {o} xong sau {time.time() - t1:.0f}s", flush=True)
    pickers = {"v2_tron_theo_nhom": lambda o: blended(test, o, weights), "v2_mo_hinh_trung_binh": lambda o: test[o]["fc"][best_mean],
               "v2_mo_hinh_trung_vi": lambda o: test[o]["fc"]["trung_vi"], "naive": lambda o: test[o]["fc"]["naive"],
               "mean13": lambda o: test[o]["fc"]["mean13"], "tsb_a0.2": lambda o: test[o]["fc"]["tsb_a0.2"]}
    v1 = pd.read_parquet(os.path.join(ROOT, "reports", "forecast", "forecasts_test.parquet"), columns=["unique_id", "origin", "chon_tu_do", "gbm_revenue"])
    for o in TEST_ORIGINS:
        d = v1[v1["origin"] == weeks[o]]
        assert np.array_equal(d["unique_id"].to_numpy()[::H], ids)
        test[o]["fc"]["v1_moi_nhom_tu_chon"] = d["chon_tu_do"].to_numpy().reshape(n_series, H)
        test[o]["fc"]["v1_gbm_doanh_thu"] = d["gbm_revenue"].to_numpy().reshape(n_series, H)
    pickers["v1_moi_nhom_tu_chon"] = lambda o: test[o]["fc"]["v1_moi_nhom_tu_chon"]
    pickers["v1_gbm_doanh_thu"] = lambda o: test[o]["fc"]["v1_gbm_doanh_thu"]
    res_test = {k: full_scores(test, TEST_ORIGINS, pk, seller0, weeks) for k, pk in pickers.items()}
    v1r = json.load(open(os.path.join(ROOT, "reports", "forecast", "results.json"), encoding="utf-8"))
    out = {"setup": {"horizon": H, "val_origins": [str(weeks[o].date()) for o in VAL_ORIGINS], "test_origins": [str(weeks[o].date()) for o in TEST_ORIGINS],
                     "selection_criterion": "điểm VN1 (MAE% + |Bias%| của từng kỳ 13 tuần, trung bình các kỳ) trên mốc xác thực", "n_features_old": len(OLD), "n_features_all": len(ALL),
                     "new_features": [c for c in ALL if c not in OLD], "max_train_rows": MAX_ROWS},
           "validation_study": study, "validation_by_group": study_groups,
           "chosen": {"weight_power": p, "feature_set": "mới" if cfgs[best_mean]["cols"] is ALL else "cũ", "mean_model": best_mean,
                      "mean_model_cfg": cfgs[best_mean]["cfg"], "blend_components": comps,
                      "blend_weights": {GROUPS[c]: w for c, w in weights.items()}},
           "test_composition": {g: v1r["test_composition"][g]["share_test_revenue"] for g in GROUPS},
           "test": res_test}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    rows = []
    for o in TEST_ORIGINS:
        rows.append(pd.DataFrame({"unique_id": np.repeat(ids, H), "origin": weeks[o], "h": np.tile(np.arange(1, H + 1), n_series),
                                  "week": np.tile(weeks[o:o + H].values, n_series), "group": np.repeat(np.array(GROUPS)[test[o]["group"]], H),
                                  "forecast": blended(test, o, weights).ravel(), "actual": test[o]["actual"].ravel()}))
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(OUT, "forecasts_test_v2.parquet"), index=False)
    shutil.rmtree(CACHE, ignore_errors=True)
    make_figures(out)
    print(json.dumps({"chosen": out["chosen"], "test": {k: v["toan_bo"] for k, v in res_test.items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")))
    else:
        main()
