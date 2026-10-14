"""Giai đoạn mô hình hóa: dự báo doanh thu 13 tuần tới theo từng nhóm kiểu nhu cầu (VN1).

Chạy:  python scripts/06_forecast.py                  (khoảng 7 phút)
       python scripts/06_forecast.py --figures-only   (chỉ vẽ lại hình từ results.json)

Đọc clean/ (không sửa), ghi vào reports/forecast/:
  results.json                 mọi con số
  forecasts_test.parquet       dự báo trên 4 mốc kiểm tra: cột forecast (theo quyết định), chon_tu_do, gbm_revenue, naive
  fig01..fig04 *.png

Nguyên tắc
  - Mốc XÁC THỰC (chọn phương pháp): tuần 79, 92, 105, 118. Mốc KIỂM TRA (báo cáo): tuần 131, 144, 157, 170.
    Tập kiểm tra không được dùng để chọn phương pháp hay tham số.
  - Tại mốc o chỉ dùng dữ liệu của các tuần < o. Mô hình học máy chỉ học từ các mốc t có t + 13 <= o.
  - Lịch sử đã cắt: các dòng is_flat_period (nhà bán hàng 0, đến 2022-08-22) không thuộc lịch sử và không được chấm điểm.
  - Nhóm của mỗi chuỗi được gán lại tại từng mốc bằng đúng quy tắc của scripts/03_cluster.py
    (smooth / erratic / intermittent / lumpy; new = dưới 26 tuần lịch sử; dormant = 26 tuần không bán; no_price).
  - Chuỗi chưa ra mắt tại mốc: dự báo 0 ở mọi phương pháp (không thể biết trước), và vẫn được tính vào sai số.

Phương pháp ứng viên (đều cho ra doanh thu của 13 tuần)
  Đơn giản      : naive, mean13, mean_all (trung bình mọi tuần đã có), zero, mean13_season (mean13 x hệ số mùa vụ toàn cục)
  Số lượng x giá: ses, croston, sba, tsb với alpha trong ALPHAS; dự báo số lượng/tuần rồi nhân với giá gần nhất
  Học máy       : gbm_revenue (dự báo doanh thu trực tiếp), gbm_units (dự báo số lượng rồi nhân giá gần nhất); hàm mất mát Poisson

Ba cách ghép
  theo_quyet_dinh : smooth/erratic chọn giữa gbm_revenue và gbm_units; intermittent/lumpy chọn trong croston/sba/tsb;
                    new dùng mean_all; dormant và no_price dùng zero  (các lựa chọn "chọn giữa" lấy theo mốc xác thực)
  chon_tu_do      : mỗi nhóm lấy phương pháp tốt nhất trên mốc xác thực trong TẤT CẢ ứng viên
  mot_mo_hinh     : một phương pháp duy nhất cho mọi chuỗi (để trả lời: chia nhóm có giúp gì không)
"""
import json
import os
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
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "reports", "forecast")
H = 13
VAL_ORIGINS = [79, 92, 105, 118]
TEST_ORIGINS = [131, 144, 157, 170]
ALPHAS = [0.05, 0.1, 0.2]
TRAIN_ORIGIN_MIN, TRAIN_ORIGIN_STEP, MAX_TRAIN_ROWS = 30, 3, 1_200_000
GROUPS = ["smooth", "erratic", "intermittent", "lumpy", "new", "dormant", "no_price", "chua_ra_mat"]
SEED = 0
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------- trạng thái chuỗi tại mốc
def wmean(M, o, w, age):
    """Trung bình w tuần cuối trước mốc o, chỉ tính các tuần thuộc lịch sử (M đã bằng 0 trước history_start)."""
    return M[:, max(0, o - w):o].sum(axis=1) / np.maximum(np.minimum(w, age), 1)


def series_state(o, D):
    Q, start = D["Qh"][:, :o], D["start"]
    exists = start < o
    age = np.where(exists, o - start, 0)
    nz = Q > 0
    n_nz = nz.sum(axis=1)
    mean_nz = Q.sum(axis=1) / n_nz
    cv2 = ((Q ** 2).sum(axis=1) / n_nz - mean_nz ** 2) / mean_nz ** 2
    adi = age / n_nz
    last_sale = np.where(n_nz > 0, o - 1 - np.argmax(nz[:, ::-1], axis=1), -1)
    zero_run = np.where(exists, np.minimum(o - 1 - last_sale, age), 0)
    has_price = ((D["Praw"][:, :o] > 0) & nz).any(axis=1)
    sb = np.where(adi < 1.32, np.where(cv2 < 0.49, 0, 1), np.where(cv2 < 0.49, 2, 3))
    g = np.where(~exists, 7, np.where(age < 26, 4, np.where(zero_run >= 26, 5, np.where(~has_price, 6, sb))))
    r52 = D["Rh"][:, o - 52:o]
    q4 = np.isin(D["month"][o - 52:o], [10, 11, 12])
    q4_share = np.where((age >= 52) & (r52.sum(axis=1) > 0), r52[:, q4].sum(axis=1) / r52.sum(axis=1), np.nan)
    return {"exists": exists, "age": age, "zero_run": zero_run, "group": g, "adi": adi, "cv2": cv2, "q4_share": q4_share,
            "flag_q4": (g < 4) & (q4_share >= 0.40)}


def season_ratios(o, D):
    """Hệ số mùa vụ từ cùng kỳ năm trước trên một nhóm chuỗi CỐ ĐỊNH (đã có lịch sử từ tuần o-65):
    doanh thu tuần o-52+h chia mức nền 13 tuần trước đó. Toàn cục và theo nhà bán hàng (co về toàn cục)."""
    cohort = D["start"] <= o - 65
    Rc = np.where(cohort[:, None], D["Rh"], 0.0)
    tot = Rc.sum(axis=0)
    g = tot[o - 52:o - 52 + H] / tot[o - 65:o - 52].mean()
    n_clients = D["client"].max() + 1
    cr = np.zeros((n_clients, D["Rh"].shape[1]))
    np.add.at(cr, D["client"], Rc)
    base = cr[:, o - 65:o - 52].mean(axis=1)
    c = cr[:, o - 52:o - 52 + H] / base[:, None]
    c = np.where(np.isfinite(c) & (base[:, None] > 0), c, g[None, :])
    return np.clip(g, 0.5, 3.0), np.clip(0.5 * c + 0.5 * g[None, :], 0.5, 3.0)


# ---------------------------------------------------------------- phương pháp cho nhu cầu gián đoạn
def smoothing_family(o, D, alpha):
    """SES, Croston, SBA, TSB trên số lượng tuần, chạy trên lịch sử dùng được. Trả về số lượng dự báo mỗi tuần."""
    Q, start = D["Qh"], D["start"]
    n = len(Q)
    s = np.full(n, np.nan)                                  # SES
    z = np.full(n, np.nan); p = np.full(n, np.nan); q = np.ones(n)   # Croston: cỡ, khoảng cách, bộ đếm
    d = np.full(n, np.nan)                                  # TSB: xác suất có bán
    init_len = np.minimum(13, np.maximum(o - start, 1))
    for t in range(o):
        act = t >= start
        y = Q[:, t]
        sale = act & (y > 0)
        s = np.where(act & np.isnan(s), y, np.where(act, s + alpha * (y - s), s))
        first = sale & np.isnan(z)
        upd = sale & ~first
        p = np.where(first, q, np.where(upd, p + alpha * (q - p), p))
        z = np.where(first, y, np.where(upd, z + alpha * (y - z), z))
        q = np.where(sale, 1, np.where(act, q + 1, q))
        at_start = act & np.isnan(d)
        d = np.where(act & ~at_start, d + alpha * (sale.astype(float) - d), d)
        if at_start.any():                                    # khởi tạo: tỷ lệ tuần có bán trong tối đa 13 tuần đầu
            rows = np.flatnonzero(at_start)
            d[rows] = np.array([(Q[i, t:min(o, t + init_len[i])] > 0).mean() for i in rows])
    cro = z / p
    return {"ses": np.nan_to_num(s), "croston": np.nan_to_num(cro), "sba": np.nan_to_num((1 - alpha / 2) * cro),
            "tsb": np.nan_to_num(d * z)}


# ---------------------------------------------------------------- mô hình học máy
def build_features(o, M, D, with_target):
    st = series_state(o, D)
    age, g = st["age"], st["group"]
    m13, m52 = wmean(M, o, 13, age), wmean(M, o, 52, age)
    scale = np.where(m13 > 0, m13, m52)
    use = st["exists"] & (scale > 0) & (g != 5)
    idx = np.flatnonzero(use)
    s, a, Mi = scale[idx], age[idx], M[idx]
    nzf = (Mi > 0).astype(float)
    f = {"age": np.minimum(a, 156), "zero_run": st["zero_run"][idx], "group": g[idx], "client": D["client"][idx],
         "log_scale": np.log1p(s), "last": Mi[:, o - 1] / s,
         "m4": wmean(Mi, o, 4, a) / s, "m13": wmean(Mi, o, 13, a) / s, "m26": wmean(Mi, o, 26, a) / s, "m52": wmean(Mi, o, 52, a) / s,
         "max13": Mi[:, o - 13:o].max(axis=1) / s, "nz4": wmean(nzf, o, 4, a), "nz13": wmean(nzf, o, 13, a), "nz52": wmean(nzf, o, 52, a),
         "adi": np.minimum(st["adi"][idx], 60), "cv2": np.minimum(np.nan_to_num(st["cv2"][idx]), 20), "q4_share": st["q4_share"][idx]}
    price_last = D["P"][idx, o - 1]
    price_med = np.nanmedian(np.where(D["Qh"][idx, :o] > 0, D["P"][idx, :o], np.nan), axis=1)
    f["log_price"] = np.log(np.where(price_last > 0, price_last, np.nan))
    f["price_rel"] = np.clip(price_last / price_med, 1 / 3, 3)
    n = len(idx)
    X = pd.DataFrame({k: np.repeat(v, H) for k, v in f.items()})
    hh = np.tile(np.arange(1, H + 1), n)
    X["h"] = hh
    X["woy"] = D["woy_ext"][o + hh - 1]
    if o >= 65:
        sg, sc = season_ratios(o, D)
        X["season_global"] = np.tile(sg, n)
        X["season_client"] = sc[D["client"][idx]].ravel()
    else:
        X["season_global"] = np.nan
        X["season_client"] = np.nan
    y = (Mi[:, o:o + H] / s[:, None]).ravel() if with_target else None
    return X.astype(np.float32), y, idx, s


def gbm_forecast(o, M, D):
    rng = np.random.default_rng(SEED)
    Xs, ys = [], []
    for t in range(TRAIN_ORIGIN_MIN, o - H + 1, TRAIN_ORIGIN_STEP):
        X, y, _, _ = build_features(t, M, D, True)
        Xs.append(X); ys.append(y)
    X, y = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    if len(X) > MAX_TRAIN_ROWS:
        keep = rng.choice(len(X), MAX_TRAIN_ROWS, replace=False)
        X, y = X.iloc[keep], y[keep]
    cat = [X.columns.get_loc(c) for c in ("group", "client")]
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.07, max_leaf_nodes=63, min_samples_leaf=100,
                                          l2_regularization=1.0, categorical_features=cat, random_state=SEED)
    model.fit(X, np.clip(y, 0, 30))
    Xo, _, idx, s = build_features(o, M, D, False)
    out = np.zeros((M.shape[0], H))
    out[idx] = np.clip(model.predict(Xo), 0, None).reshape(len(idx), H) * s[:, None]
    return out, len(X)


# ---------------------------------------------------------------- tất cả dự báo tại một mốc
def all_forecasts(o, D):
    st = series_state(o, D)
    age, exists = st["age"], st["exists"]
    rep = lambda v: np.repeat(np.asarray(v, dtype=float)[:, None], H, axis=1)
    price = np.nan_to_num(D["P"][:, o - 1])
    m13 = wmean(D["Rh"], o, 13, age)
    fc = {"naive": rep(D["Rh"][:, o - 1]), "mean13": rep(m13), "mean_all": rep(wmean(D["Rh"], o, 10 ** 6, age)),
          "zero": np.zeros((len(age), H))}
    if o >= 65:
        sg, _ = season_ratios(o, D)
        fc["mean13_season"] = rep(m13) * sg[None, :]
    for a in ALPHAS:
        for name, units in smoothing_family(o, D, a).items():
            fc[f"{name}_a{a}"] = rep(units * price)
    n_train = {}
    fc["gbm_revenue"], n_train["revenue"] = gbm_forecast(o, D["Rh"], D)
    units, n_train["units"] = gbm_forecast(o, D["Qh"], D)
    fc["gbm_units"] = units * price[:, None]
    for k in fc:
        fc[k] = np.where(exists[:, None], fc[k], 0.0)
    return fc, st, n_train


def run(origins, D, label):
    store = {}
    for o in origins:
        t0 = time.time()
        fc, st, n_train = all_forecasts(o, D)
        # Thang đo RMSSE: trung bình bình phương chênh lệch tuần-qua-tuần của doanh thu trong lịch sử dùng được trước mốc
        diff2 = (D["Rh"][:, 1:o] - D["Rh"][:, :o - 1]) ** 2
        ok = np.arange(o - 1)[None, :] >= D["start"][:, None]
        n_diff = ok.sum(axis=1)
        scale = np.where(n_diff > 0, (diff2 * ok).sum(axis=1) / np.maximum(n_diff, 1), 0.0)
        store[o] = {"scale": scale, "weight": D["Rh"][:, o - 13:o].sum(axis=1),
                    "fc": fc, "group": st["group"], "flag_q4": st["flag_q4"], "actual": D["R"][:, o:o + H],
                    "valid": ~D["F"][:, o:o + H], "n_train": n_train}
        print(f"[{label}] mốc {o} xong sau {time.time() - t0:.0f}s", flush=True)
    return store


def score(store, origins, pick, mask=None):
    """pick(o) -> ma trận dự báo. Trả về WMAPE cấp chuỗi, độ lệch, WMAPE cấp tổng, tổng thực tế, và các thước đo bình phương:
    rmse (gộp mọi ô), rmsse của từng chuỗi = sqrt(MSE 13 tuần / thang đo lịch sử) rồi lấy trung vị và trung bình có trọng số
    doanh thu 13 tuần trước mốc (wrmsse). Chuỗi không có thang đo (chưa có lịch sử, hoặc lịch sử không đổi) không tính được RMSSE."""
    ab = act = bias = tab = tact = 0.0
    se = n_cells = act_scaled = 0.0
    rs, ws = [], []
    for o in origins:
        s = store[o]
        m = s["valid"] if mask is None else s["valid"] & mask(o)[:, None]
        err = np.where(m, pick(o) - s["actual"], 0.0)
        a = np.where(m, s["actual"], 0.0)
        ab += np.abs(err).sum(); act += a.sum(); bias += err.sum()
        tab += np.abs(err.sum(axis=0)).sum(); tact += a.sum()
        se += (err ** 2).sum(); n_cells += m.sum()
        cnt = m.sum(axis=1)
        use = (cnt > 0) & (s["scale"] > 0)
        rs.append(np.sqrt((err[use] ** 2).sum(axis=1) / cnt[use] / s["scale"][use])); ws.append(s["weight"][use])
        act_scaled += a[use].sum()
    r, w = np.concatenate(rs), np.concatenate(ws)
    return {"wmape": ab / act if act else None, "bias": bias / act if act else None, "wmape_total_level": tab / tact if tact else None,
            "abs_err": ab, "actual": act,
            "rmse": float(np.sqrt(se / n_cells)) if n_cells else None,
            "rmsse_median": float(np.median(r)) if len(r) else None,
            "wrmsse": float((r * w).sum() / w.sum()) if len(r) and w.sum() > 0 else None,
            "rmsse_n": int(len(r)), "rmsse_revenue_coverage": act_scaled / act if act else None}


def compose(store, o, choice):
    g = store[o]["group"]
    out = np.zeros_like(store[o]["actual"])
    for code, name in choice.items():
        out[g == code] = store[o]["fc"][name][g == code]
    return out


LABELS = {"theo_quyet_dinh": "Ghép theo quyết định", "chon_tu_do": "Mỗi nhóm tự chọn", "gbm_revenue": "GBM doanh thu trực tiếp",
          "gbm_units": "GBM số lượng x giá", "naive": "Naive", "mean13": "Trung bình 13 tuần", "mean13_season": "Trung bình 13 tuần x mùa vụ",
          "mean_all": "Trung bình mọi tuần"}


def label(m):
    if m in LABELS:
        return LABELS[m]
    fam, a = m.split("_a")
    return f"{fam.upper() if fam != 'croston' else 'Croston'} (alpha {a})"


def make_figures(out):
    """Vẽ lại mọi hình từ results.json (không cần chạy lại mô hình)."""
    res_val, res_test, comp = out["validation"], out["test"], out["test_composition"]
    methods = [m for m in res_test if m not in ("theo_quyet_dinh", "chon_tu_do")]
    BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10, "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 130, "legend.frameon": False})
    best_alpha = {fam: min((m for m in methods if m.startswith(fam + "_a")), key=lambda m: res_val[m]["toan_bo"]["wmape"])
                  for fam in ("ses", "croston", "sba", "tsb")}
    show = ["theo_quyet_dinh", "chon_tu_do", "gbm_revenue", "gbm_units", "naive", "mean13", "mean13_season"] + list(best_alpha.values())
    show = sorted(show, key=lambda m: res_test[m]["toan_bo"]["wmape"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vals = [res_test[m]["toan_bo"]["wmape"] * 100 for m in show]
    colors = [AQUA if m in ("theo_quyet_dinh", "chon_tu_do") else BLUE if m.startswith("gbm") else GRAY for m in show]
    bars = ax.barh(range(len(show)), vals, color=colors, height=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(show)), [label(m) for m in show]); ax.invert_yaxis(); ax.set_xlabel("MAE% cấp chuỗi trên 4 mốc kiểm tra (%)")
    ax.set_title("Sai số theo phương pháp")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_wmape_theo_phuong_phap.png")); plt.close(fig)

    trio = [("naive", GRAY), ("gbm_revenue", BLUE), ("chon_tu_do", AQUA)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(5)
    for j, (m, color) in enumerate(trio):
        v = [min((res_test[m]["theo_nhom"][g]["wmape"] or 0) * 100, 200) for g in GROUPS[:5]]
        bars = ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=label(m))
        for b, val_ in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val_ + 1.5, f"{val_:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{g}\n{comp[g]['share_test_revenue']:.0%} doanh thu" for g in GROUPS[:5]])
    ax.set_ylim(0, 105); ax.set_ylabel("MAE%"); ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_title("Sai số theo nhóm kiểu nhu cầu trên mốc kiểm tra")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_wmape_theo_nhom.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(3)
    for j, (m, color) in enumerate(trio):
        v = [res_test[m][k]["wmape"] * 100 for k in ("toan_bo", "nha_ban_hang_0", "con_lai")]
        bars = ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=label(m))
        for b, val_ in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val_ + 0.8, f"{val_:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Toàn bộ", "Nhà bán hàng 0", "45 nhà bán hàng còn lại"]); ax.set_ylim(0, 70); ax.set_ylabel("MAE%")
    ax.legend(loc="upper left", ncol=3, fontsize=9); ax.set_title("Sai số tách theo nhà bán hàng")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig03_wmape_tach_nha_ban_hang.png")); plt.close(fig)

    origins = list(res_test["naive"]["theo_moc"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(len(origins))
    for ax, key, title in ((axes[0], "wmape", "MAE% cấp chuỗi theo từng mốc (%)"), (axes[1], "bias", "Độ lệch theo từng mốc (%, dương = dự báo cao)")):
        for j, (m, color) in enumerate(trio):
            v = [res_test[m]["theo_moc"][o][key] * 100 for o in origins]
            bars = ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=label(m))
            for b, val_ in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val_ + (1 if val_ >= 0 else -3.2), f"{val_:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, [f"mốc {pd.Timestamp(o).strftime('%d/%m/%Y')}" for o in origins], fontsize=8); ax.set_title(title)
        ax.axhline(0, color=MUTED, lw=1)
    axes[0].set_ylim(0, 78); axes[0].legend(loc="upper center", ncol=3, fontsize=8)
    axes[1].set_ylim(-26, 24)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig04_theo_tung_moc.png")); plt.close(fig)



def load_data():
    """Đọc bảng chính và dựng các ma trận chuỗi x tuần dùng chung cho mọi phương pháp."""
    df = pd.read_parquet(os.path.join(CLEAN, "vn1_long.parquet"))
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    ids = np.sort(df["unique_id"].unique())
    piv = lambda col: df.pivot(index="unique_id", columns="week", values=col).reindex(index=ids, columns=weeks)
    meta = df.groupby("unique_id").agg(client=("client", "first"), warehouse=("warehouse", "first"), product=("product", "first"),
                                       start=("history_start_week", "first")).loc[ids]
    R = piv("revenue").fillna(0.0).to_numpy()
    Q = piv("sales").fillna(0.0).to_numpy()
    F = piv("is_flat_period").fillna(False).to_numpy(dtype=bool)
    start = np.searchsorted(weeks.values, meta["start"].values)
    hist = np.arange(len(weeks))[None, :] >= start[:, None]
    ext = pd.date_range(weeks[0], periods=len(weeks) + H, freq="7D")
    D = {"R": R, "F": F, "Rh": np.where(hist, R, 0.0), "Qh": np.where(hist, Q, 0.0), "P": piv("price").to_numpy(),
         "Praw": np.where(hist, piv("price_raw").to_numpy(), np.nan), "start": start, "client": pd.factorize(meta["client"])[0],
         "weeks": weeks, "month": weeks.month.to_numpy(), "woy_ext": ext.isocalendar().week.to_numpy().astype(int)}
    D["wh"] = pd.factorize(meta["warehouse"])[0]
    D["prod"] = pd.factorize(meta["product"])[0]
    seller0 = (meta["client"] == 0).to_numpy()
    return D, ids, weeks, seller0


def main():
    os.makedirs(OUT, exist_ok=True)
    D, ids, weeks, seller0 = load_data()

    # Kiểm tra: nhóm tính ở đây phải trùng với clean/clusters/assignments.parquet tại 4 mốc kiểm tra
    asg = pd.read_parquet(os.path.join(CLEAN, "clusters", "assignments.parquet"))
    agree = {}
    for o in TEST_ORIGINS:
        a = asg[asg["origin"] == o].set_index("unique_id").loc[ids, "label_A"].to_numpy()
        g = np.array(GROUPS)[series_state(o, D)["group"]]
        g = np.where(g == "chua_ra_mat", "new", g)
        agree[o] = float((a == g).mean())
    print("khớp nhãn với assignments.parquet:", agree, flush=True)

    val = run(VAL_ORIGINS, D, "xác thực")
    test = run(TEST_ORIGINS, D, "kiểm tra")
    methods = sorted(set.intersection(*(set(val[o]["fc"]) for o in VAL_ORIGINS)))

    # ---- chọn phương pháp trên mốc xác thực ----
    def val_err(method, code):
        return score(val, VAL_ORIGINS, lambda o: val[o]["fc"][method], lambda o: val[o]["group"] == code)["abs_err"]

    table_val = {GROUPS[c]: {m: val_err(m, c) for m in methods} for c in range(7)}
    best = lambda code, cands: min(cands, key=lambda m: table_val[GROUPS[code]][m])
    crost = [m for m in methods if m.split("_a")[0] in ("croston", "sba", "tsb")]
    choice_decision = {0: best(0, ["gbm_revenue", "gbm_units"]), 1: best(1, ["gbm_revenue", "gbm_units"]),
                       2: best(2, crost), 3: best(3, crost), 4: "mean_all", 5: "zero", 6: "zero", 7: "zero"}
    choice_free = {c: best(c, methods) for c in range(6)} | {6: "zero", 7: "zero"}
    single_best = min(methods, key=lambda m: score(val, VAL_ORIGINS, lambda o: val[o]["fc"][m])["wmape"])

    # ---- chấm điểm ----
    def evaluate(store, origins):
        res = {}
        pickers = {m: (lambda o, m=m: store[o]["fc"][m]) for m in methods}
        pickers["theo_quyet_dinh"] = lambda o: compose(store, o, choice_decision)
        pickers["chon_tu_do"] = lambda o: compose(store, o, choice_free)
        for name, pk in pickers.items():
            r = {"toan_bo": score(store, origins, pk),
                 "nha_ban_hang_0": score(store, origins, pk, lambda o: seller0),
                 "con_lai": score(store, origins, pk, lambda o: ~seller0),
                 "co_mua_vu_q4": score(store, origins, pk, lambda o: store[o]["flag_q4"]),
                 "theo_nhom": {GROUPS[c]: score(store, origins, pk, lambda o, c=c: store[o]["group"] == c) for c in range(8)},
                 "theo_moc": {str(D["weeks"][o].date()): {k: v for k, v in score(store, [o], pk).items() if k in ("wmape", "bias", "wmape_total_level")}
                              for o in origins}}
            res[name] = r
        return res

    res_val, res_test = evaluate(val, VAL_ORIGINS), evaluate(test, TEST_ORIGINS)
    tot_act = res_test["naive"]["toan_bo"]["actual"]
    comp = {GROUPS[c]: {"share_test_revenue": res_test["naive"]["theo_nhom"][GROUPS[c]]["actual"] / tot_act,
                        "series_per_origin": float(np.mean([(test[o]["group"] == c).sum() for o in TEST_ORIGINS]))} for c in range(8)}
    out = {"setup": {"horizon": H, "val_origins": [str(weeks[o].date()) for o in VAL_ORIGINS], "test_origins": [str(weeks[o].date()) for o in TEST_ORIGINS],
                     "metric": "WMAPE = tổng |sai số| / tổng doanh thu thực tế, gộp 4 mốc x 13 tuần; ô thuộc giai đoạn phẳng không được chấm",
                     "alphas": ALPHAS, "label_agreement_with_assignments": agree,
                     "gbm_train_rows": {str(o): test[o]["n_train"] for o in TEST_ORIGINS}},
           "choice_theo_quyet_dinh": {GROUPS[c]: m for c, m in choice_decision.items()},
           "choice_chon_tu_do": {GROUPS[c]: m for c, m in choice_free.items()},
           "best_single_method_on_validation": single_best,
           "test_composition": comp, "validation": res_val, "test": res_test}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    rows = []
    for o in TEST_ORIGINS:
        n = len(ids)
        rows.append(pd.DataFrame({"unique_id": np.repeat(ids, H), "origin": weeks[o], "h": np.tile(np.arange(1, H + 1), n),
                                  "week": np.tile(weeks[o:o + H].values, n), "group": np.repeat(np.array(GROUPS)[test[o]["group"]], H),
                                  "method": np.repeat(np.array([choice_decision[c] for c in test[o]["group"]]), H),
                                  "forecast": compose(test, o, choice_decision).ravel(),
                                  "chon_tu_do": compose(test, o, choice_free).ravel(), "gbm_revenue": test[o]["fc"]["gbm_revenue"].ravel(),
                                  "naive": test[o]["fc"]["naive"].ravel(), "actual": test[o]["actual"].ravel()}))
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(OUT, "forecasts_test.parquet"), index=False)

    make_figures(out)
    show = ["theo_quyet_dinh", "chon_tu_do", "gbm_revenue", "gbm_units", "naive", "mean13"]

    r3 = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                    if k in ("wmape", "bias", "wmape_total_level", "rmse", "rmsse_median", "wrmsse", "rmsse_revenue_coverage")}
    brief = {"agree": agree, "choice_decision": out["choice_theo_quyet_dinh"], "choice_free": out["choice_chon_tu_do"], "single_best_val": single_best,
             "comp": {k: round(v["share_test_revenue"], 4) for k, v in comp.items()},
             "val_all": {m: r3(res_val[m]["toan_bo"]) for m in res_val},
             "test_all": {m: r3(res_test[m]["toan_bo"]) for m in res_test},
             "test_seller0": {m: r3(res_test[m]["nha_ban_hang_0"]) for m in show},
             "test_rest": {m: r3(res_test[m]["con_lai"]) for m in show},
             "test_q4": {m: r3(res_test[m]["co_mua_vu_q4"]) for m in show},
             "test_by_group": {m: {g: (round(res_test[m]["theo_nhom"][g]["wmape"], 3) if res_test[m]["theo_nhom"][g]["wmape"] is not None else None)
                                   for g in GROUPS} for m in res_test}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8")))
    else:
        main()
