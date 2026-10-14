"""EDA cho dữ liệu VN1 đã làm sạch.

Chạy:  python scripts/02_eda.py

Đọc clean/ (không sửa), ghi vào reports/eda/:
  fig01..fig11 *.png   hình minh họa
  eda_numbers.json     mọi con số dùng trong báo cáo

Quy ước giai đoạn: TRAIN = 131 tuần đầu (đến 2023-01-02), TEST = 52 tuần cuối.
Các thống kê mô tả hành vi chuỗi (thưa, giá, đột biến...) chỉ tính trên TRAIN để không nhìn trước tập kiểm tra.

Cắt lịch sử: các dòng có cờ is_flat_period (giai đoạn số liệu phẳng của nhà bán hàng 0, đến 2022-08-22) KHÔNG được dùng
cho thống kê hành vi chuỗi (mục 4-9) và cho baseline cấp chuỗi; lịch sử của các chuỗi đó bắt đầu từ history_start_week.
Các con số tổng (mục 1-3: doanh thu theo thời gian, mùa vụ, mức tập trung) vẫn dùng toàn bộ dữ liệu vì tổng của giai đoạn phẳng
được coi là đúng; riêng mùa vụ được tính thêm một bản không có nhà bán hàng bị cắt.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "clean")
OUT = os.path.join(ROOT, "reports", "eda")
TEST_WEEKS, HORIZON = 52, 13

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "figure.dpi": 130, "legend.frameon": False,
})
NUM = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def millions(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def top_share(values, pct):
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(round(len(v) * pct)))
    return float(v[:k].sum() / v.sum())


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def acf(x, lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    return float((x[lag:] * x[:-lag]).sum() / (x * x).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(CLEAN, "vn1_long.parquet"))
    summ = pd.read_csv(os.path.join(CLEAN, "series_summary.csv"), parse_dates=["first_sale_week", "history_start_week"]).set_index("unique_id")
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    n_weeks = len(weeks)
    n_train = n_weeks - TEST_WEEKS
    split_date = weeks[n_train]
    df["rev"] = df["revenue"].fillna(0.0)
    train = df[~df["is_test"]]                       # mọi dòng huấn luyện (dùng cho các con số tổng)
    train_h = train[~train["is_flat_period"]]        # lịch sử dùng được (dùng cho hành vi chuỗi)
    test = df[df["is_test"]]

    # Ma trận chuỗi x tuần (0 trước tuần bán đầu tiên)
    R = df.pivot(index="unique_id", columns="week", values="rev").reindex(columns=weeks).fillna(0.0)
    Q = df.pivot(index="unique_id", columns="week", values="sales").reindex(columns=weeks).fillna(0.0)
    ids = R.index
    Rv, Qv = R.to_numpy(), Q.to_numpy()
    first_idx = np.searchsorted(weeks.values, summ.loc[ids, "first_sale_week"].values)
    active = np.arange(n_weeks)[None, :] >= first_idx[:, None]
    start_idx = np.searchsorted(weeks.values, summ.loc[ids, "history_start_week"].values)
    active_h = np.arange(n_weeks)[None, :] >= start_idx[:, None]          # từ history_start_week
    flat_cell = active & ~active_h
    flat_clients = sorted(int(c) for c in summ.loc[ids, "client"][(summ.loc[ids, "n_flat_weeks"] > 0).to_numpy()].unique())
    cflat = summ.loc[ids, "client"].isin(flat_clients).to_numpy()
    NUM["history_cut"] = {
        "flat_clients": flat_clients, "n_series": int(cflat.sum()), "flat_rows": int(flat_cell.sum()),
        "history_starts": str(weeks[int(start_idx[cflat].min())].date()) if cflat.any() else None,
        "share_of_train_rows": float(flat_cell[:, :n_train].sum() / active[:, :n_train].sum()),
        "share_train_revenue_flat_clients": float(Rv[cflat, :n_train].sum() / Rv[:, :n_train].sum()),
        "share_test_revenue_flat_clients": float(Rv[cflat, n_train:].sum() / Rv[:, n_train:].sum()),
        "share_train_revenue_in_flat_rows": float(np.where(flat_cell, Rv, 0)[:, :n_train].sum() / Rv[:, :n_train].sum()),
    }

    # ---------- 1. Tổng quan theo thời gian ----------
    tot_rev, tot_units = Rv.sum(axis=0), Qv.sum(axis=0)
    n_active = active.sum(axis=0)
    n_selling = (Qv > 0).sum(axis=0)
    year = weeks.year
    yearly = pd.DataFrame({"rev": tot_rev, "units": tot_units}, index=weeks).groupby(year).sum()
    h2 = pd.Series(tot_rev, index=weeks)[weeks.month >= 7].groupby(year[weeks.month >= 7]).sum()
    cohort2020 = first_idx < int((year == 2020).sum())
    cohort_rev = pd.DataFrame(Rv[cohort2020].sum(axis=0)[None, :], columns=weeks).T.groupby(year).sum()[0]
    NUM["overview"] = {
        "period": "toàn bộ 183 tuần",
        "revenue_by_year": {int(y): float(v) for y, v in yearly["rev"].items()},
        "units_by_year": {int(y): float(v) for y, v in yearly["units"].items()},
        "yoy_full_year": {"2022_vs_2021": float(yearly.loc[2022, "rev"] / yearly.loc[2021, "rev"] - 1),
                          "2023_vs_2022": float(yearly.loc[2023, "rev"] / yearly.loc[2022, "rev"] - 1)},
        "h2_revenue_by_year": {int(y): float(v) for y, v in h2.items() if y < 2024},
        "active_series": {str(weeks[i].date()): int(n_active[i]) for i in (0, 26, 78, 130, 182)},
        "selling_series_share_last_week": float(n_selling[-1] / n_active[-1]),
        "cohort2020_n_series": int(cohort2020.sum()),
        "cohort2020_revenue_by_year": {int(y): float(v) for y, v in cohort_rev.items()},
        "cohort2020_share_of_2023_revenue": float(cohort_rev[2023] / yearly.loc[2023, "rev"]),
        "revenue_per_active_series_week": {int(y): float(tot_rev[year == y].sum() / n_active[year == y].sum())
                                           for y in (2020, 2021, 2022, 2023)},
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for ax, y, title, color in ((axes[0], tot_rev, "Tổng doanh thu mỗi tuần (triệu)", BLUE),
                                (axes[1], tot_units, "Tổng số lượng bán mỗi tuần (triệu đơn vị)", BLUE)):
        ax.plot(weeks, y, color=color)
        ax.set_title(title)
        millions(ax)
        ax.set_ylim(0)
    axes[2].plot(weeks, n_active, color=BLUE, label="Chuỗi đã ra mắt")
    axes[2].plot(weeks, n_selling, color=ORANGE, label="Chuỗi có bán trong tuần")
    axes[2].set_title("Số chuỗi")
    axes[2].legend(loc="upper left")
    axes[2].set_ylim(0)
    for ax in axes:
        ax.axvspan(split_date, weeks[-1], color=GRAY, alpha=0.12, lw=0)
    axes[0].text(split_date, axes[0].get_ylim()[1] * 0.93, "  tập kiểm tra (52 tuần)", color=MUTED, fontsize=9)
    save(fig, "fig01_tong_quan_theo_thoi_gian.png")

    # ---------- 2. Mùa vụ ----------
    s_rev = pd.Series(tot_rev, index=weeks)
    trend = s_rev.rolling(52, center=True).mean()
    ratio = (s_rev / trend).dropna()                       # đã khử xu hướng
    month_idx = ratio.groupby(ratio.index.month).mean()
    woy_idx = ratio.groupby(ratio.index.isocalendar().week.values).mean()
    per_series = s_rev / n_active                          # doanh thu trên mỗi chuỗi đã ra mắt
    yoy_month = {}
    for y in (2021, 2022, 2023):
        m = per_series[per_series.index.year == y]
        yoy_month[y] = (m.groupby(m.index.month).mean() / m.mean())
    diff = np.diff(np.log(tot_rev))
    s_rev_o = pd.Series(Rv[~cflat].sum(axis=0), index=weeks)       # không có nhà bán hàng bị cắt lịch sử
    ratio_o = (s_rev_o / s_rev_o.rolling(52, center=True).mean()).dropna()
    month_idx_o = ratio_o.groupby(ratio_o.index.month).mean()
    woy_idx_o = ratio_o.groupby(ratio_o.index.isocalendar().week.values).mean()
    wk = s_rev.index.isocalendar().week.values
    NUM["seasonality"] = {
        "period": "chỉ số tháng/tuần: tổng doanh thu chia trung bình trượt 52 tuần, các tuần 2021-01 .. 2023-07",
        "month_index": {int(k): float(v) for k, v in month_idx.items()},
        "peak_weeks": {int(k): float(v) for k, v in woy_idx.sort_values(ascending=False).head(5).items()},
        "trough_weeks": {int(k): float(v) for k, v in woy_idx.sort_values().head(5).items()},
        "q4_vs_rest": float(month_idx[[10, 11, 12]].mean() / month_idx[[1, 2, 3, 4, 5, 6, 7, 8, 9]].mean()),
        "nov_index": float(month_idx[11]),
        "month_index_per_series_by_year": {int(y): {int(k): float(v) for k, v in s.items()} for y, s in yoy_month.items()},
        "acf_total_revenue_train": {f"lag{k}": acf(tot_rev[:n_train], k) for k in (1, 4, 13, 52)},
        "acf_log_diff_train": {f"lag{k}": acf(diff[:n_train - 1], k) for k in (1, 4, 13, 52)},
        "black_friday_week_vs_prev4": {int(y): float(s_rev[(s_rev.index.year == y) & (s_rev.index.isocalendar().week.values == 47)].iloc[0]
                                                    / s_rev[(s_rev.index.year == y) & (np.isin(s_rev.index.isocalendar().week.values, [43, 44, 45, 46]))].mean())
                                       for y in (2020, 2021, 2022, 2023)},
        "excluding_flat_clients": {
            "month_index": {int(k): float(v) for k, v in month_idx_o.items()},
            "peak_weeks": {int(k): float(v) for k, v in woy_idx_o.sort_values(ascending=False).head(5).items()},
            "q4_vs_rest": float(month_idx_o[[10, 11, 12]].mean() / month_idx_o[[1, 2, 3, 4, 5, 6, 7, 8, 9]].mean()),
            "black_friday_week_vs_prev4": {int(y): float(s_rev_o[(s_rev_o.index.year == y) & (wk == 47)].iloc[0]
                                                        / s_rev_o[(s_rev_o.index.year == y) & np.isin(wk, [43, 44, 45, 46])].mean())
                                           for y in (2020, 2021, 2022, 2023)},
            "acf_total_revenue_train": {f"lag{k}": acf(s_rev_o.to_numpy()[:n_train], k) for k in (1, 4, 13, 52)},
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(month_idx.index - 0.2, month_idx.values, color=BLUE, width=0.38, label="Mọi nhà bán hàng")
    axes[0].bar(month_idx_o.index + 0.2, month_idx_o.values, color=ORANGE, width=0.38, label="Không có nhà bán hàng 0")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].axhline(1, color=MUTED, lw=1)
    axes[0].set_title("Chỉ số mùa vụ theo tháng (1 = mức xu hướng)")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_ylim(0.6, max(1.3, month_idx_o.max() * 1.12))
    axes[1].plot(woy_idx.index, woy_idx.values, color=BLUE)
    axes[1].plot(woy_idx_o.index, woy_idx_o.values, color=ORANGE)
    axes[1].axhline(1, color=MUTED, lw=1)
    axes[1].set_title("Chỉ số mùa vụ theo tuần ISO trong năm")
    axes[1].set_xlabel("tuần ISO")
    save(fig, "fig02_mua_vu.png")

    # ---------- 3. Mức độ tập trung (TRAIN) ----------
    tr_rev = train.groupby("unique_id")["rev"].sum()
    tr_rev = tr_rev[tr_rev > 0]
    by_client = train.groupby("client")["rev"].sum()
    by_wh = train.groupby("warehouse")["rev"].sum()
    by_prod = train.groupby("product")["rev"].sum()
    prod_per_client = df.groupby("client")["product"].nunique()
    series_per_prod = df.groupby("product")["unique_id"].nunique()
    wh_per_client = df.groupby("client")["warehouse"].nunique()
    NUM["concentration"] = {
        "period": "TRAIN",
        "n_series_with_train_revenue": int(len(tr_rev)),
        "series_top_share": {p: top_share(tr_rev, q) for p, q in (("1%", .01), ("5%", .05), ("20%", .2), ("50%", .5))},
        "series_gini": gini(tr_rev),
        "client_top_share": {"top1": top_share(by_client, 1 / len(by_client)), "top5": top_share(by_client, 5 / len(by_client)),
                             "top10": top_share(by_client, 10 / len(by_client))},
        "warehouse_top_share": {"top1%": top_share(by_wh, .01), "top5%": top_share(by_wh, .05), "top20%": top_share(by_wh, .2)},
        "top5_clients": {int(k): float(v / by_client.sum()) for k, v in by_client.sort_values(ascending=False).head(5).items()},
        "products_per_client": {"median": float(prod_per_client.median()), "min": int(prod_per_client.min()),
                                "max": int(prod_per_client.max())},
        "warehouses_per_client": {"median": float(wh_per_client.median()), "max": int(wh_per_client.max())},
        "series_per_product": {"share_single_series": float((series_per_prod == 1).mean()), "max": int(series_per_prod.max()),
                               "mean": float(series_per_prod.mean())},
        "n_products_with_train_revenue": int((by_prod > 0).sum()),
        "median_series_train_revenue": float(tr_rev.median()),
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for vals, label, color in ((tr_rev, f"Chuỗi ({len(tr_rev):,})".replace(",", "."), BLUE),
                               (by_wh[by_wh > 0], f"Kho ({(by_wh > 0).sum()})", ORANGE),
                               (by_client[by_client > 0], f"Nhà bán hàng ({(by_client > 0).sum()})", AQUA)):
        v = np.sort(vals.to_numpy())[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v) * 100, v.cumsum() / v.sum() * 100, color=color, label=label)
    ax.plot([0, 100], [0, 100], color=GRAY, lw=1, ls="--")
    ax.set_xlabel("% số đơn vị, xếp từ lớn đến nhỏ")
    ax.set_ylabel("% doanh thu cộng dồn")
    ax.set_title("Đường Pareto doanh thu (giai đoạn huấn luyện)")
    ax.legend(loc="lower right")
    save(fig, "fig03_pareto.png")

    # ---------- 4. Tính gián đoạn (TRAIN, >= 26 tuần hoạt động trong train) ----------
    Qt, At = np.where(active_h, Qv, 0.0)[:, :n_train], active_h[:, :n_train]     # chỉ lịch sử dùng được
    n_act_tr = At.sum(axis=1)
    n_nz = (Qt > 0).sum(axis=1)
    elig = n_act_tr >= 26
    Qnz = np.where(Qt > 0, Qt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_nz = np.nanmean(Qnz, axis=1)
        cv2 = (np.nanstd(Qnz, axis=1) / mean_nz) ** 2
        adi = n_act_tr / n_nz
        zero_share = 1 - n_nz / n_act_tr
    klass = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"), np.where(cv2 < 0.49, "intermittent", "lumpy"))
    inter = pd.DataFrame({"adi": adi, "cv2": cv2, "zero_share": zero_share, "class": klass,
                          "rev": np.where(active_h, Rv, 0.0)[:, :n_train].sum(axis=1), "mean_nz": mean_nz}, index=ids)[elig]
    cls = inter.groupby("class").agg(n=("adi", "size"), rev=("rev", "sum"), zero_share=("zero_share", "median"),
                                     mean_nz=("mean_nz", "median"))
    NUM["intermittency"] = {
        "period": "TRAIN, từ history_start_week, chuỗi có >= 26 tuần lịch sử dùng được trong train",
        "n_series": int(elig.sum()),
        "zero_share_quantiles": {q: float(inter["zero_share"].quantile(float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
        "share_series_zero_over_half": float((inter["zero_share"] > 0.5).mean()),
        "share_series_never_zero": float((inter["zero_share"] == 0).mean()),
        "adi_median": float(inter["adi"].median()), "cv2_median": float(inter["cv2"].median()),
        "classes": {k: {"n": int(r["n"]), "share_series": float(r["n"] / elig.sum()),
                        "share_revenue": float(r["rev"] / cls["rev"].sum()),
                        "median_zero_share": float(r["zero_share"]), "median_nonzero_size": float(r["mean_nz"])}
                    for k, r in cls.iterrows()},
        "median_weekly_units_when_selling": float(inter["mean_nz"].median()),
        "share_series_mean_nonzero_below_5": float((inter["mean_nz"] < 5).mean()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.scatter(inter["adi"], inter["cv2"].clip(lower=0.01), s=4, color=BLUE, alpha=0.18, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(1.32, color=INK, lw=1); ax.axhline(0.49, color=INK, lw=1)
    for (x, y, k) in ((1.02, 0.013, "smooth"), (1.02, 9, "erratic"), (12, 0.013, "intermittent"), (12, 9, "lumpy")):
        ax.text(x, y, f"{k}\n{NUM['intermittency']['classes'][k]['share_series']:.0%} chuỗi", fontsize=9, color=INK)
    ax.set_xlabel("ADI: số tuần trung bình giữa hai lần có bán (log)")
    ax.set_ylabel("CV² của lượng bán khi có bán (log)")
    ax.set_title("Phân loại Syntetos-Boylan")
    order = ["smooth", "erratic", "intermittent", "lumpy"]
    y = np.arange(4)
    axes[1].barh(y + 0.2, [NUM["intermittency"]["classes"][k]["share_series"] * 100 for k in order], height=0.36, color=BLUE, label="% số chuỗi")
    axes[1].barh(y - 0.2, [NUM["intermittency"]["classes"][k]["share_revenue"] * 100 for k in order], height=0.36, color=ORANGE, label="% doanh thu")
    axes[1].set_yticks(y, order); axes[1].invert_yaxis(); axes[1].legend(); axes[1].set_title("Tỷ trọng theo nhóm (%)")
    save(fig, "fig04_gian_doan.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(inter["zero_share"] * 100, bins=40, color=BLUE, rwidth=0.9)
    ax.set_xlabel("% số tuần bằng 0 sau tuần bán đầu tiên")
    ax.set_ylabel("số chuỗi")
    ax.set_title("Phân bố tỷ lệ tuần không bán của mỗi chuỗi (huấn luyện)")
    save(fig, "fig05_ty_le_tuan_bang_0.png")

    # ---------- 5. Giá (TRAIN, chỉ dùng tuần có giá quan sát được) ----------
    obs = train_h[(train_h["sales"] > 0) & (train_h["price_raw"] > 0)].sort_values(["unique_id", "week"])
    g = obs.groupby("unique_id")
    med_price = g["price_raw"].median()
    prev_p, prev_s, prev_w = g["price_raw"].shift(), g["sales"].shift(), g["week"].shift()
    consec = (obs["week"] - prev_w).dt.days == 7
    dlp = np.log(obs["price_raw"] / prev_p)[consec]
    dls = np.log(obs["sales"] / prev_s)[consec]
    changed = dlp.abs() > 0.01
    n_obs = g.size()
    n_changes = (np.log(obs["price_raw"] / prev_p).abs() > 0.01).groupby(obs["unique_id"]).sum()
    rng = g["price_raw"].max() / g["price_raw"].min()
    slope = float(np.polyfit(dlp[changed], dls[changed], 1)[0])
    rel = obs["price_raw"] / obs["unique_id"].map(med_price)
    size_rel = obs["sales"] / obs["unique_id"].map(g["sales"].median())
    disc = rel < 0.9
    bins = pd.cut(dlp[changed], [-10, -0.3, -0.1, -0.03, 0.03, 0.1, 0.3, 10])
    binned = dls[changed].groupby(bins, observed=True).agg(["median", "size"])
    NUM["price"] = {
        "period": "TRAIN, các tuần có bán và giá > 0",
        "median_price_quantiles": {q: float(med_price.quantile(float(q))) for q in ("0.01", "0.1", "0.25", "0.5", "0.75", "0.9", "0.99")},
        "share_series_any_price_change": float((n_changes > 0).mean()),
        "price_changes_per_observed_week": float(n_changes.sum() / (n_obs - 1).clip(lower=0).sum()),
        "median_changes_per_series": float(n_changes.median()),
        "share_series_max_min_ratio_over_2": float((rng > 2).mean()),
        "share_series_max_min_ratio_over_10": float((rng > 10).mean()),
        "corr_dlog_price_dlog_sales_all": float(np.corrcoef(dlp, dls)[0, 1]),
        "corr_dlog_price_dlog_sales_changed": float(np.corrcoef(dlp[changed], dls[changed])[0, 1]),
        "pooled_elasticity_slope_changed": slope,
        "n_consecutive_pairs": int(consec.sum()), "n_pairs_with_change": int(changed.sum()),
        "discount_weeks_share": float(disc.mean()),
        "median_sales_ratio_in_discount_weeks": float(size_rel[disc].median()),
        "median_sales_ratio_other_weeks": float(size_rel[~disc].median()),
        "binned_dlog_sales_by_dlog_price": {str(k): {"median_dlog_sales": float(r["median"]), "n": int(r["size"])} for k, r in binned.iterrows()},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].hist(np.log10(med_price), bins=50, color=BLUE, rwidth=0.9)
    axes[0].set_xticks([-1, 0, 1, 2, 3], ["0,1", "1", "10", "100", "1.000"])
    axes[0].set_xlabel("giá trung vị của chuỗi (thang log)"); axes[0].set_ylabel("số chuỗi")
    axes[0].set_title("Phân bố mức giá giữa các chuỗi")
    labels = ["giảm >26%", "giảm 10-26%", "giảm 3-10%", "±3%", "tăng 3-10%", "tăng 10-35%", "tăng >35%"]
    vals = (np.exp(binned["median"].to_numpy()) - 1) * 100
    axes[1].bar(range(len(vals)), vals, color=[AQUA if v > 0 else ORANGE for v in vals], width=0.7)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks(range(len(vals)), labels[:len(vals)], rotation=30, ha="right")
    axes[1].set_ylabel("% thay đổi lượng bán (trung vị)")
    axes[1].set_title("Giá đổi so với tuần trước thì lượng bán đổi thế nào")
    save(fig, "fig06_gia.png")

    # ---------- 6. Vòng đời ----------
    fs = summ.loc[ids, "first_sale_week"]
    per_q = fs.dt.to_period("Q").value_counts().sort_index()
    ramp_src = train_h[(train_h["weeks_since_first_sale"] <= 52) & ~train_h["unique_id"].isin(ids[cflat])]
    long_enough = ramp_src["unique_id"].map(pd.Series(active[:, :n_train].sum(axis=1), index=ids)) >= 53
    ramp = ramp_src[long_enough].groupby("weeks_since_first_sale")["sales"].agg(["mean", lambda s: (s > 0).mean()])
    ramp.columns = ["mean_units", "share_selling"]
    last13_sell = (Qv[:, n_train - 13:n_train] > 0).any(axis=1)
    cohort_year = fs.dt.year.to_numpy()
    survival = {int(y): float(last13_sell[(cohort_year == y) & (first_idx < n_train - 13)].mean()) for y in (2020, 2021, 2022)}
    trail_tr = np.zeros(len(ids), dtype=int)
    for t in range(n_train):
        trail_tr = np.where((Qt[:, t] == 0) & At[:, t], trail_tr + 1, 0)
    NUM["lifecycle"] = {
        "new_series_per_quarter": {str(k): int(v) for k, v in per_q.items()},
        "first_sale_in_first_week_of_data": int((first_idx == 0).sum()),
        "first_sale_in_test": int((first_idx >= n_train).sum()),
        "n_weeks_active_quantiles_all": {q: float(summ["n_weeks_active"].quantile(float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
        "dormant_at_end_of_all_data": int(summ["is_dormant"].sum()),
        "dormant_at_end_of_train_26w": int(((trail_tr >= 26) & (n_act_tr > 0)).sum()),
        "series_in_train": int((n_act_tr > 0).sum()),
        "share_still_selling_in_last13_train_weeks_by_cohort": survival,
        "ramp_mean_units": {int(k): float(v) for k, v in ramp["mean_units"].iloc[[0, 1, 2, 4, 8, 13, 26, 52]].items()},
        "ramp_share_selling": {int(k): float(v) for k, v in ramp["share_selling"].iloc[[0, 1, 2, 4, 8, 13, 26, 52]].items()},
        "ramp_n_series": int(long_enough.groupby(ramp_src["unique_id"]).first().sum()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(len(per_q))
    axes[0].bar(x, per_q.values, color=[GRAY if p.start_time >= split_date else BLUE for p in per_q.index], width=0.7)
    axes[0].set_xticks(x, [str(p) for p in per_q.index], rotation=45, ha="right")
    axes[0].set_title("Số chuỗi mới ra mắt mỗi quý (xám: trong tập kiểm tra)")
    axes[1].plot(ramp.index, ramp["share_selling"] * 100, color=BLUE)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("số tuần kể từ lần bán đầu tiên"); axes[1].set_ylabel("% chuỗi có bán trong tuần")
    axes[1].set_title("Sau khi ra mắt, bao nhiêu chuỗi còn bán mỗi tuần")
    save(fig, "fig07_vong_doi.png")

    # ---------- 7. Chuỗi tuần bằng 0 (TRAIN) ----------
    zr = train_h["zero_run"].clip(upper=26)
    haz = (train_h["sales"] > 0).groupby(zr).agg(["mean", "size"])
    runs = train_h[(train_h["sales"] > 0) & (train_h["zero_run"] > 0)]["zero_run"]
    NUM["zero_runs"] = {
        "period": "TRAIN",
        "p_sale_given_zero_run": {int(k): float(v) for k, v in haz["mean"].items() if k in (0, 1, 2, 3, 4, 8, 13, 26)},
        "rows_by_zero_run": {int(k): int(v) for k, v in haz["size"].items() if k in (0, 1, 4, 13, 26)},
        "completed_zero_run_length_quantiles": {q: float(runs.quantile(float(q))) for q in ("0.5", "0.75", "0.9", "0.99")},
        "share_zero_rows_in_runs_ge_13": float(((train_h["sales"] == 0) & (train_h["zero_run"] >= 12)).sum() / (train_h["sales"] == 0).sum()),
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(haz.index, haz["mean"] * 100, color=BLUE, width=0.7)
    ax.set_xticks([0, 2, 4, 8, 13, 20, 26], ["0", "2", "4", "8", "13", "20", "26+"])
    ax.set_xlabel("số tuần liên tiếp không bán ngay trước tuần này"); ax.set_ylabel("% có bán trong tuần này")
    ax.set_title("Càng ngừng bán lâu, xác suất bán lại càng thấp")
    save(fig, "fig08_xac_suat_ban_lai.png")

    # ---------- 8. Đột biến (TRAIN) ----------
    sp = train_h[train_h["is_spike"]]
    by_month_rows = train_h.groupby(train_h["week"].dt.month).size()
    sp_month = sp.groupby(sp["week"].dt.month).size().reindex(range(1, 13), fill_value=0)
    sp_week = sp.groupby("week").size().sort_values(ascending=False)
    NUM["spikes"] = {
        "period": "TRAIN",
        "n_spike_rows": int(len(sp)), "n_series_with_spike": int(sp["unique_id"].nunique()),
        "share_of_rows": float(len(sp) / len(train_h)),
        "share_of_revenue": float(sp["rev"].sum() / train_h["rev"].sum()),
        "share_of_units": float(sp["sales"].sum() / train_h["sales"].sum()),
        "spikes_per_10k_rows_by_month": {int(m): float(sp_month[m] / by_month_rows[m] * 1e4) for m in range(1, 13)},
        "top_weeks": {str(k.date()): int(v) for k, v in sp_week.head(5).items()},
        "share_spikes_in_q4": float(sp_month[[10, 11, 12]].sum() / max(1, len(sp))),
        "share_spikes_first_4_weeks_of_series": float((sp["weeks_since_first_sale"] < 4).mean()),
        "n_test_spike_rows": int(test["is_spike"].sum()),
        "test_spike_revenue_share": float(test.loc[test["is_spike"], "rev"].sum() / test["rev"].sum()),
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(range(1, 13), [NUM["spikes"]["spikes_per_10k_rows_by_month"][m] for m in range(1, 13)], color=BLUE, width=0.7)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("tháng"); ax.set_ylabel("tuần đột biến trên 10.000 chuỗi-tuần")
    ax.set_title("Tần suất đột biến theo tháng (huấn luyện)")
    save(fig, "fig09_dot_bien.png")

    # ---------- 9. Tập kiểm tra gồm những gì ----------
    test_rev = Rv[:, n_train:].sum(axis=1)
    cold = first_idx >= n_train
    flat_short = cflat & (~cold) & (n_act_tr < 26)             # nhà bán hàng bị cắt lịch sử: chưa đủ 26 tuần thật
    short = (~cold) & (n_act_tr < 26) & ~flat_short
    no_price = summ.loc[ids, "no_valid_price"].to_numpy()
    dorm_tr = (~cold) & (~short) & (~flat_short) & (trail_tr >= 26)
    clean = ~(cold | short | flat_short | dorm_tr | no_price)
    groups = {"cold_start (ra mắt trong test)": cold, "nhà bán hàng 0 (lịch sử cắt, <26 tuần thật)": flat_short,
              "ngắn (<26 tuần train)": short,
              "ngủ đông cuối train (26 tuần = 0)": dorm_tr, "không có giá": no_price & ~cold & ~short & ~flat_short & ~dorm_tr,
              "đủ điều kiện": clean}
    NUM["test_composition"] = {
        "period": "doanh thu TEST, phân nhóm theo thông tin biết được ở cuối TRAIN (các nhóm loại trừ nhau)",
        "total_test_revenue": float(test_rev.sum()),
        "groups": {k: {"n_series": int(m.sum()), "share_series": float(m.mean()),
                       "share_test_revenue": float(test_rev[m].sum() / test_rev.sum())} for k, m in groups.items()},
        "clean_7136_flag_share_test_revenue": float(test_rev[~(summ.loc[ids, ["is_dormant", "too_short_for_backtest", "no_valid_price"]].any(axis=1).to_numpy())].sum() / test_rev.sum()),
        "dormant_at_train_end_that_sell_again_in_test": float((test_rev[dorm_tr] > 0).mean()) if dorm_tr.any() else None,
        "cold_start_share_by_test_quarter": {f"Q{q + 1}": float(Rv[cold][:, n_train + 13 * q:n_train + 13 * (q + 1)].sum()
                                                                / Rv[:, n_train + 13 * q:n_train + 13 * (q + 1)].sum()) for q in range(4)},
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(groups)
    yy = np.arange(len(names))
    ax.barh(yy + 0.2, [NUM["test_composition"]["groups"][k]["share_series"] * 100 for k in names], height=0.36, color=BLUE, label="% số chuỗi")
    ax.barh(yy - 0.2, [NUM["test_composition"]["groups"][k]["share_test_revenue"] * 100 for k in names], height=0.36, color=ORANGE, label="% doanh thu kiểm tra")
    ax.set_yticks(yy, names); ax.invert_yaxis(); ax.legend(loc="center right")
    ax.set_title("Doanh thu 52 tuần kiểm tra đến từ nhóm chuỗi nào")
    save(fig, "fig10_thanh_phan_tap_kiem_tra.png")

    # ---------- 10. Baseline (4 gốc dự báo, tầm 13 tuần) ----------
    origins = [n_train + HORIZON * k for k in range(4)]

    def forecasts(M, o):
        return {"naive": np.repeat(M[..., o - 1:o], HORIZON, axis=-1),
                "seasonal_naive_52": M[..., o - 52:o - 52 + HORIZON],
                "mean_last_13": np.repeat(M[..., o - 13:o].mean(axis=-1, keepdims=True), HORIZON, axis=-1),
                "mean_last_52": np.repeat(M[..., o - 52:o].mean(axis=-1, keepdims=True), HORIZON, axis=-1)}

    Rh = np.where(flat_cell, np.nan, Rv)          # cấp chuỗi: giai đoạn phẳng không thuộc lịch sử

    def forecasts_series(o):
        m13 = np.nanmean(Rh[:, o - 13:o], axis=1)
        sn = Rh[:, o - 52:o - 52 + HORIZON]
        rep_ = lambda v: np.repeat(v[:, None], HORIZON, axis=1)
        return {"naive": rep_(Rv[:, o - 1]), "seasonal_naive_52": np.where(np.isnan(sn), m13[:, None], sn),
                "mean_last_13": rep_(m13), "mean_last_52": rep_(np.nanmean(Rh[:, o - 52:o], axis=1))}

    LEVELS = ("total", "series_all", "series_existing", "series_flat_clients", "series_other_clients")
    res = {lvl: {m: {"abs": 0.0, "act": 0.0, "bias": 0.0} for m in ("naive", "seasonal_naive_52", "mean_last_13", "mean_last_52")}
           for lvl in LEVELS}
    by_origin = {}
    for o in origins:
        act_t, act_s = tot_rev[o:o + HORIZON], Rv[:, o:o + HORIZON]
        exists = first_idx < o
        ft, fser = forecasts(tot_rev, o), forecasts_series(o)
        by_origin[str(weeks[o].date())] = {m: float(np.abs(ft[m] - act_t).sum() / act_t.sum()) for m in ft}
        for m in ft:
            for lvl, f, a in (("total", ft[m], act_t), ("series_all", fser[m], act_s), ("series_existing", fser[m][exists], act_s[exists]),
                              ("series_flat_clients", fser[m][cflat], act_s[cflat]), ("series_other_clients", fser[m][~cflat], act_s[~cflat])):
                res[lvl][m]["abs"] += float(np.abs(f - a).sum()); res[lvl][m]["act"] += float(a.sum()); res[lvl][m]["bias"] += float((f - a).sum())
    NUM["baselines"] = {
        "period": "TEST; 4 gốc dự báo cách nhau 13 tuần, mỗi gốc dự báo 13 tuần; MAE% = tổng |sai số| / tổng thực tế",
        "origins": [str(weeks[o].date()) for o in origins],
        "wmape": {lvl: {m: v["abs"] / v["act"] for m, v in d.items()} for lvl, d in res.items()},
        "bias": {lvl: {m: v["bias"] / v["act"] for m, v in d.items()} for lvl, d in res.items()},
        "total_wmape_by_origin": by_origin,
    }
    fig, ax = plt.subplots(figsize=(8, 3.8))
    methods = ["naive", "seasonal_naive_52", "mean_last_13", "mean_last_52"]
    xx = np.arange(len(methods))
    for k, (lvl, label, color) in enumerate((("total", "Tổng doanh thu", BLUE), ("series_all", "Từng chuỗi (mọi chuỗi)", ORANGE))):
        vals = [NUM["baselines"]["wmape"][lvl][m] * 100 for m in methods]
        bars = ax.bar(xx + (k - 0.5) * 0.38, vals, width=0.36, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(xx, ["naive\n(tuần cuối)", "seasonal naive\n(cùng tuần năm trước)", "trung bình\n13 tuần cuối", "trung bình\n52 tuần cuối"])
    ax.set_ylabel("MAE%"); ax.legend(); ax.set_title("Sai số của các baseline đơn giản trên tập kiểm tra")
    save(fig, "fig11_baseline.png")

    # ---------- Phase 2 (chỉ số lượng) ----------
    p2 = pd.read_parquet(os.path.join(CLEAN, "vn1_phase2_sales.parquet"))
    p2s = p2.groupby("unique_id")["sales"].sum().reindex(ids)
    NUM["phase2"] = {
        "share_series_zero_all_13_weeks": float((p2s == 0).mean()),
        "units_phase2_vs_last13_of_phase1": float(p2s.sum() / Qv[:, -13:].sum()),
        "units_phase2_vs_same_weeks_2023": float(p2s.sum() / Qv[:, n_weeks - 52:n_weeks - 39].sum()),
    }

    with open(os.path.join(OUT, "eda_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(NUM, f, ensure_ascii=False, indent=1)
    print(json.dumps(NUM, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
