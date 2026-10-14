"""Khảo sát cách ghi số liệu của nhà bán hàng 0 (đoạn lặp và bội số của 7).

Chạy:  python scripts/05_nha_ban_hang_0.py

Đọc raw/ (không sửa), ghi vào reports/dq/:
  dq_constant_run_series.csv        mỗi dòng là một chuỗi có đoạn lặp y hệt >= 13 tuần liên tiếp
  fig_dq04_373_chuoi_ban_do.png     bản đồ các đoạn lặp của nhà bán hàng 0
  fig_dq05_373_chuoi_vi_du.png      24 chuỗi ví dụ
  fig_dq06_khach_hang_0.png         doanh thu tuần của nhà bán hàng 0
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "reports", "dq")
KEYS = ["Client", "Warehouse", "Product"]
MIN_RUN, N_TRAIN, SELLER = 13, 131, 0
BLUE, ORANGE, GRAY, GRID, SURF, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#e4e3df", "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.dpi": 130})


def main():
    os.makedirs(OUT, exist_ok=True)
    load = lambda kind: pd.concat([pd.read_csv(os.path.join(RAW, f"Phase {i} - {kind}.csv")).set_index(KEYS).sort_index() for i in (0, 1)], axis=1)
    S = load("Sales")
    P = load("Price").reindex(S.index)
    weeks = pd.to_datetime(S.columns)
    Sv, Pv = S.to_numpy(float), P.to_numpy(float)
    R = np.where(Pv > 0, Sv * Pv, 0.0)
    client = S.index.get_level_values(0).to_numpy()

    in_run = np.zeros_like(Sv, dtype=bool)
    rows = []
    for i in range(len(Sv)):
        v = Sv[i]
        ch = np.flatnonzero(np.diff(v) != 0) + 1
        runs = [(a, b) for a, b in zip(np.r_[0, ch], np.r_[ch, len(v)]) if v[a] > 0 and b - a >= MIN_RUN]
        if not runs:
            continue
        for a, b in runs:
            in_run[i, a:b] = True
        a, b = max(runs, key=lambda r: r[1] - r[0])
        n, ns = sum(b - a for a, b in runs), int((v > 0).sum())
        pr = Pv[i][(v > 0) & (Pv[i] > 0)]
        rows.append({"unique_id": "_".join(map(str, S.index[i])), "client": S.index[i][0], "warehouse": S.index[i][1], "product": S.index[i][2],
                     "n_runs": len(runs), "weeks_in_runs": n, "selling_weeks": ns, "share_of_selling_weeks": round(n / ns, 3),
                     "longest_run_weeks": b - a, "longest_run_value": int(v[a]), "longest_run_start": weeks[a].date(),
                     "longest_run_end": weeks[b - 1].date(), "run_values": " ".join(str(int(x)) for x in sorted({v[a] for a, b in runs})),
                     "total_units": int(v.sum()), "median_price": round(float(np.median(pr)), 2) if len(pr) else np.nan,
                     "price_cv": round(float(pr.std() / pr.mean()), 2) if len(pr) else np.nan, "revenue": round(float(R[i].sum()))})
    pd.DataFrame(rows).sort_values("weeks_in_runs", ascending=False).to_csv(os.path.join(OUT, "dq_constant_run_series.csv"), index=False)

    c0 = client == SELLER
    f0 = in_run.any(axis=1) & c0
    idx = np.flatnonzero(f0)
    first_run = np.array([np.argmax(in_run[i]) for i in idx])
    order = idx[np.lexsort((-(in_run[idx].sum(axis=1)), first_run))]
    state = np.where(in_run[order], 2, np.where(Sv[order] > 0, 1, 0))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(state, aspect="auto", cmap=ListedColormap([SURF, "#b9d3f2", ORANGE]), interpolation="nearest", extent=[0, len(weeks), len(order), 0])
    ticks = [i for i, w in enumerate(weeks) if w.month in (1, 7) and w.day <= 7]
    ax.set_xticks(ticks, [weeks[i].strftime("%m/%Y") for i in ticks])
    ax.axvline(N_TRAIN, color=INK, lw=1.2)
    ax.text(N_TRAIN + 1, 8, "52 tuần kiểm tra", fontsize=9, color=INK)
    ax.set_ylabel(f"{len(order)} chuỗi của nhà bán hàng {SELLER} (mỗi hàng một chuỗi)")
    ax.set_title(f"Cam: tuần nằm trong đoạn lặp từ {MIN_RUN} tuần. Xanh nhạt: tuần có bán bình thường. Trắng: không bán.")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dq04_373_chuoi_ban_do.png")); plt.close(fig)

    pick = np.random.default_rng(1).choice(idx, 24, replace=False)
    fig, axes = plt.subplots(6, 4, figsize=(13, 11), sharex=True)
    for ax, i in zip(axes.ravel(), pick):
        ax.plot(weeks, Sv[i], color=BLUE, lw=1)
        ax.plot(weeks, np.where(in_run[i], Sv[i], np.nan), color=ORANGE, lw=2.2)
        ax.axvspan(weeks[N_TRAIN], weeks[-1], color=GRAY, alpha=0.13, lw=0)
        ax.set_ylim(0); ax.grid(True, color=GRID, lw=0.5)
        ax.set_title("_".join(map(str, S.index[i])), fontsize=9, fontweight="normal"); ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.suptitle(f"24 chuỗi ngẫu nhiên trong {len(idx)} chuỗi: số lượng bán mỗi tuần. Cam = đoạn lặp, vùng xám = 52 tuần kiểm tra.", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dq05_373_chuoi_vi_du.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(weeks, R[c0].sum(axis=0) / 1e6, color=BLUE, lw=2, label=f"Nhà bán hàng {SELLER}, mọi chuỗi ({int(c0.sum())})")
    ax.plot(weeks, R[f0].sum(axis=0) / 1e6, color=ORANGE, lw=2, label=f"{len(idx)} chuỗi có đoạn lặp")
    ax.axvspan(weeks[N_TRAIN], weeks[-1], color=GRAY, alpha=0.13, lw=0)
    ax.set_ylim(0); ax.grid(True, color=GRID, lw=0.5); ax.legend(frameon=False)
    ax.set_title(f"Doanh thu tuần của nhà bán hàng {SELLER} (triệu). Vùng xám = 52 tuần kiểm tra.")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dq06_khach_hang_0.png")); plt.close(fig)
    print(f"{len(rows)} series with constant runs; {len(idx)} of seller {SELLER}")


if __name__ == "__main__":
    main()
