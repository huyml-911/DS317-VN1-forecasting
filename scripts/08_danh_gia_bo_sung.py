"""Đánh giá bổ sung cho bản cải tiến (v2).

Chạy:  python scripts/08_danh_gia_bo_sung.py              (khoảng 15-30 phút)
       python scripts/08_danh_gia_bo_sung.py --levels-only  (chỉ phần A, vài giây)

Cần chạy trước: scripts/06_forecast.py và scripts/07_forecast_v2.py.
Ghi vào reports/danh_gia_bo_sung/: results.json, forecast_phase2_units.parquet, fig01, fig02.

Phần A. WMAPE và độ lệch ở bốn cấp tổng hợp (chuỗi, kho, nhà bán hàng, tổng) trên 4 mốc kiểm tra doanh thu,
        cho naive, bản đầu (v1) và bản cải tiến (v2). Dự báo của từng chuỗi được cộng lên cấp tương ứng rồi mới so với thực tế.
          diem_vn1     = MAE% + |Bias%| của từng kỳ 13 tuần ở cấp đó, trung bình bốn kỳ   (thước đo chính)
          wmape_tuan   = tổng |sai số của (đơn vị, tuần)| / tổng thực tế, gộp bốn kỳ
          lech_13_tuan = tổng |sai số cộng dồn 13 tuần của (đơn vị, mốc)| / tổng thực tế   (đúng tổng cả kỳ hay không)
          bias         = (tổng dự báo - tổng thực tế) / tổng thực tế                     (giống nhau ở mọi cấp)

Phần B. Chấm như cuộc thi VN1: dùng toàn bộ Phase 0 + 1, dự báo SỐ LƯỢNG của 13 tuần Phase 2 cho cả 15.053 chuỗi,
        điểm = MAE% + |Bias%| = tổng |sai số| / tổng thực tế + |tổng sai số| / tổng thực tế.
        Cấu hình mô hình lấy nguyên từ bản v2 (không tinh chỉnh lại); tỷ lệ trộn theo nhóm được chọn trên 4 mốc
        131, 144, 157, 170 (đều nằm trong Phase 0 + 1), rồi mô hình được huấn luyện lại đến hết Phase 1 và dự báo Phase 2 một lần.
"""
import importlib.util
import json
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "danh_gia_bo_sung")
REF_RANK1, REF_NAIVE = 0.4637, 0.5070      # điểm của hạng 1 và của naive trong cuộc thi, do người dùng cung cấp
SEL_ORIGINS, FINAL_ORIGIN = [131, 144, 157, 170], 183


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- Phần A
def level_metrics():
    v1 = pd.read_parquet(os.path.join(ROOT, "reports", "forecast", "forecasts_test.parquet"),
                         columns=["unique_id", "origin", "week", "naive", "gbm_revenue", "chon_tu_do", "actual"])
    v2 = pd.read_parquet(os.path.join(ROOT, "reports", "forecast_v2", "forecasts_test_v2.parquet"), columns=["unique_id", "origin", "week", "forecast"])
    df = v1.merge(v2.rename(columns={"forecast": "v2"}), on=["unique_id", "origin", "week"], validate="one_to_one")
    parts = df["unique_id"].str.split("_", expand=True)
    df["nha_ban_hang"] = parts[0]
    df["kho"] = parts[0] + "_" + parts[1]
    methods = {"naive": "naive", "v1_gbm_doanh_thu": "gbm_revenue", "v1_moi_nhom_tu_chon": "chon_tu_do", "v2_tron_theo_nhom": "v2"}
    levels = {"chuoi": ["unique_id"], "kho": ["kho"], "nha_ban_hang": ["nha_ban_hang"], "tong": []}
    total = df["actual"].sum()
    act_o = df.groupby("origin")["actual"].sum()
    res = {}
    for lname, keys in levels.items():
        g = df.groupby(keys + ["origin", "week"], sort=False)[list(methods.values()) + ["actual"]].sum()
        blk = g.groupby(keys + ["origin"], sort=False).sum() if keys else g.groupby("origin").sum()
        res[lname] = {"n_don_vi": int(df[keys].drop_duplicates().shape[0]) if keys else 1}
        for m, col in methods.items():
            mae_o = (g[col] - g["actual"]).abs().groupby(level="origin").sum() / act_o
            bias_o = (df.groupby("origin")[col].sum() - act_o) / act_o
            res[lname][m] = {"diem_vn1": float((mae_o + bias_o.abs()).mean()),
                             "wmape_tuan": float((g[col] - g["actual"]).abs().sum() / total),
                             "lech_13_tuan": float((blk[col] - blk["actual"]).abs().sum() / total),
                             "bias": float((df[col].sum() - total) / total)}
    # từng nhà bán hàng: 10 nhà lớn nhất theo doanh thu kiểm tra
    s = df.groupby(["nha_ban_hang", "origin", "week"])[["naive", "chon_tu_do", "v2", "actual"]].sum()
    per = []
    for seller, d in s.groupby(level=0):
        a = d["actual"].sum()
        ao = d["actual"].groupby(level="origin").sum()
        vn1 = {k: float((((d[c] - d["actual"]).abs().groupby(level="origin").sum() + (d[c] - d["actual"]).groupby(level="origin").sum().abs()) / ao).mean())
               if (ao > 0).all() else None for k, c in (("naive", "naive"), ("v1", "chon_tu_do"), ("v2", "v2"))}
        per.append({"nha_ban_hang": seller, "share_revenue": float(a / total), **{f"diem_vn1_{k}": v for k, v in vn1.items()},
                    **{f"wmape_{k}": float((d[c] - d["actual"]).abs().sum() / a) if a else None for k, c in (("naive", "naive"), ("v1", "chon_tu_do"), ("v2", "v2"))},
                    **{f"bias_{k}": float((d[c].sum() - a) / a) if a else None for k, c in (("naive", "naive"), ("v1", "chon_tu_do"), ("v2", "v2"))}})
    per = sorted(per, key=lambda r: -r["share_revenue"])
    v2_better = sum(1 for r in per if r["diem_vn1_v2"] is not None and r["diem_vn1_v2"] < r["diem_vn1_naive"])
    return {"levels": res, "top_sellers": per[:10], "sellers_where_v2_beats_naive": v2_better, "n_sellers": len(per)}


# ---------------------------------------------------------------- Phần B
def competition_score(fc, actual):
    err = fc - actual
    mae, bias = np.abs(err).sum() / actual.sum(), err.sum() / actual.sum()
    return {"mae": float(mae), "bias": float(bias), "score": float(mae + abs(bias))}


def phase2_units():
    v2 = load_module("forecast_v2", "07_forecast_v2.py")
    base = v2.base
    H = base.H
    chosen = json.load(open(os.path.join(ROOT, "reports", "forecast_v2", "results.json"), encoding="utf-8"))["chosen"]
    D, ids, weeks, seller0 = base.load_data()
    Du = dict(D)
    Du["Rh"] = D["Qh"]            # mọi đặc trưng và mục tiêu tính trên SỐ LƯỢNG thay cho doanh thu
    Du["Praw"] = np.ones_like(D["Praw"])   # với mục tiêu số lượng, thiếu giá không phải lý do để dự báo 0: bỏ nhãn no_price
    n = len(ids)
    v2.CACHE = os.path.join(OUT, "_cache_units")
    os.makedirs(v2.CACHE, exist_ok=True)
    t0 = time.time()
    for t in sorted(set(range(v2.TRAIN_MIN, FINAL_ORIGIN - H + 1, v2.TRAIN_STEP)) | set(SEL_ORIGINS)):
        v2.cache_origin(t, Du, True)
    v2.cache_origin(FINAL_ORIGIN, Du, False)
    cols = v2.ALL if chosen["feature_set"] == "mới" else v2.OLD
    cfg_mean = chosen["mean_model_cfg"]
    cfg_med = cfg_mean | {"loss": "absolute_error"}
    print(f"đặc trưng số lượng xong sau {time.time() - t0:.0f}s; cấu hình: {cfg_mean}", flush=True)

    def store_at(o):
        st = base.series_state(o, Du)
        rep = lambda v: np.repeat(np.asarray(v, dtype=float)[:, None], H, axis=1)
        fc = {"naive": rep(Du["Rh"][:, o - 1]), "mean13": rep(base.wmean(Du["Rh"], o, 13, st["age"])), "zero": np.zeros((n, H)),
              "tsb_a0.2": rep(base.smoothing_family(o, Du, 0.2)["tsb"])}
        fc = {k: np.where(st["exists"][:, None], v, 0.0) for k, v in fc.items()}
        t1 = time.time()
        fc["gbm_trung_binh"] = v2.fit_predict(o, cols, cfg_mean, n)
        fc["gbm_trung_vi"] = v2.fit_predict(o, cols, cfg_med, n)
        print(f"mốc {o}: hai mô hình xong sau {time.time() - t1:.0f}s", flush=True)
        actual = Du["Rh"][:, o:o + H] if o + H <= Du["Rh"].shape[1] else None
        return {"fc": fc, "group": st["group"], "actual": actual, "valid": np.ones((n, H), dtype=bool)}

    comps = ["gbm_trung_binh", "gbm_trung_vi", "naive", "mean13", "tsb_a0.2"]
    sel = {o: store_at(o) for o in SEL_ORIGINS}
    weights = v2.blend_weights(sel, SEL_ORIGINS, comps)
    sel_scores = {}
    for name in comps + ["tron_theo_nhom"]:
        fc = np.concatenate([v2.blended(sel, o, weights) if name == "tron_theo_nhom" else sel[o]["fc"][name] for o in SEL_ORIGINS])
        sel_scores[name] = competition_score(fc, np.concatenate([sel[o]["actual"] for o in SEL_ORIGINS]))
    del sel

    fin = store_at(FINAL_ORIGIN)
    p2 = pd.read_parquet(os.path.join(ROOT, "clean", "vn1_phase2_sales.parquet"))
    weeks2 = sorted(p2["week"].unique())
    actual = p2.pivot(index="unique_id", columns="week", values="sales").reindex(index=ids, columns=weeks2).to_numpy(dtype=float)
    assert not np.isnan(actual).any() and actual.shape == (n, H)
    assert (pd.Timestamp(weeks2[0]) - weeks[-1]).days == 7
    fin["actual"] = actual
    fin["fc"]["tron_theo_nhom"] = v2.blended({FINAL_ORIGIN: fin}, FINAL_ORIGIN, weights)
    scores = {k: competition_score(v, actual) for k, v in fin["fc"].items() if k != "zero"}
    g = fin["group"]
    by_group = {base.GROUPS[c]: {"n_series": int((g == c).sum()), "share_units": float(actual[g == c].sum() / actual.sum()),
                                 **{k: competition_score(fin["fc"][k][g == c], actual[g == c]) for k in ("naive", "gbm_trung_binh", "tron_theo_nhom")}}
                for c in range(8) if (g == c).any() and actual[g == c].sum() > 0}
    by_seller = {name: {k: competition_score(fin["fc"][k][m], actual[m]) for k in ("naive", "gbm_trung_binh", "tron_theo_nhom")} | {"share_units": float(actual[m].sum() / actual.sum())}
                 for name, m in (("nha_ban_hang_0", seller0), ("con_lai", ~seller0))}
    pd.DataFrame({"unique_id": np.repeat(ids, H), "week": np.tile(np.array(weeks2), n), "h": np.tile(np.arange(1, H + 1), n),
                  "group": np.repeat(np.array(base.GROUPS)[g], H), "forecast_units": fin["fc"]["tron_theo_nhom"].ravel(),
                  "naive_units": fin["fc"]["naive"].ravel(), "actual_units": actual.ravel()}).to_parquet(os.path.join(OUT, "forecast_phase2_units.parquet"), index=False)
    shutil.rmtree(v2.CACHE, ignore_errors=True)
    return {"setup": {"train_through": str(weeks[-1].date()), "forecast_weeks": [str(pd.Timestamp(w).date()) for w in (weeks2[0], weeks2[-1])],
                      "n_series": n, "target": "số lượng (units)", "no_price_fix": True, "metric": "MAE% + |Bias%|", "selection_origins": [str(weeks[o].date()) for o in SEL_ORIGINS],
                      "model_cfg": cfg_mean, "feature_set": chosen["feature_set"]},
            "reference": {"rank1": REF_RANK1, "naive_benchmark": REF_NAIVE, "source": "do người dùng cung cấp, chưa đối chiếu với trang cuộc thi"},
            "blend_weights": {base.GROUPS[c]: {k: v for k, v in w.items() if v > 0} for c, w in weights.items()},
            "selection_scores": sel_scores, "phase2_scores": scores, "phase2_by_group": by_group, "phase2_by_seller": by_seller}


def make_figures(out):
    BLUE, ORANGE, AQUA, GRAY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10, "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 130, "legend.frameon": False})
    lv = out["levels"]["levels"]
    names = {"naive": ("Naive", GRAY), "v1_moi_nhom_tu_chon": ("v1: mỗi nhóm tự chọn", BLUE), "v2_tron_theo_nhom": ("v2: trộn theo nhóm", AQUA)}
    cats = [("chuoi", "Chuỗi"), ("kho", "Kho"), ("nha_ban_hang", "Nhà bán hàng"), ("tong", "Tổng")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "diem_vn1", "Điểm VN1 ở từng cấp (trung bình 4 kỳ)"), (axes[1], "lech_13_tuan", "Sai lệch cộng dồn 13 tuần ở từng cấp (%)")):
        x = np.arange(len(cats))
        for j, (m, (lab, color)) in enumerate(names.items()):
            v = [lv[c][m][key] * 100 for c, _ in cats]
            bars = ax.bar(x + (j - 1) * 0.27, v, width=0.25, color=color, label=lab)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [f"{lab}\n({lv[c]['n_don_vi']:,})".replace(",", ".") for c, lab in cats]); ax.set_title(title)
        ax.set_ylim(0, max(lv["chuoi"][m][key] for m in names) * 100 * 1.18)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_cac_cap_tong_hop.png")); plt.close(fig)

    if "competition" in out:
        sc = out["competition"]["phase2_scores"]
        order = sorted(sc, key=lambda k: sc[k]["score"])
        lab = {"tron_theo_nhom": "v2: trộn theo nhóm", "gbm_trung_binh": "v2: một mô hình (trung bình)", "gbm_trung_vi": "v2: một mô hình (trung vị)",
               "naive": "Naive (lặp lại tuần cuối)", "mean13": "Trung bình 13 tuần", "tsb_a0.2": "TSB (alpha 0,2)"}
        fig, ax = plt.subplots(figsize=(9, 3.8))
        mae = [sc[k]["mae"] * 100 for k in order]
        bi = [abs(sc[k]["bias"]) * 100 for k in order]
        ax.barh(range(len(order)), mae, color=BLUE, height=0.7, label="MAE%")
        ax.barh(range(len(order)), bi, left=mae, color=ORANGE, height=0.7, label="|Bias%|")
        for i, (a, b) in enumerate(zip(mae, bi)):
            ax.text(a + b + 0.5, i, f"{a + b:.2f}", va="center", fontsize=9)
        for ref, text, ha, dx in ((out["competition"]["reference"]["rank1"], "hạng 1", "right", -0.6),
                                  (out["competition"]["reference"]["naive_benchmark"], "naive của cuộc thi", "left", 0.6)):
            ax.axvline(ref * 100, color=INK, lw=1, ls="--")
            ax.text(ref * 100 + dx, -0.78, f"{text}: {ref * 100:.2f}", ha=ha, fontsize=8, color=INK)
        ax.set_yticks(range(len(order)), [lab[k] for k in order]); ax.invert_yaxis(); ax.set_ylim(len(order) - 0.5, -1.1)
        ax.set_xlim(0, max(a + b for a, b in zip(mae, bi)) * 1.3); ax.legend(loc="center right")
        ax.set_title("Điểm kiểu cuộc thi VN1 trên Phase 2 (số lượng, thấp hơn là tốt hơn)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig02_diem_cuoc_thi_phase2.png")); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.json")
    out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    out["levels"] = level_metrics()
    if "--levels-only" not in sys.argv:
        prev = out.get("competition")
        if prev and not prev["setup"].get("no_price_fix"):       # giữ lại kết quả của lượt chạy đầu (trước khi bỏ nhãn no_price) để đối chiếu
            out["competition_first_run"] = {"phase2_scores": prev["phase2_scores"], "note": "lượt đầu: chuỗi no_price bị dự báo 0 dù mục tiêu là số lượng"}
        out["competition"] = phase2_units()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    make_figures(out)
    print(json.dumps({"levels": out["levels"]["levels"], "competition": {k: out.get("competition", {}).get(k) for k in ("phase2_scores", "selection_scores", "blend_weights")}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
