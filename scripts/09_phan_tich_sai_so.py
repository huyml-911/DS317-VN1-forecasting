"""Phân tích sai số của bản cải tiến (v2) trên 4 kỳ kiểm tra doanh thu.

Chạy:  python scripts/09_phan_tich_sai_so.py     (vài giây; cần 06, 07 đã chạy)
Ghi:   reports/phan_tich_sai_so/error_analysis.json
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "phan_tich_sai_so")


def vn1(d, col="fc"):
    """MAE%, Bias% gộp trên các dòng của d."""
    a = d["actual"].sum()
    return {"mae": float((d[col] - d["actual"]).abs().sum() / a), "bias": float((d[col].sum() - a) / a), "share_actual": None}


def main():
    os.makedirs(OUT, exist_ok=True)
    v2 = pd.read_parquet(os.path.join(ROOT, "reports", "forecast_v2", "forecasts_test_v2.parquet"))
    v1 = pd.read_parquet(os.path.join(ROOT, "reports", "forecast", "forecasts_test.parquet"), columns=["unique_id", "origin", "week", "naive"])
    df = v2.rename(columns={"forecast": "fc"}).merge(v1, on=["unique_id", "origin", "week"], validate="one_to_one")
    df["err"] = df["fc"] - df["actual"]
    df["ae"] = df["err"].abs()
    tot_a, tot_ae = df["actual"].sum(), df["ae"].sum()
    res = {"overall": {"mae": float(tot_ae / tot_a), "bias": float(df["err"].sum() / tot_a), "n_cells": int(len(df))}}

    # 1. theo tầm dự báo
    res["by_horizon"] = [{"h": int(h), "mae": float(d["ae"].sum() / d["actual"].sum()), "bias": float(d["err"].sum() / d["actual"].sum()),
                          "mae_naive": float((d["naive"] - d["actual"]).abs().sum() / d["actual"].sum()),
                          "bias_naive": float((d["naive"].sum() - d["actual"].sum()) / d["actual"].sum())} for h, d in df.groupby("h")]

    # 2. dự báo cao hay thấp; ô thực tế bằng 0
    over, zero = df["err"] > 0, df["actual"] == 0
    res["direction"] = {"share_ae_over": float(df.loc[over, "ae"].sum() / tot_ae), "share_ae_under": float(df.loc[~over, "ae"].sum() / tot_ae),
                        "share_cells_actual_zero": float(zero.mean()), "share_ae_on_zero_cells": float(df.loc[zero, "ae"].sum() / tot_ae),
                        "forecast_on_zero_cells_share_of_forecast": float(df.loc[zero, "fc"].sum() / df["fc"].sum()),
                        "share_cells_forecast_zero": float((df["fc"] == 0).mean()),
                        "actual_on_zero_forecast_cells_share_of_actual": float(df.loc[df["fc"] == 0, "actual"].sum() / tot_a)}

    # 3. mức nền hay thời điểm: sai số của tổng 13 tuần so với sai số từng tuần
    blk = df.groupby(["unique_id", "origin"]).agg(fc=("fc", "sum"), actual=("actual", "sum"), ae=("ae", "sum"), group=("group", "first"))
    blk["err"] = blk["fc"] - blk["actual"]
    level = blk["err"].abs().sum()
    res["level_vs_timing"] = {"weekly_mae": float(tot_ae / tot_a), "cumulative_13w": float(level / tot_a), "share_level": float(level / tot_ae)}
    by_g = {}
    for g, d in blk.groupby("group"):
        if d["actual"].sum() > 0:
            by_g[g] = {"share_level": float(d["err"].abs().sum() / d["ae"].sum()), "weekly_mae": float(d["ae"].sum() / d["actual"].sum())}
    res["level_vs_timing"]["by_group"] = by_g

    # 4. độ chính xác tổng 13 tuần của từng chuỗi (chuỗi có doanh thu thực tế)
    pos = blk[blk["actual"] > 0]
    r = pos["fc"] / pos["actual"]
    w = pos["actual"] / pos["actual"].sum()
    res["series_13w_accuracy"] = {"n": int(len(pos)), **{f"within_{int(t * 100)}pct_share_series": float(((r - 1).abs() <= t).mean()) for t in (0.2, 0.5)},
                                  **{f"within_{int(t * 100)}pct_share_revenue": float(w[((r - 1).abs() <= t)].sum()) for t in (0.2, 0.5)},
                                  "over_2x_share_series": float((r > 2).mean()), "under_half_share_series": float((r < 0.5).mean()),
                                  "over_2x_share_revenue": float(w[r > 2].sum()), "under_half_share_revenue": float(w[r < 0.5].sum())}

    # 5. theo quy mô chuỗi (xếp theo doanh thu thực tế trong kỳ) và mức tập trung của sai số
    s = blk.sort_values("actual", ascending=False)
    n = len(s)
    buckets = [("1% lớn nhất", 0, 0.01), ("1-5%", 0.01, 0.05), ("5-20%", 0.05, 0.2), ("20-50%", 0.2, 0.5), ("50% nhỏ nhất", 0.5, 1.0)]
    res["by_size"] = []
    for name, lo, hi in buckets:
        d = s.iloc[int(n * lo):int(n * hi)]
        res["by_size"].append({"bucket": name, "share_actual": float(d["actual"].sum() / tot_a), "share_ae": float(d["ae"].sum() / tot_ae),
                               "mae": float(d["ae"].sum() / d["actual"].sum()) if d["actual"].sum() else None,
                               "bias": float(d["err"].sum() / d["actual"].sum()) if d["actual"].sum() else None})
    e = blk.sort_values("ae", ascending=False)
    res["error_concentration"] = {f"top_{p}": float(e["ae"].iloc[:max(1, int(n * q))].sum() / tot_ae) for p, q in (("1pct", 0.01), ("5pct", 0.05), ("20pct", 0.2))}

    # 6. tuần cao điểm cuối năm so với phần còn lại
    wk = df["week"].dt.isocalendar().week
    peak = wk.isin([47, 48, 49, 50, 51])
    res["peak_weeks"] = {"peak": {**vn1(df[peak]), "mae_naive": float((df.loc[peak, "naive"] - df.loc[peak, "actual"]).abs().sum() / df.loc[peak, "actual"].sum()),
                                  "bias_naive": float((df.loc[peak, "naive"].sum() - df.loc[peak, "actual"].sum()) / df.loc[peak, "actual"].sum()),
                                  "share_actual": float(df.loc[peak, "actual"].sum() / tot_a), "share_ae": float(df.loc[peak, "ae"].sum() / tot_ae)},
                         "other": {**vn1(df[~peak]), "mae_naive": float((df.loc[~peak, "naive"] - df.loc[~peak, "actual"]).abs().sum() / df.loc[~peak, "actual"].sum()),
                                   "share_actual": float(df.loc[~peak, "actual"].sum() / tot_a)}}

    # 7. theo nhóm: tỷ trọng doanh thu và tỷ trọng sai số
    res["by_group"] = {g: {"share_actual": float(d["actual"].sum() / tot_a), "share_ae": float(d["ae"].sum() / tot_ae),
                           "mae": float(d["ae"].sum() / d["actual"].sum()) if d["actual"].sum() else None,
                           "bias": float(d["err"].sum() / d["actual"].sum()) if d["actual"].sum() else None,
                           "bias_contribution": float(d["err"].sum() / tot_a)} for g, d in df.groupby("group")}

    # 8. mười khối (chuỗi, kỳ) sai nhiều nhất
    top = e.head(10).reset_index()
    res["top_errors"] = [{"unique_id": r_["unique_id"], "origin": str(r_["origin"].date()), "group": r_["group"], "actual_13w": float(r_["actual"]),
                          "forecast_13w": float(r_["fc"]), "share_of_total_ae": float(r_["ae"] / tot_ae)} for _, r_ in top.iterrows()]
    with open(os.path.join(OUT, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
