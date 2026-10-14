"""Làm sạch dữ liệu VN1: raw/ (bảng rộng) -> clean/ (bảng dài).

Chạy:  python scripts/01_clean.py

Chỉ đọc từ raw/, chỉ ghi vào clean/:
  vn1_long.parquet           Phase 0 + Phase 1 (183 tuần), mỗi dòng là một chuỗi-tuần kể từ tuần bán đầu tiên
  vn1_phase2_sales.parquet   Phase 2 (chỉ có số lượng, không có giá) để kiểm tra thêm theo đơn vị
  series_summary.csv         Một dòng cho mỗi chuỗi client x warehouse x product
  cleaning_report.md         Nhật ký làm sạch

Nếu máy chưa có pyarrow/fastparquet thì hai file parquet được ghi thành .csv cùng tên.

Lưu ý rò rỉ: price_raw chỉ có giá trị ở tuần có bán hàng, và revenue là biến mục tiêu;
không dùng price_raw (hay việc nó trống) làm đặc trưng dự báo. zero_run chỉ đếm các tuần TRƯỚC tuần hiện tại.
"""
import os
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "clean")
KEYS = ["Client", "Warehouse", "Product"]
TEST_WEEKS = 52      # 52 tuần cuối của Phase 0+1 là tập kiểm tra doanh thu
DORMANT_WEEKS = 26
MIN_TRAIN_WEEKS = 26
SPIKE_RATIO = 20
FLAT_SHARE = 0.5         # tuần "phẳng": >= 50% ô có bán của nhà bán hàng là bội số của 7
FLAT_MIN_CELLS = 50      # chỉ xét tuần mà nhà bán hàng có >= 50 chuỗi bán
FLAT_MIN_WEEKS = 26      # nhà bán hàng bị coi là có giai đoạn phẳng nếu có >= 26 tuần như vậy


def load_wide(name):
    return pd.read_csv(os.path.join(RAW, name)).set_index(KEYS)


def save_table(df, stem):
    """Ghi parquet; nếu thiếu engine thì ghi csv. Trả về tên file đã ghi."""
    for ext in (".parquet", ".csv"):
        stale = os.path.join(OUT, stem + ext)
        if os.path.exists(stale):
            os.remove(stale)
    try:
        df.to_parquet(os.path.join(OUT, stem + ".parquet"), index=False)
        return stem + ".parquet"
    except ImportError:
        df.to_csv(os.path.join(OUT, stem + ".csv"), index=False)
        return stem + ".csv"


def unique_ids(index):
    c, w, p = (index.get_level_values(k).astype(str) for k in KEYS)
    return (c + "_" + w + "_" + p).to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    s0, s1 = load_wide("Phase 0 - Sales.csv"), load_wide("Phase 1 - Sales.csv")
    p0, p1 = load_wide("Phase 0 - Price.csv"), load_wide("Phase 1 - Price.csv")
    s2 = load_wide("Phase 2 - Sales.csv")
    sales = pd.concat([s0, s1.reindex(s0.index)], axis=1).sort_index()
    price = pd.concat([p0, p1], axis=1).reindex(index=sales.index, columns=sales.columns)
    weeks = pd.to_datetime(sales.columns)
    n_series_raw, n_weeks = sales.shape
    assert not sales.isna().any().any() and (np.diff(weeks.values).astype("timedelta64[D]").astype(int) == 7).all()

    S = sales.to_numpy(dtype=float)
    P_raw = price.to_numpy(dtype=float)
    cols = np.arange(n_weeks)

    # 1. Tuần bán đầu tiên; bỏ chuỗi không bán gì
    has_sale = (S > 0).any(axis=1)
    n_all_zero = int((~has_sale).sum())
    S, P_raw, index = S[has_sale], P_raw[has_sale], sales.index[has_sale]
    first = (S > 0).argmax(axis=1)
    active = cols[None, :] >= first[:, None]

    # 1b. Giai đoạn "phẳng": nhà bán hàng có số lượng tuần gần như toàn bội số của 7 trong thời gian dài
    #     (số liệu tuần được suy ra từ con số thô hơn). Các tuần này được GIỮ nhưng gắn cờ is_flat_period;
    #     lịch sử dùng cho đặc trưng, phân nhóm và huấn luyện bắt đầu từ history_start (tuần đầu sau giai đoạn phẳng).
    client = index.get_level_values("Client").to_numpy()
    flat_cut = {}
    for cl in np.unique(client):
        X = S[client == cl]
        n_sale = (X > 0).sum(axis=0)
        share = np.where(n_sale >= FLAT_MIN_CELLS, ((X > 0) & (X % 7 == 0)).sum(axis=0) / np.maximum(n_sale, 1), 0.0)
        flat_weeks = np.flatnonzero(share >= FLAT_SHARE)
        if len(flat_weeks) >= FLAT_MIN_WEEKS:
            flat_cut[int(cl)] = int(flat_weeks[-1]) + 1
    cut = np.array([flat_cut.get(int(cl), 0) for cl in client])
    start = np.maximum(first, cut)                       # history_start của từng chuỗi
    is_flat = active & (cols[None, :] < cut[:, None])

    # 2. Giá: 0 hoặc trống -> thiếu; forward fill rồi backward fill trong từng chuỗi
    price_was_zero = active & (P_raw <= 0)
    valid = np.where(active & (P_raw > 0), P_raw, np.nan)
    P = pd.DataFrame(valid).ffill(axis=1).bfill(axis=1).to_numpy()
    no_valid_price = np.isnan(valid).all(axis=1)

    # 3. Doanh thu
    revenue = np.where(S > 0, S * P, 0.0)

    # 4. zero_run: số tuần liên tiếp bằng 0 ngay TRƯỚC tuần hiện tại (trong phần đã hoạt động)
    zero_run = np.zeros(S.shape, dtype=np.int32)
    run = np.zeros(len(S), dtype=np.int32)
    for t in cols:
        zero_run[:, t] = run
        run = np.where((S[:, t] == 0) & (t >= start), run + 1, 0)   # không đếm số 0 của giai đoạn phẳng
    trailing_zero = run

    # 5. Đột biến: so với trung vị các tuần có bán trong giai đoạn huấn luyện
    n_train = n_weeks - TEST_WEEKS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hist = (S[:, :n_train] > 0) & ~is_flat[:, :n_train]
        med = np.nanmedian(np.where(hist, S[:, :n_train], np.nan), axis=1)
    is_spike = S > SPIKE_RATIO * med[:, None]

    # Bảng dài
    r, c = np.nonzero(active)
    uid = unique_ids(index)
    long = pd.DataFrame({
        "client": index.get_level_values("Client").to_numpy()[r],
        "warehouse": index.get_level_values("Warehouse").to_numpy()[r],
        "product": index.get_level_values("Product").to_numpy()[r],
        "unique_id": uid[r],
        "week": weeks.values[c],
        "sales": S[r, c].astype(np.int64),
        "price_raw": P_raw[r, c],
        "price": P[r, c],
        "revenue": revenue[r, c],
        "first_sale_week": weeks.values[first[r]],
        "weeks_since_first_sale": (c - first[r]).astype(np.int32),
        "zero_run": zero_run[r, c],
        "is_spike": is_spike[r, c],
        "price_was_zero": price_was_zero[r, c],
        "is_test": c >= n_train,
        "is_flat_period": is_flat[r, c],
        "history_start_week": weeks.values[start[r]],
    })
    long_name = save_table(long, "vn1_long")

    # Phase 2 (chỉ số lượng)
    s2 = s2.sort_index()
    r2, c2 = np.indices(s2.shape)
    r2, c2 = r2.ravel(), c2.ravel()
    weeks2 = pd.to_datetime(s2.columns)
    phase2 = pd.DataFrame({
        "client": s2.index.get_level_values("Client").to_numpy()[r2],
        "warehouse": s2.index.get_level_values("Warehouse").to_numpy()[r2],
        "product": s2.index.get_level_values("Product").to_numpy()[r2],
        "unique_id": unique_ids(s2.index)[r2],
        "week": weeks2.values[c2],
        "sales": s2.to_numpy(dtype=np.int64).ravel(),
    })
    p2_name = save_table(phase2, "vn1_phase2_sales")

    # Bảng tóm tắt theo chuỗi
    n_active = n_weeks - first
    n_history = n_weeks - start
    summary = pd.DataFrame({
        "client": index.get_level_values("Client"),
        "warehouse": index.get_level_values("Warehouse"),
        "product": index.get_level_values("Product"),
        "unique_id": uid,
        "first_sale_week": weeks.values[first],
        "n_weeks_active": n_active,
        "history_start_week": weeks.values[start],
        "n_weeks_history": n_history,
        "n_flat_weeks": is_flat.sum(axis=1),
        "n_zero_weeks": (active & (S == 0)).sum(axis=1),
        "total_sales": S.sum(axis=1).astype(np.int64),
        "total_revenue": np.where(no_valid_price, np.nan, np.nansum(revenue, axis=1)),
        "train_median_nonzero_sales": med,
        "n_spikes": is_spike.sum(axis=1),
        "n_price_was_zero": price_was_zero.sum(axis=1),
        "ends_with_zero_run_length": trailing_zero,
        "is_dormant": trailing_zero >= DORMANT_WEEKS,
        "too_short_for_backtest": n_history < TEST_WEEKS + MIN_TRAIN_WEEKS,
        "no_valid_price": no_valid_price,
    })
    summary.to_csv(os.path.join(OUT, "series_summary.csv"), index=False)

    # Nhật ký
    cells_raw = n_series_raw * n_weeks
    n_rows = len(long)
    sold = long["sales"] > 0
    n_zero_price = int(long["price_was_zero"].sum())
    n_filled = int((long["price_raw"].isna() & long["price"].notna()).sum()) + int((long["price_was_zero"] & long["price"].notna()).sum())
    n_no_price_rows = int(long["price"].isna().sum())
    rev_nan = int(long["revenue"].isna().sum())
    by_year = long.assign(year=long["week"].dt.year).groupby("year").agg(
        rows=("sales", "size"), sales=("sales", "sum"), revenue=("revenue", "sum"))
    split = long.groupby("is_test").agg(rows=("sales", "size"), series=("unique_id", "nunique"),
                                        sales=("sales", "sum"), revenue=("revenue", "sum"))
    fmt = lambda x: f"{x:,.0f}".replace(",", ".")
    L = [
        "# Nhật ký làm sạch dữ liệu VN1", "",
        f"Tạo bởi `scripts/01_clean.py`. Nguồn: `raw/` (không bị thay đổi).", "",
        "## Đầu vào", "",
        f"- Phase 0 + Phase 1 ghép liên tục: {fmt(n_series_raw)} chuỗi x {n_weeks} tuần "
        f"({weeks[0].date()} .. {weeks[-1].date()}) = {fmt(cells_raw)} ô chuỗi-tuần.",
        f"- Phase 2: {fmt(len(s2))} chuỗi x {s2.shape[1]} tuần ({weeks2[0].date()} .. {weeks2[-1].date()}), chỉ có số lượng.", "",
        "## Các bước", "",
        f"1. **Bỏ chuỗi không bán gì:** {n_all_zero} chuỗi bị bỏ; còn {fmt(len(S))} chuỗi.",
        f"2. **Bỏ các tuần trước tuần bán đầu tiên:** bỏ {fmt(cells_raw - n_all_zero * n_weeks - n_rows)} ô; "
        f"còn {fmt(n_rows)} dòng ({n_rows / cells_raw:.1%} số ô ban đầu).",
        f"3. **Giá bằng 0 khi có bán hàng coi là thiếu:** {fmt(n_zero_price)} ô (cờ `price_was_zero`), "
        f"thuộc {fmt(int((summary['n_price_was_zero'] > 0).sum()))} chuỗi.",
        f"4. **Điền giá (forward fill rồi backward fill trong chuỗi):** {fmt(n_filled)} ô được điền "
        f"({n_filled / n_rows:.1%} số dòng). {fmt(n_no_price_rows)} dòng thuộc {int(no_valid_price.sum())} chuỗi "
        f"không có giá hợp lệ nào nên `price` để trống (cờ `no_valid_price` trong bảng tóm tắt).",
        f"5. **Doanh thu:** `revenue = sales x price` ở tuần có bán, bằng 0 ở tuần không bán. "
        f"{fmt(rev_nan)} dòng có bán nhưng không tính được doanh thu (thuộc các chuỗi không có giá).",
        f"6. **Số 0 sau tuần bán đầu tiên được giữ nguyên:** {fmt(int((~sold).sum()))} dòng "
        f"({(~sold).mean():.1%}). `zero_run` = số tuần liên tiếp bằng 0 ngay trước tuần hiện tại.",
        f"7. **Đột biến (`is_spike`):** sales > {SPIKE_RATIO} x trung vị các tuần có bán trong giai đoạn huấn luyện "
        f"({n_train} tuần đầu). {fmt(int(long['is_spike'].sum()))} dòng thuộc "
        f"{fmt(int((summary['n_spikes'] > 0).sum()))} chuỗi; giá trị không bị sửa. "
        f"{fmt(int(np.isnan(med).sum()))} chuỗi chưa bán gì trong giai đoạn huấn luyện (ngoài giai đoạn phẳng) nên không xét đột biến.",
        "8. **Giai đoạn phẳng (`is_flat_period`):** " + ("; ".join(
            f"nhà bán hàng {cl}: số lượng là bội số của 7 ở phần lớn ô có bán cho đến {weeks[k - 1].date()}, "
            f"lịch sử dùng được bắt đầu từ {weeks[k].date()} ({fmt(int((client == cl).sum()))} chuỗi, "
            f"{fmt(int(is_flat[client == cl].sum()))} dòng bị gắn cờ)" for cl, k in flat_cut.items()) or "không phát hiện nhà bán hàng nào")
        + ". Các dòng này được giữ trong file nhưng không dùng cho đặc trưng, phân nhóm và huấn luyện; "
          "cột `history_start_week` cho biết lịch sử dùng được của từng chuỗi bắt đầu từ đâu. "
          f"`zero_run`, ngưỡng đột biến và cờ `too_short_for_backtest` đều tính từ `history_start_week`.", "",
        "## Chia tập", "",
        f"Tập kiểm tra doanh thu là {TEST_WEEKS} tuần cuối của Phase 0+1 "
        f"({weeks[n_train].date()} .. {weeks[-1].date()}), cột `is_test`.", "",
        "| Tập | Dòng | Chuỗi | Số lượng | Doanh thu |", "|---|---:|---:|---:|---:|",
    ]
    for flag, name in ((False, "Huấn luyện"), (True, "Kiểm tra")):
        row = split.loc[flag]
        L.append(f"| {name} | {fmt(row['rows'])} | {fmt(row['series'])} | {fmt(row['sales'])} | {fmt(row['revenue'])} |")
    L += ["", "## Doanh thu theo năm", "", "| Năm | Dòng | Số lượng | Doanh thu |", "|---|---:|---:|---:|"]
    for year, row in by_year.iterrows():
        L.append(f"| {year} | {fmt(row['rows'])} | {fmt(row['sales'])} | {fmt(row['revenue'])} |")
    L += [
        f"| Tổng | {fmt(n_rows)} | {fmt(long['sales'].sum())} | {fmt(long['revenue'].sum())} |", "",
        "## Cờ theo chuỗi (`series_summary.csv`)", "",
        f"- `is_dormant` (bằng 0 suốt {DORMANT_WEEKS} tuần cuối): {fmt(int(summary['is_dormant'].sum()))} chuỗi.",
        f"- `too_short_for_backtest` (ít hơn {TEST_WEEKS + MIN_TRAIN_WEEKS} tuần hoạt động): "
        f"{fmt(int(summary['too_short_for_backtest'].sum()))} chuỗi.",
        f"- `no_valid_price`: {int(no_valid_price.sum())} chuỗi.",
        f"- Không có cờ nào trong ba cờ trên: "
        f"{fmt(int((~(summary['is_dormant'] | summary['too_short_for_backtest'] | summary['no_valid_price'])).sum()))} chuỗi.", "",
        "## File đầu ra", "",
        f"- `{long_name}`: {fmt(n_rows)} dòng, {long.shape[1]} cột.",
        f"- `{p2_name}`: {fmt(len(phase2))} dòng, tổng {fmt(phase2['sales'].sum())} đơn vị.",
        f"- `series_summary.csv`: {fmt(len(summary))} dòng.", "",
        "## Lưu ý khi dùng", "",
        "- `price_raw` chỉ có giá trị ở tuần có bán hàng; không dùng nó (hay việc nó trống) làm đặc trưng.",
        "- `price` ở tuần không bán là giá của lần bán gần nhất, không phải giá niêm yết của tuần đó.",
        "- Tổng số lượng sau làm sạch bằng đúng tổng của raw vì chỉ bỏ các ô bằng 0.",
    ]
    with open(os.path.join(OUT, "cleaning_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    assert long["sales"].sum() == sales.to_numpy().sum()
    print(f"wrote {long_name} ({n_rows} rows), {p2_name} ({len(phase2)} rows), series_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
