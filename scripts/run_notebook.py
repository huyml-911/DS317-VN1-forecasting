"""Chuyển file .py (định dạng percent của jupytext) thành .ipynb và chạy toàn bộ, lưu kèm kết quả.

Dùng:  python scripts/run_notebook.py notebooks/04_chat_luong_du_lieu.py

Viết riêng vì máy không có nbconvert/nbclient; chỉ cần jupytext, nbformat, jupyter_client, ipykernel.
"""
import os
import sys

import jupytext
import nbformat
from jupyter_client.manager import start_new_kernel
from nbformat.v4 import new_output


def run(src):
    nb = jupytext.read(src)
    out_path = os.path.splitext(src)[0] + ".ipynb"
    km, kc = start_new_kernel(kernel_name="python3", cwd=os.path.dirname(os.path.abspath(src)))
    failed = False
    try:
        count = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            msg_id = kc.execute(cell.source)
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=1800)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "status" and c["execution_state"] == "idle":
                    break
                if t == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                        outputs[-1]["text"] += c["text"]
                    else:
                        outputs.append(new_output("stream", name=c["name"], text=c["text"]))
                elif t in ("display_data", "execute_result"):
                    kw = {"execution_count": count} if t == "execute_result" else {}
                    outputs.append(new_output(t, data=c["data"], metadata=c.get("metadata", {}), **kw))
                elif t == "error":
                    outputs.append(new_output("error", ename=c["ename"], evalue=c["evalue"], traceback=c["traceback"]))
                    failed = True
                    print(f"Cell {count} failed: {c['ename']}: {c['evalue']}", file=sys.stderr)
            cell["execution_count"] = count
            cell["outputs"] = outputs
            if failed:
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    nbformat.write(nb, out_path)
    print(f"wrote {out_path}" + (" (stopped at first error)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
