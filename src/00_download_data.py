# -*- coding: utf-8 -*-
"""
00_download_data.py  下载 NASA PCoE 电池老化数据(GitHub 全量镜像,支持断点续传)

用法: python src/00_download_data.py
数据源: https://github.com/devhaaana/nasa-battery-aging-dataset (NASA PCoE 官方数据镜像)
"""
import os
import subprocess
import sys

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nasa")
os.makedirs(RAW, exist_ok=True)

BASE = "https://raw.githubusercontent.com/devhaaana/nasa-battery-aging-dataset/HEAD/data"
FILES = ["B0005", "B0006", "B0007", "B0018",   # 24 ℃ / 2A 常温基准
         "B0033", "B0034", "B0036",           # 24 ℃ / 4A(B36 为 2A)
         "B0029", "B0030", "B0031", "B0032",  # 43 ℃ / 4A
         "B0053", "B0054", "B0055", "B0056",  # 4 ℃ / 2A
         "B0041", "B0042", "B0043", "B0044"]  # 4 ℃ / 1A·4A 混合(备选)
SIZES = {"B0005": 15956874, "B0006": 16028159, "B0007": 16049460, "B0018": 8502218,
         "B0029": 2879840, "B0030": 2880995, "B0031": 2882510, "B0032": 2918592,
         "B0033": 11057919, "B0034": 11091182, "B0036": 11404543,
         "B0041": 3733558, "B0042": 7499364, "B0043": 7483891, "B0044": 7511915,
         "B0053": 3347771, "B0054": 7089384, "B0055": 7199776, "B0056": 7279819}
README_FILES = ["README_05_06_07_18.txt", "README_25_26_27_28.txt", "README_29_30_31_32.txt",
                "README_33_34_36.txt", "README_41_42_43_44.txt", "README_53_54_55_56.txt"]


def fetch(url, dest, expect=None):
    if expect and os.path.exists(dest) and os.path.getsize(dest) == expect:
        print(f"  跳过(已存在): {os.path.basename(dest)}")
        return True
    r = subprocess.run(["curl", "-sL", "-C", "-", "--retry", "8", "--retry-all-errors",
                        "-o", dest, url, "--connect-timeout", "15", "--max-time", "600"])
    ok = r.returncode == 0 and (expect is None or os.path.getsize(dest) == expect)
    print(f"  {'OK ' if ok else 'FAIL'} {os.path.basename(dest)} "
          f"({os.path.getsize(dest) if os.path.exists(dest) else 0} B)")
    return ok


def main():
    print("下载 NASA PCoE 数据到 data/raw/nasa/ ...")
    fails = 0
    for b in FILES:
        fails += 0 if fetch(f"{BASE}/{b}.mat", os.path.join(RAW, f"{b}.mat"), SIZES[b]) else 1
    for f in README_FILES:
        fetch(f"{BASE}/{f}", os.path.join(RAW, f))
    print("全部完成" if fails == 0 else f"{fails} 个文件失败,重跑本脚本可断点续传")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
