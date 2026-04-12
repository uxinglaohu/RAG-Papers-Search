"""
ACL 2025 Volume 1: Long Papers 批量下载脚本
下载地址: https://aclanthology.org/2025.acl-long.{1-1603}.pdf
共 1603 篇论文

依赖: pip install requests tqdm
"""

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============ 配置 ============
SAVE_DIR = "./data/papers"
START_ID = 1
END_ID = 1603
MAX_WORKERS = 5
RETRY_TIMES = 3
TIMEOUT = 30
BASE_URL = "https://aclanthology.org/2025.acl-long.{}.pdf"
# ==============================


def download_one(paper_id: int) -> tuple[int, bool, str]:
    """下载单篇论文，返回 (编号, 是否成功, 信息)"""
    url = BASE_URL.format(paper_id)
    filename = f"2025.acl-long.{paper_id}.pdf"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        return paper_id, True, "跳过"

    for attempt in range(1, RETRY_TIMES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                return paper_id, True, "OK"
            elif resp.status_code == 404:
                return paper_id, False, "404"
            else:
                time.sleep(2)
        except Exception as e:
            if attempt < RETRY_TIMES:
                time.sleep(2)
            else:
                return paper_id, False, str(e)

    return paper_id, False, "重试失败"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    total = END_ID - START_ID + 1
    fail_list = []

    print(f"ACL 2025 Long Papers 下载中...")
    print(f"保存目录: {os.path.abspath(SAVE_DIR)}\n")

    pbar = tqdm(total=total, desc="下载进度", unit="篇",
                bar_format="{l_bar}{bar:30}{r_bar}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one, pid): pid
            for pid in range(START_ID, END_ID + 1)
        }

        for future in as_completed(futures):
            pid, ok, msg = future.result()
            if not ok:
                fail_list.append(pid)
            pbar.update(1)
            pbar.set_postfix_str(f"Paper {pid}: {msg}")

    pbar.close()

    # 汇总
    success = total - len(fail_list)
    print(f"\n完成！成功: {success}/{total}，失败: {len(fail_list)}/{total}")
    if fail_list:
        fail_list.sort()
        print(f"失败编号: {fail_list[:20]}{'...' if len(fail_list) > 20 else ''}")
        with open(os.path.join(SAVE_DIR, "failed.txt"), "w") as f:
            for pid in fail_list:
                f.write(f"{pid}\n")
        print(f"失败列表已保存到 {SAVE_DIR}/failed.txt")


if __name__ == "__main__":
    main()