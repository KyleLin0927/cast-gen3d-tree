import os
import sys

def rename_files(folder_path, a, b, c, start_number=1):
    """
    將資料夾內所有「非隱藏」檔案重新命名為 {a}-{b}-{c}-{number}.{ext}
    - 不修改隱藏檔 (檔名以 '.' 開頭) 與任何資料夾
    - 若偵測到名稱衝突，會先把所有目標檔案改為「臨時名稱」再統一改名
    - 數字部分固定三位數（001, 012, 105）
    """
    if not os.path.isdir(folder_path):
        print(f"❌ 找不到資料夾：{folder_path}")
        return

    # 只挑「非隱藏 且 是檔案」；不處理資料夾與隱藏檔
    files = [
        f for f in os.listdir(folder_path)
        if not f.startswith(".") and os.path.isfile(os.path.join(folder_path, f))
    ]
    files.sort()

    # 預先生成最終檔名（與原本順序對齊）
    planned_names = []
    number = start_number
    for filename in files:
        _, ext = os.path.splitext(filename)
        planned_names.append(f"{a}-{b}-{c}-{number:03d}{ext}")
        number += 1

    # 檢查是否有任一最終檔名與目前資料夾內「任何檔案」衝突（含非目標檔、含隱藏檔）
    conflict = any(
        os.path.exists(os.path.join(folder_path, name)) for name in planned_names
    )

    # 若有衝突：先把所有目標檔案改成臨時名稱，避免互相覆蓋或撞到既有檔名
    temp_paths = []  # 與 files/planned_names 對齊
    if conflict:
        print("⚠️ 偵測到名稱衝突，先進行臨時改名以確保安全...")
        for i, filename in enumerate(files):
            old_path = os.path.join(folder_path, filename)
            base_ext = os.path.splitext(filename)[1]
            # 產生不會撞名的臨時名稱
            j = 0
            while True:
                temp_name = f"__temp__{i}_{j}{base_ext}"
                temp_path = os.path.join(folder_path, temp_name)
                if not os.path.exists(temp_path):
                    break
                j += 1
            os.rename(old_path, temp_path)
            temp_paths.append(temp_path)
        # 後續改名以 temp_paths 為準
        src_paths = temp_paths
    else:
        # 無衝突就直接以原檔名改名
        src_paths = [os.path.join(folder_path, f) for f in files]

    # 第二階段：將來源（可能是臨時檔）改為最終檔名
    number = start_number
    for src_path, final_name in zip(src_paths, planned_names):
        _, ext = os.path.splitext(src_path)
        # 以 planned_names 的副檔名為準；若需保留原來源副檔名，可改成：
        # final_name = f"{a}-{b}-{c}-{number:03d}{ext}"
        dst_path = os.path.join(folder_path, final_name)
        os.rename(src_path, dst_path)
        print(f"✅ {os.path.basename(src_path)} → {final_name}")
        number += 1

    print("🎉 全部改名完成！（已跳過隱藏檔與資料夾）")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("使用方式：python rename_files.py <folder_path> <a> <b> <c> <start_number>")
    else:
        folder_path = sys.argv[1]
        a = sys.argv[2]
        b = sys.argv[3]
        c = sys.argv[4]
        start_number = int(sys.argv[5])
        rename_files(folder_path, a, b, c, start_number)