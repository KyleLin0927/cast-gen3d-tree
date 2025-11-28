#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量處理 .litematic 檔：
1) 來源方塊清單可自訂（--sources），不只樹葉。
2) 兩種模式：
   - persistent：把來源方塊設為 persistent=true（保留其他屬性）
   - replace   ：把來源方塊整塊替換成 --target 指定的方塊
3) 固定輸出尺寸為 32x32x32：
   - X/Z 水平置中：不足補空氣，超過以中心為基準裁切
   - Y 只往上補空氣；若超過 32，保留底部 32 層（自上方裁切）
4) 檔名規則：tree-0001.litematic -> tree-001-<suffix>.litematic
   - 無數字則 <原名>-<suffix>.litematic
5) 一定輸出（不論是否有修改）
6) 相容 litemapy 多種 API / 簽名（含 0.11.0b0；支援 range_x/y/z 與相對/絕對座標）
7) 實體清除：
   - --strip-entities                 ：清除所有「實體 + 方塊實體」
   - --strip-only-entities           ：只清除「實體」
   - --strip-only-tile-entities      ：只清除「方塊實體」
   三者互斥（不可同時使用）
   - 注意：如果原始文件缺少 Entities/TileEntities 鍵（表示沒有實體數據），
     會自動跳過實體清除步驟，因為沒有實體需要處理
8) 跳過超尺寸：
   - --skip-if-oversize 當任一 Region 大於 32x32x32 且需要裁切時，整檔跳過不輸出
9) 遞迴搜尋所有子資料夾，並保持輸出資料夾結構與輸入資料夾結構相同
10) 錯誤處理：
    - --verbose-errors：顯示詳細的錯誤堆棧跟踪信息（用於調試）
    - 默認只顯示簡單錯誤信息（文件名: 錯誤消息）

重要說明：
- 本腳本「絕對不會修改原始文件」，所有操作都是只讀
- 如果遇到缺少 Entities/TileEntities 鍵的文件：
  * 會在內存中修復 NBT 數據（使用臨時文件）
  * 加載修復後的數據進行處理
  * 處理完成後刪除臨時文件
  * 原始文件保持不變
- 所有輸出都寫入到 --out 指定的輸出資料夾
"""

import argparse
import re
import traceback
from pathlib import Path
from typing import Iterable, Tuple

from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

from litemapy import Schematic, Region, BlockState

console = Console()


# ========================= 基本常數 =========================
DEFAULT_SOURCE_BLOCK_IDS = {
    "minecraft:oak_leaves",
    "minecraft:spruce_leaves",
    "minecraft:birch_leaves",
    "minecraft:jungle_leaves",
    "minecraft:acacia_leaves",
    "minecraft:dark_oak_leaves",
    "minecraft:mangrove_leaves",
    "minecraft:cherry_leaves",
}
AIR = BlockState("minecraft:air")
TARGET_SIZE = (32, 32, 32)
NAME_NUMERIC_RE = re.compile(r"(.*?)(\d+)(\.litematic)$", re.IGNORECASE)


# ========================= 工具函式 =========================
def parse_sources(arg_val: str | None) -> set[str]:
    """
    --sources 的解析：
    - None：使用預設樹葉集合
    - "id1,id2,id3"：逗號分隔的方塊 ID
    - "@/path/to/list.txt"：每行一個方塊 ID，# 開頭為註解
    """
    if arg_val is None:
        return set(DEFAULT_SOURCE_BLOCK_IDS)

    if arg_val.startswith("@"):
        p = Path(arg_val[1:]).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"來源清單檔不存在：{p}")
        ids = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
        return ids

    return {s.strip() for s in arg_val.split(",") if s.strip()}


def build_output_name(src_name: str, suffix: str) -> str:
    """
    tree-0001.litematic -> tree-001-<suffix>.litematic
    抓不到數字：name-<suffix>.litematic
    """
    if not suffix:
        return src_name
    m = NAME_NUMERIC_RE.match(src_name)
    if not m:
        base, _ = src_name.rsplit(".", 1)
        return f"{base}-{suffix}.litematic"
    prefix, digits, ext = m.groups()
    num = int(digits)
    return f"{prefix}{num:03d}-{suffix}{ext}"


def region_ranges(region: Region) -> Tuple[list, list, list]:
    """
    取得區域的真實座標軸（考慮反向/負尺寸）：
    優先使用 range_x()/range_y()/range_z()，
    沒有則退回 xrange()/yrange()/zrange()。
    """
    if all(hasattr(region, n) for n in ("range_x", "range_y", "range_z")):
        return list(region.range_x()), list(region.range_y()), list(region.range_z())
    elif all(hasattr(region, n) for n in ("xrange", "yrange", "zrange")):
        return list(region.xrange()), list(region.yrange()), list(region.zrange())
    else:
        # 最後保底：以欄位估尺寸推 [0..size-1]
        sx, sy, sz = region_dims_by_fields(region)
        return list(range(sx)), list(range(sy)), list(range(sz))


def region_dims_by_fields(region: Region) -> Tuple[int, int, int]:
    """從欄位估尺寸（保底用）。"""
    if hasattr(region, "size"):
        sx, sy, sz = region.size
        return int(sx), int(sy), int(sz)
    if all(hasattr(region, a) for a in ("width", "height", "length")):
        return int(region.width), int(region.height), int(region.length)
    if all(hasattr(region, a) for a in ("size_x", "size_y", "size_z")):
        return int(region.size_x), int(region.size_y), int(region.size_z)
    raise AttributeError("Unsupported Region API: cannot determine region dimensions")


def region_dims(region: Region) -> Tuple[int, int, int]:
    """以 range_* 長度為準的尺寸（涵蓋負尺寸/反向）。"""
    rx, ry, rz = region_ranges(region)
    return len(rx), len(ry), len(rz)


def get_block_compat(region: Region, x: int, y: int, z: int) -> BlockState:
    """
    相容讀取：先用索引語法；失敗再 getblock。
    x,y,z 必須是「該區域自身座標系」中的座標（來自 range_*）。
    """
    try:
        return region[x, y, z]
    except Exception:
        pass
    if hasattr(region, "getblock"):
        return region.getblock(x, y, z)
    raise RuntimeError("Region doesn't support index or getblock for reading")


# --- Region 相容建構子（支援多簽名） ---
def make_region(tx: int, ty: int, tz: int, name: str | None = None) -> Region:
    """
    自動相容各種 litemapy Region.__init__ 簽名：
    可能的參數含：width/height/length、x/y/z、name，順序也可能不同。
    若需要 x/y/z 就一律帶 0,0,0。僅在支援時才傳 name。
    """
    from litemapy import Region  # 保持與匯入一致
    import inspect

    sig = inspect.signature(Region.__init__)
    params = list(sig.parameters.values())[1:]  # 跳過 self
    names = [p.name for p in params]

    needs_whl = {"width", "height", "length"}.issubset(names)
    has_xyz = {"x", "y", "z"}.issubset(names)
    has_name = "name" in names

    # 1) 具名參數（最穩）
    if needs_whl:
        kwargs = {"width": tx, "height": ty, "length": tz}
        if has_xyz:
            kwargs.update({"x": 0, "y": 0, "z": 0})
        if has_name and name is not None:
            kwargs["name"] = name
        return Region(**kwargs)

    # 2) 依參數名順序用位置參數
    if len(names) >= 6 and names[:6] == ["x", "y", "z", "width", "height", "length"]:
        args = [0, 0, 0, tx, ty, tz]
        if len(names) >= 7 and names[6] == "name" and name is not None:
            args.append(name)
        return Region(*args)
    if len(names) >= 6 and names[:6] == ["width", "height", "length", "x", "y", "z"]:
        args = [tx, ty, tz, 0, 0, 0]
        if len(names) >= 7 and names[6] == "name" and name is not None:
            args.append(name)
        return Region(*args)
    if len(names) >= 7 and names[:7] == ["name", "x", "y", "z", "width", "height", "length"]:
        return Region(name if name is not None else "", 0, 0, 0, tx, ty, tz)

    # 3) 落回：逐一嘗試常見組合
    attempts = [
        (tx, ty, tz),
        (tx, ty, tz, 0, 0, 0),
        (0, 0, 0, tx, ty, tz),
        (tx, ty, tz, name) if name is not None else None,
        (tx, ty, tz, 0, 0, 0, name) if name is not None else None,
        (name, tx, ty, tz) if name is not None else None,
        (name, 0, 0, 0, tx, ty, tz) if name is not None else None,
    ]
    last_err = None
    for at in attempts:
        if at is None:
            continue
        try:
            return Region(*at)
        except TypeError as e:
            last_err = e
            continue

    raise TypeError(f"Unsupported Region constructor signature: {sig} (last error: {last_err})")


# ========================= palette 轉換 =========================
def map_blockstate(bs: BlockState, mode: str, source_ids: set[str], target_id: str | None) -> BlockState:
    """
    將 palette 中的 BlockState 做對應轉換。
    - mode == "persistent"：若在來源清單內 -> 設 persistent=true（保留其他屬性）
    - mode == "replace"   ：若在來源清單內 -> 換成 target_id（完全替換為另一方塊）
    其他：原樣回傳
    """
    if bs.id not in source_ids:
        return bs

    if mode == "persistent":
        return bs.with_properties(persistent="true")

    if mode == "replace":
        if not target_id:
            raise ValueError("mode=replace 需要 --target 方塊 id")
        return BlockState(target_id)

    raise ValueError(f"未知的 mode：{mode}")


def transform_palette_in_region(region: Region, mode: str, source_ids: set[str], target_id: str | None) -> None:
    """對單一 Region 使用 palette 級別的 filter：極快。"""
    region.filter(lambda bs: map_blockstate(bs, mode, source_ids, target_id))


# ========================= 32x32x32 重製（用 range_* 真座標） =========================
def crop_or_pad_to_32(region: Region) -> Region:
    """
    產生新的 32x32x32 Region：
      * X/Z 置中補空氣或置中裁切
      * Y 往上補空氣或從上方裁切（保留底部 32）
    使用 range_x/y/z 取得真實座標，避免負尺寸/反向座標造成全空氣。
    """
    rx, ry, rz = region_ranges(region)
    sx, sy, sz = len(rx), len(ry), len(rz)
    tx, ty, tz = TARGET_SIZE

    def calc_span(src_n: int, tgt_n: int, center: bool) -> tuple[int, int, int]:
        """回傳 (src_start_idx, src_end_idx, dst_start_idx) —— 針對 rx/ry/rz 的 index 範圍。"""
        if src_n < tgt_n:
            dst_start = (tgt_n - src_n) // 2 if center else 0
            return 0, src_n, dst_start
        else:
            src_start = (src_n - tgt_n) // 2 if center else 0
            return src_start, src_start + tgt_n, 0

    # X/Z 置中；Y 自底開始（>32 僅保留底下 32）
    ix0, ix1, dx0 = calc_span(sx, tx, center=True)
    iz0, iz1, dz0 = calc_span(sz, tz, center=True)
    iy0, iy1, dy0 = (0, min(sy, ty), 0)

    # 建目標 32³ 區域
    new_reg = make_region(tx, ty, tz)

    # 先填空氣（陣列語法；setblock 已棄用）
    for x in range(tx):
        for y in range(ty):
            for z in range(tz):
                new_reg[x, y, z] = AIR

    # 用「index -> 真座標值」來讀寫
    for ix in range(ix0, ix1):
        vx = rx[ix]
        for iy in range(iy0, iy1):
            vy = ry[iy]
            for iz in range(iz0, iz1):
                vz = rz[iz]
                try:
                    bs = get_block_compat(region, vx, vy, vz)
                except Exception:
                    continue
                new_reg[ix - ix0 + dx0, iy - iy0 + dy0, iz - iz0 + dz0] = bs

    return new_reg


# ========================= 實體清除 =========================
def _clear_container(obj, attr) -> int:
    if not hasattr(obj, attr):
        return 0
    cont = getattr(obj, attr)
    try:
        n = len(cont)
    except Exception:
        n = 0
    try:
        if isinstance(cont, (list, dict, set)):
            cont.clear()
        else:
            setattr(obj, attr, [])
    except Exception:
        pass
    return n


def clear_entities_on_region(region: Region, mode: str) -> int:
    """
    清除單一 Region 的實體資料。
    mode: 'all' | 'entities' | 'tiles' | 'none'
    回傳估計移除數（盡力估計，不保證精準）。
    """
    if mode == "none":
        return 0
    removed = 0

    # 優先試 dedicated methods（若存在）
    if mode in ("all", "entities"):
        for fn in ("clear_entities",):
            if hasattr(region, fn):
                try:
                    before = 0
                    for attr in ("entities", "Entities"):
                        if hasattr(region, attr):
                            try:
                                before += len(getattr(region, attr))
                            except Exception:
                                pass
                    getattr(region, fn)()
                    removed += before
                except Exception:
                    pass

    if mode in ("all", "tiles"):
        for fn in ("clear_tile_entities", "clear_tileentities"):
            if hasattr(region, fn):
                try:
                    before = 0
                    for attr in ("tile_entities", "tileentities", "TileEntities"):
                        if hasattr(region, attr):
                            try:
                                before += len(getattr(region, attr))
                            except Exception:
                                pass
                    getattr(region, fn)()
                    removed += before
                except Exception:
                    pass

    # 再粗暴清空容器（不同版本命名不同）
    if mode in ("all", "entities"):
        for attr in ("entities", "Entities"):
            removed += _clear_container(region, attr)
    if mode in ("all", "tiles"):
        for attr in ("tile_entities", "tileentities", "TileEntities"):
            removed += _clear_container(region, attr)

    return removed


def clear_all_entities(schem: Schematic, mode: str) -> int:
    """
    清除整個 Schematic 的實體資料（逐 Region 與頂層）。
    mode: 'all' | 'entities' | 'tiles' | 'none'
    """
    if mode == "none":
        return 0
    total = 0
    # Region 級
    for reg in schem.regions.values():
        total += clear_entities_on_region(reg, mode)
    # Schematic 頂層（保險）
    if mode in ("all", "entities"):
        total += _clear_container(schem, "entities")
        total += _clear_container(schem, "Entities")
    if mode in ("all", "tiles"):
        total += _clear_container(schem, "tile_entities")
        total += _clear_container(schem, "tileentities")
        total += _clear_container(schem, "TileEntities")
    return total


# ========================= NBT 修復 =========================
def safe_load_schematic(file_path: Path) -> tuple[Schematic, bool]:
    """
    安全地加載 Schematic，如果遇到缺失 Entities/TileEntities 鍵的錯誤，會在內存中修復（不保存文件）。
    返回 (Schematic, has_entities)
    - has_entities: False 表示原始文件缺少 Entities/TileEntities 鍵（沒有實體數據）
    """
    try:
        # 先嘗試正常加載
        schem = Schematic.load(str(file_path))
        return schem, True  # 正常加載，假設有實體數據
    except KeyError as e:
        # 如果是 Entities 或 TileEntities 相關的 KeyError，在內存中修復
        if "Entities" in str(e) or "TileEntities" in str(e):
            try:
                import nbtlib
                from nbtlib import File
                import tempfile
                
                # 讀取 NBT 文件（.litematic 文件是 gzipped 的）
                nbt_file = File.load(str(file_path), gzipped=True)
                
                # 檢查並修復每個 Region（在內存中）
                if "Regions" in nbt_file:
                    for region_name, region_data in nbt_file["Regions"].items():
                        if "Entities" not in region_data:
                            region_data["Entities"] = nbtlib.List([])
                        if "TileEntities" not in region_data:
                            region_data["TileEntities"] = nbtlib.List([])
                    
                    # 保存到臨時文件（不修改原始文件）
                    with tempfile.NamedTemporaryFile(suffix=".litematic", delete=False) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                    
                    try:
                        nbt_file.save(str(tmp_path), gzipped=True)
                        # 從臨時文件加載 Schematic
                        schem = Schematic.load(str(tmp_path))
                        # 刪除臨時文件
                        tmp_path.unlink()
                        # 返回 Schematic 和標記（False 表示原始文件沒有實體數據）
                        return schem, False
                    except Exception:
                        # 如果失敗，清理臨時文件
                        if tmp_path.exists():
                            tmp_path.unlink()
                        raise
            except Exception as fix_err:
                # 如果修復失敗，重新拋出原始錯誤
                raise e from fix_err
        
        # 如果不是 Entities/TileEntities 相關的錯誤，直接拋出
        raise


# ========================= 超尺寸檢查 =========================
def is_oversize(schem: Schematic, target=(32, 32, 32)) -> bool:
    tx, ty, tz = target
    for reg in schem.regions.values():
        sx, sy, sz = region_dims(reg)
        if sx > tx or sy > ty or sz > tz:
            return True
    return False


# ========================= 單檔處理 =========================
def process_one_file(
    src: Path,
    out_dir: Path,
    mode: str,
    source_ids: set[str],
    target_id: str | None,
    suffix: str,
    keep_original_regions: bool = False,
    strip_mode: str = "none",  # 'none' | 'all' | 'entities' | 'tiles'
    skip_if_oversize: bool = False,
    src_root: Path | None = None,  # 新增：來源根目錄，用於計算相對路徑
    dst_root: Path | None = None,  # 新增：輸出根目錄，用於鏡像結構
    verbose_errors: bool = False,  # 是否顯示詳細錯誤信息
) -> tuple[str, str]:
    """
    回傳 (status, 訊息/路徑)
    status: 'ok' | 'skip' | 'err'
    若提供 src_root 與 dst_root，會保留相對路徑結構；否則直接輸出到 out_dir
    """
    try:
        schem, has_entities = safe_load_schematic(src)

        # 若設定跳過超尺寸，且這次又會裁切（沒有 keep_original_regions），就直接跳過不輸出
        if skip_if_oversize and not keep_original_regions and is_oversize(schem, TARGET_SIZE):
            return "skip", f"{src.name}: skipped due to oversize (>32³)"

        # palette 級別轉換
        for region in schem.regions.values():
            transform_palette_in_region(region, mode, source_ids, target_id)

        # 固定 32x32x32：逐 key 覆蓋（不可對 schem.regions 整體賦值）
        if not keep_original_regions:
            for name in list(schem.regions.keys()):
                old_reg = schem.regions[name]
                schem.regions[name] = crop_or_pad_to_32(old_reg)

        # 清除實體（若指定且文件有實體數據）
        removed = 0
        if strip_mode != "none":
            if has_entities:
                removed = clear_all_entities(schem, strip_mode)
            # 如果 has_entities 為 False，表示原始文件沒有實體數據，跳過實體清除步驟

        # 計算輸出路徑（保留相對路徑結構）
        if src_root is not None and dst_root is not None:
            rel_path = src.relative_to(src_root)
            out_name = build_output_name(rel_path.name, suffix)
            out_path = dst_root / rel_path.parent / out_name
        else:
            out_name = build_output_name(src.name, suffix)
            out_path = out_dir / out_name
        
        # 確保輸出目錄存在
        out_path.parent.mkdir(parents=True, exist_ok=True)
        schem.save(str(out_path))

        if strip_mode != "none":
            return "ok", f"{out_path} (entities removed ~{removed})"
        return "ok", str(out_path)

    except Exception as e:
        if verbose_errors:
            # 显示详细的错误信息，包括异常类型和完整堆栈跟踪
            exc_type = type(e).__name__
            exc_msg = str(e)
            tb_full = traceback.format_exc()
            return "err", f"{src.name}: [{exc_type}] {exc_msg}\n{tb_full}"
        else:
            # 保持原来的简单错误信息格式
            return "err", f"{src.name}: {e}"


def iter_litematics(indir: Path) -> Iterable[Path]:
    """遞迴收集 indir 底下所有 .litematic（含所有子資料夾）"""
    yield from sorted(indir.rglob("*.litematic"))


# ========================= CLI 主程式 =========================
def main():
    ap = argparse.ArgumentParser(description="批量修改 .litematic，並轉成 32x32x32（遞迴搜尋所有子資料夾並保持結構）")
    ap.add_argument("--in", dest="indir", required=True, help="來源資料夾（會遞迴搜尋所有子資料夾）")
    ap.add_argument("--out", dest="outdir", required=True, help="輸出資料夾（會鏡像來源資料夾的子資料夾結構）")
    ap.add_argument(
        "--mode",
        choices=["persistent", "replace"],
        default="persistent",
        help="persistent=把來源方塊設 persistent=true；replace=把來源方塊換成 --target",
    )
    ap.add_argument(
        "--sources",
        default=None,
        help="來源方塊清單：逗號分隔的方塊ID，或 '@檔案路徑'（每行一個）。未提供則使用 8 種樹葉。",
    )
    ap.add_argument(
        "--target",
        default="minecraft:lime_wool",
        help="mode=replace 時要替換成的方塊 ID（預設 minecraft:lime_wool）",
    )
    ap.add_argument("--suffix", default=None, help="輸出檔名要附加的字串（例：mytag），未提供則維持原檔名。")
    ap.add_argument(
        "--keep-original-regions",
        action="store_true",
        help="保留原 Region 尺寸（不轉 32x32x32）。預設會轉。",
    )
    # 清除實體選項（互斥）
    ap.add_argument("--strip-entities", action="store_true", help="清除所有實體（entities 與 tile entities）")
    ap.add_argument("--strip-only-entities", action="store_true", help="只清除實體（entities）")
    ap.add_argument("--strip-only-tile-entities", action="store_true", help="只清除方塊實體（tile entities）")

    # 超尺寸跳過
    ap.add_argument(
        "--skip-if-oversize",
        action="store_true",
        help="若任一 Region 超過 32x32x32 且會裁切，則整檔跳過不輸出",
    )

    # 詳細錯誤信息
    ap.add_argument(
        "--verbose-errors",
        action="store_true",
        help="顯示詳細的錯誤堆棧跟踪信息（用於調試）",
    )

    args = ap.parse_args()

    # 實體選項互斥檢查
    chosen = sum(
        bool(x)
        for x in (args.strip_entities, args.strip_only_entities, args.strip_only_tile_entities)
    )
    if chosen > 1:
        raise SystemExit("錯誤：--strip-entities / --strip-only-entities / --strip-only-tile-entities 只能擇一使用")

    if args.strip_entities:
        strip_mode = "all"
    elif args.strip_only_entities:
        strip_mode = "entities"
    elif args.strip_only_tile_entities:
        strip_mode = "tiles"
    else:
        strip_mode = "none"

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not indir.is_dir():
        raise SystemExit(f"來源資料夾不存在：{indir}")

    source_ids = parse_sources(args.sources)

    files = list(iter_litematics(indir))
    if not files:
        raise SystemExit(f"來源資料夾及其子資料夾內找不到 .litematic：{indir}")

    rprint(f"[bold]模式[/bold]: {args.mode}")
    rprint(f"[bold]來源方塊數[/bold]: {len(source_ids)}")
    rprint(f"[bold]來源資料夾[/bold]: {indir}")
    rprint(f"[bold]輸出資料夾[/bold]: {outdir}")
    rprint(f"[bold]固定輸出尺寸[/bold]: 32x32x32  (除非指定 --keep-original-regions)")
    if args.suffix:
        rprint(f"[bold]輸出檔名 suffix[/bold]: {args.suffix}")
    if args.mode == "replace":
        rprint(f"[bold]目標方塊[/bold]: {args.target}")
    if strip_mode != "none":
        rprint(f"[bold yellow]將清除實體：{strip_mode}[/bold yellow]")
    if args.skip_if_oversize and not args.keep_original_regions:
        rprint(f"[bold yellow]遇到超過 32³ 的檔案將跳過[/bold yellow]")
    rprint()

    ok, fail, skipped = 0, 0, 0
    skip_logs = []
    fail_logs = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]處理中...", total=len(files))
        
        for f in files:
            status, msg = process_one_file(
                f,
                outdir,
                args.mode,
                source_ids,
                args.target if args.mode == "replace" else None,
                args.suffix,
                keep_original_regions=args.keep_original_regions,
                strip_mode=strip_mode,
                skip_if_oversize=args.skip_if_oversize,
                src_root=indir,  # 傳入來源根目錄
                dst_root=outdir,  # 傳入輸出根目錄
                verbose_errors=args.verbose_errors,  # 傳入詳細錯誤選項
            )
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
                skip_logs.append(msg)  # 先收集，不在迴圈內印
            else:
                fail += 1
                fail_logs.append(msg)  # 先收集，不在迴圈內印
            
            # 更新進度條描述
            progress.update(task, advance=1, description=f"[cyan]處理中... (✓{ok} ⊘{skipped} ✗{fail})")

    # 進度條結束後，再一次性列出訊息
    if skip_logs:
        rprint("\n[bold yellow]跳過清單（oversize）[/bold yellow]")
        for s in skip_logs:
            rprint(f"[yellow]跳過[/yellow] {s}")
    if fail_logs:
        rprint("\n[bold red]失敗清單[/bold red]")
        for s in fail_logs:
            # 如果错误信息包含换行符，分别显示
            if "\n" in s:
                lines = s.split("\n", 1)
                rprint(f"[red]失敗[/red] {lines[0]}")
                rprint(f"[dim]{lines[1]}[/dim]")
            else:
                rprint(f"[red]失敗[/red] {s}")

    rprint(f"\n[bold green]成功[/bold green]: {ok}  |  [bold yellow]跳過[/bold yellow]: {skipped}  |  [bold red]失敗[/bold red]: {fail}")

if __name__ == "__main__":
    main()