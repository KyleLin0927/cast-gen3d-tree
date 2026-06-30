#!/usr/bin/env python3
"""
3D voxel 算圖（matplotlib）：把單一 voxel 樣本畫成立體方塊圖，產生論文常見的
「3/4 視角 + 柔光」定性圖（qualitative / hero 圖）。

與 ``voxel_orthoslices`` / ``voxel_label_projections`` 分工：
  - projections（三視圖）：壓掉深度的快速總覽。
  - orthoslices（切片）：逐層截面，診斷「中空椅 vs 實心 blob」。
  - **render（本檔）：立體方塊圖，最接近論文裡那種「視覺性高」的 3D ShapeNet 圖**。

刻意只用 matplotlib（無 Blender/Mitsuba 重依賴），所以能在訓練機上跟 sampling 一起跑、
也能被 generate 腳本內嵌呼叫。品質定位是「乾淨體面」而非雜誌級；要雜誌級 hero 圖，請另外
把 voxel 匯出到 Blender 離線算圖。

座標慣例（與 ``voxel_label_projections`` / ``voxel_sample_metrics`` 一致）：
  labels 形狀為 ``[Z, Y, X]``，其中 **Y（axis=1）為高度（上）**。
  繪圖時轉置成 ``[X, Z, Y]``，讓 Y（高度）對到 matplotlib 3D 的垂直軸，椅子會直立。
  視角預設 ``elev=22, azim=-60``（3/4 俯視）；若椅子朝向不理想，調 ``azim`` 即可。

著色沿用 ``voxel_label_projections`` 的配色，與三視圖/切片視覺一致：
  - "component"（預設）：wood 以 26-connectivity 上色（最大塊一色、其餘碎塊警示色）、
    leaf 綠。一眼看出「連通」與「浮空碎塊」。
  - "clay"：全部佔據用單一霧面陶土色——最接近論文 hero 圖的乾淨質感。
  - "occupancy"：純佔據中性亮色，最乾淨地讀密度。

於函式內 import pyplot，呼叫端可先執行 ``matplotlib.use("Agg")``。
本模組只依賴 numpy（+ 選用 scipy/matplotlib），不 import train_unet_diffusion，故不需 torch、
也不會觸發 voxel_sample_metrics 的循環匯入。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

try:  # 套件脈絡（from utils.voxel_render import ...）
    from .voxel_label_projections import (
        DISPLAY_AIR,
        DISPLAY_COLORS,
        DISPLAY_FRAGMENT,
        DISPLAY_LARGEST,
        DISPLAY_LEAF,
        HAS_SCIPY,
        format_component_stats_line,
        make_component_legend_patches,
        _wood_component_volume,
    )
    from .voxel_npz_io import load_voxel_npz
except ImportError:  # 允許 `python utils/voxel_render.py ...` 直接執行
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.voxel_label_projections import (
        DISPLAY_AIR,
        DISPLAY_COLORS,
        DISPLAY_FRAGMENT,
        DISPLAY_LARGEST,
        DISPLAY_LEAF,
        HAS_SCIPY,
        format_component_stats_line,
        make_component_legend_patches,
        _wood_component_volume,
    )
    from utils.voxel_npz_io import load_voxel_npz

# clay / occupancy 模式配色（與 voxel_orthoslices 的中性亮色對齊）
_CLAY_FILLED = (0.74, 0.71, 0.66)
_OCC_FILLED = (0.85, 0.85, 0.88)
_EDGE_COLOR = (0.12, 0.12, 0.14, 0.55)

_COLOR_MODES = ("component", "clay", "occupancy")


def _coerce_labels(arr: np.ndarray) -> np.ndarray:
    """接受 [Z,Y,X] 離散標籤；若給 4D（含 channel/機率）則沿最小軸 argmax 還原成標籤。"""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        ch_axis = int(np.argmin(arr.shape))  # channel 通常是最小維（如 3）
        return np.argmax(arr, axis=ch_axis).astype(np.uint8)
    raise ValueError(f"預期 3D [Z,Y,X] 標籤（或 4D 含 channel），但收到 shape={arr.shape}")


def _voxel_facecolors(labels: np.ndarray, color_mode: str):
    """
    回傳 (filled[X,Z,Y] bool, facecolors[X,Z,Y,4] float, comp_info)。

    已把 labels 從 [Z,Y,X] 轉置成 [X,Z,Y]，使高度（Y）對到繪圖垂直軸。
    """
    if color_mode not in _COLOR_MODES:
        raise ValueError(f"color_mode 必須是 {_COLOR_MODES} 之一，收到 {color_mode!r}")

    vol = np.transpose(labels, (2, 0, 1))  # [Z,Y,X] -> [X,Z,Y]
    filled = vol != 0
    facecolors = np.zeros(filled.shape + (4,), dtype=np.float32)
    comp_info = None

    if color_mode == "component" and HAS_SCIPY:
        voxel_class, n_comp, llr = _wood_component_volume(labels)  # [Z,Y,X], DISPLAY_* 值
        vclass = np.transpose(voxel_class, (2, 0, 1))  # 對齊 vol
        rgb = DISPLAY_COLORS[vclass]  # [X,Z,Y,3]
        facecolors[..., :3] = rgb
        facecolors[..., 3] = filled.astype(np.float32)
        comp_info = (n_comp, llr)
    elif color_mode == "occupancy":
        facecolors[filled] = (*_OCC_FILLED, 1.0)
    else:  # "clay"（或 scipy 缺席時的 component 退路）
        facecolors[filled] = (*_CLAY_FILLED, 1.0)

    return filled, facecolors, comp_info


def _voxels_on_ax(ax, filled: np.ndarray, facecolors: np.ndarray, *, edges: bool) -> None:
    """在既有的 3D ax 上畫 voxels，跨 matplotlib 版本盡量穩健（shade/lightsource 漸進退化）。"""
    edgecol = _EDGE_COLOR if edges else None
    kwargs = dict(facecolors=facecolors, edgecolor=edgecol, linewidth=0.25, shade=True)
    try:
        from matplotlib.colors import LightSource

        kwargs["lightsource"] = LightSource(azdeg=315, altdeg=45)
    except Exception:
        pass

    # 新 → 舊：lightsource 不支援就拔掉，shade 不支援再拔掉。
    for attempt in range(3):
        try:
            ax.voxels(filled, **kwargs)
            break
        except TypeError:
            if "lightsource" in kwargs:
                kwargs.pop("lightsource")
            elif "shade" in kwargs:
                kwargs.pop("shade")
            else:
                raise


def _equalize_box(ax, shape) -> None:
    """讓三軸等比例（方塊看起來是正方體），跨版本退化。"""
    try:
        ax.set_box_aspect(shape)  # matplotlib >= 3.3
    except Exception:
        try:
            ax.set_aspect("equal")
        except Exception:
            pass


def save_voxel_render(
    labels: np.ndarray,
    out_png: str,
    *,
    color_mode: str = "component",
    elev: float = 22.0,
    azim: float = -60.0,
    edges: bool = True,
    figsize: tuple[float, float] = (5.0, 5.0),
    dpi: int = 140,
    title_suffix: str = "",
    exp_name: str = "",
    show_stats: bool = True,
    legend: bool = True,
    transparent: bool = False,
) -> None:
    """
    將單一 voxel 樣本畫成立體方塊圖（3/4 視角 + 柔光）並存成 PNG。

    Args:
        labels: ``[Z,Y,X]`` 離散標籤（0=air,1=wood,2=leaf）；亦接受 4D（含 channel，將自動 argmax）。
        out_png: 輸出 PNG 路徑。
        color_mode: "component"（連通配色，預設）/ "clay"（單一陶土色）/ "occupancy"（純佔據）。
        elev / azim: 相機仰角 / 方位角；改 ``azim`` 可旋轉椅子朝向。
        edges: 是否描方塊邊（細暗線，立體感較好）。
        figsize / dpi: 輸出尺寸與解析度。
        title_suffix / exp_name: 子標題附加字串 / 圖主標題。
        show_stats: component 模式時，標題附 occupancy / components / largest_part_ratio。
        legend: component 模式時是否附圖例。
        transparent: True 則背景透明（方便貼進論文版面）。
    """
    import matplotlib.pyplot as plt

    labels = _coerce_labels(labels)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    filled, facecolors, comp_info = _voxel_facecolors(labels, color_mode)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _voxels_on_ax(ax, filled, facecolors, edges=edges)
    ax.view_init(elev=elev, azim=azim)
    _equalize_box(ax, filled.shape)
    ax.set_axis_off()

    occ = float((labels != 0).mean())
    stats_line = f"occupancy(non-air): {occ:.3f}"
    if comp_info is not None and show_stats:
        n_comp, llr = comp_info
        stats_line += " | " + format_component_stats_line(n_comp, llr)
    elif color_mode == "component" and not HAS_SCIPY:
        stats_line += " | (scipy missing → clay coloring)"

    suptitle_parts = [p for p in (exp_name, (title_suffix or None)) if p]
    if show_stats:
        suptitle_parts.append(stats_line)
    if suptitle_parts:
        fig.suptitle("\n".join(suptitle_parts), fontsize=10, fontweight="bold", y=0.98)

    if legend and color_mode == "component" and HAS_SCIPY:
        fig.legend(
            handles=make_component_legend_patches(),
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def save_voxel_render_from_npz(npz_path, out_png: str, **kwargs) -> None:
    """便利包裝：從 ``.npz`` 載入 voxel 後算圖（kwargs 同 save_voxel_render）。"""
    labels = load_voxel_npz(npz_path)
    save_voxel_render(labels, out_png, **kwargs)


def save_voxel_render_grid(
    items,
    out_png: str,
    *,
    ncols: int = 4,
    color_mode: str = "component",
    elev: float = 22.0,
    azim: float = -60.0,
    edges: bool = True,
    per: float = 2.4,
    dpi: int = 140,
    titles=None,
    exp_name: str = "",
    transparent: bool = False,
) -> None:
    """
    把多個樣本算成一張格狀「接觸圖（contact sheet）」——論文常見的樣本陣列版面。

    Args:
        items: list，元素可為 ``[Z,Y,X]`` 標籤陣列，或 ``.npz`` 路徑（str/Path，將自動載入）。
        out_png: 輸出 PNG。
        ncols: 欄數（列數自動）。
        titles: 可選，每張子圖標題（長度需 >= 樣本數才會全標）。
        其餘參數同 save_voxel_render。
    """
    import matplotlib.pyplot as plt

    labels_list = []
    for it in items:
        if isinstance(it, (str, Path)):
            labels_list.append(_coerce_labels(load_voxel_npz(it)))
        else:
            labels_list.append(_coerce_labels(it))

    n = len(labels_list)
    if n == 0:
        raise ValueError("items 不能為空")
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(ncols * per, nrows * per))
    for k, lab in enumerate(labels_list):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        filled, facecolors, _ = _voxel_facecolors(lab, color_mode)
        _voxels_on_ax(ax, filled, facecolors, edges=edges)
        ax.view_init(elev=elev, azim=azim)
        _equalize_box(ax, filled.shape)
        ax.set_axis_off()
        if titles is not None and k < len(titles):
            ax.set_title(str(titles[k]), fontsize=8)

    if exp_name:
        fig.suptitle(exp_name, fontsize=11, fontweight="bold", y=1.0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 互動式 HTML viewer（滑鼠自由旋轉，重點是「看清楚斷裂」）
# ---------------------------------------------------------------------------

def _rgb_to_hex(rgb) -> str:
    r, g, b = [max(0, min(255, int(round(float(c) * 255)))) for c in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


# three.js 從 CDN 載入（開檔時需要網路）；用 InstancedMesh 一個顏色一群，
# 故不需要 per-instance color，相容 r128。滑鼠軌道控制為手寫，無額外依賴。
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:__BG__;color:#dcdce2;font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;overflow:hidden}
  #c{display:block;width:100vw;height:100vh}
  #hud{position:fixed;left:12px;top:12px;font-size:13px;line-height:1.55;background:rgba(0,0,0,.40);padding:10px 13px;border-radius:9px;max-width:52vw}
  #hud .t{font-size:14px;font-weight:700;margin-bottom:4px}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px}
  #hud label{display:block;margin-top:7px;cursor:pointer;user-select:none}
  #hint{position:fixed;right:12px;bottom:12px;font-size:12px;color:#99a;background:rgba(0,0,0,.32);padding:6px 9px;border-radius:6px}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <div class="t">__TITLE__</div>
  __STATS__
  <div style="margin-top:7px">
    <span class="sw" style="background:__LARGEST_HEX__"></span>主體（最大連通塊）<br/>
    <span class="sw" style="background:__FRAGMENT_HEX__"></span>斷裂碎塊（與主體不相連）<br/>
    <span class="sw" style="background:__LEAF_HEX__"></span>leaf
  </div>
  <label><input type="checkbox" id="dim"/> 突顯斷裂碎塊（主體半透明）</label>
  <label><input type="checkbox" id="spin"/> 自動旋轉</label>
</div>
<div id="hint">拖曳：旋轉　·　滾輪：縮放　·　右鍵拖曳：平移</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
var DATA = __DATA__;
var N = __N__, AUTOROTATE = __AUTOROTATE__, GAP = __GAP__;
var hud = document.getElementById('hud');
if(!window.THREE){ hud.innerHTML += '<br/><span style="color:#f88">three.js 載入失敗（開檔需要網路）</span>'; }
else{
  var cv=document.getElementById('c');
  var renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio||1);
  var scene=new THREE.Scene();
  var camera=new THREE.PerspectiveCamera(45,1,0.1,8000);
  scene.add(new THREE.AmbientLight(0xffffff,0.62));
  var dl=new THREE.DirectionalLight(0xffffff,0.85); dl.position.set(1,1.4,0.8); scene.add(dl);
  var dl2=new THREE.DirectionalLight(0xffffff,0.25); dl2.position.set(-0.6,-0.4,-0.8); scene.add(dl2);

  var off=(N-1)/2;
  var geo=new THREE.BoxGeometry(1-GAP,1-GAP,1-GAP);
  var groups={};
  function addGroup(key,coords,hex){
    if(!coords||!coords.length) return;
    var mat=new THREE.MeshLambertMaterial({color:new THREE.Color(hex)});
    var mesh=new THREE.InstancedMesh(geo,mat,coords.length);
    var m=new THREE.Matrix4();
    for(var i=0;i<coords.length;i++){
      m.setPosition(coords[i][0]-off,coords[i][1]-off,coords[i][2]-off);
      mesh.setMatrixAt(i,m);
    }
    mesh.instanceMatrix.needsUpdate=true;
    scene.add(mesh);
    groups[key]={mesh:mesh,mat:mat};
  }
  addGroup('largest',DATA.largest,'__LARGEST_HEX__');
  addGroup('fragment',DATA.fragment,'__FRAGMENT_HEX__');
  addGroup('leaf',DATA.leaf,'__LEAF_HEX__');

  var target=new THREE.Vector3(0,0,0);
  var theta=-Math.PI/3, phi=Math.PI/2.5, radius=N*2.1;
  var dragging=false, panning=false, px=0, py=0;
  function place(){
    camera.position.set(
      target.x+radius*Math.sin(phi)*Math.cos(theta),
      target.y+radius*Math.cos(phi),
      target.z+radius*Math.sin(phi)*Math.sin(theta));
    camera.lookAt(target);
  }
  cv.addEventListener('mousedown',function(e){dragging=(e.button===0);panning=(e.button===2);px=e.clientX;py=e.clientY;e.preventDefault();});
  window.addEventListener('mouseup',function(){dragging=panning=false;});
  window.addEventListener('mousemove',function(e){
    var dx=e.clientX-px, dy=e.clientY-py; px=e.clientX; py=e.clientY;
    if(dragging){ theta-=dx*0.01; phi-=dy*0.01; phi=Math.max(0.05,Math.min(Math.PI-0.05,phi)); }
    else if(panning){ var s=radius*0.0016;
      var right=new THREE.Vector3().subVectors(camera.position,target).cross(camera.up).normalize();
      target.addScaledVector(right,-dx*s); target.y+=dy*s; }
  });
  cv.addEventListener('wheel',function(e){ radius*=(e.deltaY>0?1.1:0.9); radius=Math.max(N*0.6,Math.min(N*6,radius)); e.preventDefault(); },{passive:false});
  cv.addEventListener('contextmenu',function(e){e.preventDefault();});

  document.getElementById('dim').addEventListener('change',function(e){
    var on=e.target.checked;
    ['largest','leaf'].forEach(function(k){ if(groups[k]){ groups[k].mat.transparent=on; groups[k].mat.opacity=on?0.16:1.0; groups[k].mat.needsUpdate=true; } });
  });
  var spin=AUTOROTATE; var sb=document.getElementById('spin'); sb.checked=AUTOROTATE;
  sb.addEventListener('change',function(e){spin=e.target.checked;});

  function resize(){ renderer.setSize(window.innerWidth,window.innerHeight); camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); }
  window.addEventListener('resize',resize); resize();
  (function loop(){ requestAnimationFrame(loop); if(spin) theta+=0.0045; place(); renderer.render(scene,camera); })();
}
</script>
</body>
</html>
"""


def _coords_from_mask(mask: np.ndarray):
    """[Z,Y,X] bool → list of [x, y, z]（three.js 世界座標：X 右、Y 上、Z 深）。"""
    zs, ys, xs = np.where(mask)
    return [[int(x), int(y), int(z)] for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist())]


def save_voxel_html(
    labels: np.ndarray,
    out_html: str,
    *,
    title: str = "",
    auto_rotate: bool = False,
    bg: str = "#0e0e12",
    cube_gap: float = 0.06,
) -> None:
    """
    輸出一個自包含的互動式 HTML viewer：瀏覽器開啟即可用滑鼠自由旋轉/縮放/平移。

    重點是「看清楚斷裂」：wood 以 26-connectivity 上色——**最大連通塊為主體色、
    任何與主體不相連的碎塊標紅**，並提供「主體半透明」開關，讓飄在內部/旁邊的紅色
    斷裂碎塊一眼可辨。HUD 同時顯示連通塊數與最大塊占比。

    Args:
        labels: ``[Z,Y,X]`` 離散標籤（0=air,1=wood,2=leaf）；亦接受 4D（自動 argmax）。
        out_html: 輸出 .html 路徑。
        title: HUD 標題（通常用樣本檔名）。
        auto_rotate: 載入後是否自動緩慢旋轉（也可在畫面上勾選）。
        bg: 背景色（CSS）。
        cube_gap: 方塊間隙（0=貼齊；略大可看出格界）。
    """
    labels = _coerce_labels(labels)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    N = int(max(labels.shape))
    occ = float((labels != 0).mean())

    if HAS_SCIPY:
        vclass, n_comp, llr = _wood_component_volume(labels)
        largest = _coords_from_mask(vclass == DISPLAY_LARGEST)
        fragment = _coords_from_mask(vclass == DISPLAY_FRAGMENT)
        leaf = _coords_from_mask(vclass == DISPLAY_LEAF)
        n_broken = max(n_comp - 1, 0)
        stats_html = (
            f"occupancy {occ:.3f}　|　連通塊 <b>{n_comp}</b>"
            f"（斷裂 <b>{n_broken}</b> 塊）<br/>"
            f"最大塊占比 {llr:.3f}　|　碎塊 voxel <b>{len(fragment)}</b>"
        )
    else:
        largest = _coords_from_mask(labels == 1)
        fragment = []
        leaf = _coords_from_mask(labels == 2)
        stats_html = f"occupancy {occ:.3f}<br/>（scipy 缺席：無法分連通塊，全部以主體色顯示）"

    data_json = json.dumps(
        {"largest": largest, "fragment": fragment, "leaf": leaf},
        separators=(",", ":"),
    )

    html = _HTML_TEMPLATE
    repl = {
        "__TITLE__": title or Path(out_html).stem,
        "__STATS__": stats_html,
        "__DATA__": data_json,
        "__N__": str(N),
        "__BG__": bg,
        "__GAP__": f"{float(cube_gap):.3f}",
        "__AUTOROTATE__": "true" if auto_rotate else "false",
        "__LARGEST_HEX__": _rgb_to_hex(DISPLAY_COLORS[DISPLAY_LARGEST]),
        "__FRAGMENT_HEX__": _rgb_to_hex(DISPLAY_COLORS[DISPLAY_FRAGMENT]),
        "__LEAF_HEX__": _rgb_to_hex(DISPLAY_COLORS[DISPLAY_LEAF]),
    }
    for k, v in repl.items():
        html = html.replace(k, v)

    Path(out_html).write_text(html, encoding="utf-8")


def save_voxel_html_from_npz(npz_path, out_html: str, **kwargs) -> None:
    """便利包裝：從 ``.npz`` 載入 voxel 後輸出互動 HTML（kwargs 同 save_voxel_html）。"""
    labels = load_voxel_npz(npz_path)
    if "title" not in kwargs:
        kwargs["title"] = Path(npz_path).stem
    save_voxel_html(labels, out_html, **kwargs)


# ---------------------------------------------------------------------------
# 資料夾批量轉換（遞迴掃 *.npz，輸出鏡像原層級）
# ---------------------------------------------------------------------------

def batch_convert(
    in_dir,
    out_dir,
    *,
    fmt: str = "png",
    recurse: bool = True,
    color_mode: str = "component",
    elev: float = 22.0,
    azim: float = -60.0,
    edges: bool = True,
    dpi: int = 140,
    transparent: bool = False,
    auto_rotate: bool = False,
    verbose: bool = True,
) -> int:
    """
    遞迴掃描 ``in_dir`` 下所有 ``*.npz``，逐一算圖，輸出到 ``out_dir`` 並**鏡像原資料夾層級**。

    例：``in_dir/positive/a.npz`` 在 fmt="png" 下會輸出 ``out_dir/positive/a.png``。

    Args:
        fmt: "png"（靜態圖）/ "html"（互動圖）/ "both"（兩者都出）。
        recurse: True 遞迴所有子資料夾；False 只掃最上層。
        其餘參數：png 用 color_mode/elev/azim/edges/dpi/transparent；html 用 auto_rotate。

    Returns:
        成功輸出的檔案數。
    """
    if fmt not in ("png", "html", "both"):
        raise ValueError(f"fmt 必須是 png/html/both，收到 {fmt!r}")

    in_root = Path(in_dir)
    out_root = Path(out_dir)
    files = sorted(in_root.rglob("*.npz") if recurse else in_root.glob("*.npz"))

    n_ok = 0
    for f in files:
        rel = f.relative_to(in_root)
        targets = []
        if fmt in ("png", "both"):
            targets.append(("png", out_root / rel.with_suffix(".png")))
        if fmt in ("html", "both"):
            targets.append(("html", out_root / rel.with_suffix(".html")))
        for kind, outp in targets:
            outp.parent.mkdir(parents=True, exist_ok=True)
            try:
                if kind == "png":
                    save_voxel_render_from_npz(
                        str(f), str(outp),
                        color_mode=color_mode, elev=elev, azim=azim,
                        edges=edges, dpi=dpi, transparent=transparent,
                        exp_name=f.stem,
                    )
                else:
                    save_voxel_html_from_npz(str(f), str(outp), auto_rotate=auto_rotate)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {f} -> {outp.name}: {e}")
    if verbose:
        print(f"batch: {len(files)} npz from {in_root} → {n_ok} files under {out_root}")
    return n_ok


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(
        description="3D voxel 算圖：靜態 PNG / 格狀接觸圖 / 互動 HTML；支援資料夾批量遞迴轉換。"
    )
    ap.add_argument(
        "inputs", nargs="+",
        help="輸入 .npz（一個=單張；多個=格狀接觸圖）；或單一資料夾 → 批量遞迴轉換。",
    )
    ap.add_argument(
        "output",
        help="輸出檔（.png 靜態圖 / .html 互動圖，由副檔名決定）；"
        "若 inputs 是資料夾，這裡給輸出資料夾（鏡像原層級）。",
    )
    ap.add_argument(
        "--format", choices=["png", "html", "both"], default="png",
        help="資料夾批量模式的輸出格式；單檔模式由副檔名決定、忽略此旗標。",
    )
    ap.add_argument("--no_recurse", action="store_true", help="資料夾模式只轉最上層，不遞迴子資料夾")
    ap.add_argument("--color_mode", default="component", choices=list(_COLOR_MODES))
    ap.add_argument("--elev", type=float, default=22.0, help="相機仰角")
    ap.add_argument("--azim", type=float, default=-60.0, help="相機方位角（旋轉朝向）")
    ap.add_argument("--no_edges", action="store_true", help="不描方塊邊")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--ncols", type=int, default=4, help="格狀模式欄數（多個輸入時生效）")
    ap.add_argument("--transparent", action="store_true", help="背景透明")
    ap.add_argument("--auto_rotate", action="store_true", help="HTML 模式：載入後自動旋轉")
    args = ap.parse_args()

    render_kwargs = dict(
        color_mode=args.color_mode, elev=args.elev, azim=args.azim,
        edges=not args.no_edges, dpi=args.dpi, transparent=args.transparent,
    )

    if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
        # 批量資料夾模式：遞迴找 *.npz，輸出鏡像原層級。
        batch_convert(
            args.inputs[0], args.output,
            fmt=args.format, recurse=not args.no_recurse,
            auto_rotate=args.auto_rotate, **render_kwargs,
        )
    elif args.output.lower().endswith(".html"):
        # 互動式 HTML viewer（滑鼠自由旋轉，連通配色看斷裂）；單一樣本。
        save_voxel_html_from_npz(
            args.inputs[0], args.output,
            title=Path(args.inputs[0]).stem, auto_rotate=args.auto_rotate,
        )
        if len(args.inputs) > 1:
            print(f"[note] HTML 模式只用第一個輸入：{args.inputs[0]}")
        print(f"saved: {args.output}")
    elif len(args.inputs) == 1:
        save_voxel_render_from_npz(
            args.inputs[0], args.output, exp_name=Path(args.inputs[0]).stem, **render_kwargs,
        )
        print(f"saved: {args.output}")
    else:
        save_voxel_render_grid(
            args.inputs, args.output, ncols=args.ncols,
            titles=[Path(p).stem for p in args.inputs], **render_kwargs,
        )
        print(f"saved: {args.output}")
