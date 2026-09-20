"""批量拉取 Poly Haven 的 PBR 贴图集（CC0，免署名、可商用、可随游戏分发）。

只依赖标准库（urllib + json + hashlib）：无需 pip install、无需 API key。
数据来源：https://api.polyhaven.com （免费 API，用法见 https://polyhaven.com/our-api ）

每个材质产出 3 个文件，落在 `<out>/<材质 id>/`：

    <id>_albedo.jpg       ← Poly Haven `Diffuse`    sRGB 基色（GPU 侧 albedo 层）
    <id>_roughmetal.jpg   ← Poly Haven `arm`        glTF ORM 布局：R=AO / G=Roughness / B=Metalness（GPU 侧 rough-metal 层）
    <id>_height.png       ← Poly Haven `Displacement`  位移高度场（**只在 CPU 侧**供 CSG 位移用，不进 GPU）

格式取舍：albedo / roughmetal 用 jpg（8 bit 足够、体积约为 png 的 1/5）；
height 用 png（保留高度精度，且不进显存，体积代价可接受）。

已存在的文件会用 API 给出的 md5 校验：一致即跳过 ⇒ 可重复运行、可断点续传。

用法::

    python fetch_pbr_textures.py                       # 默认 16 个材质 @1k
    python fetch_pbr_textures.py --res 2k              # 换分辨率
    python fetch_pbr_textures.py stone_wall_04 dirt    # 只拉指定材质（Poly Haven 材质 id）
    python fetch_pbr_textures.py --force               # 忽略本地已有文件，全部重下
    python fetch_pbr_textures.py --out D:/tmp/pbr      # 换输出目录
"""
import argparse
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request

API = "https://api.polyhaven.com"
USER_AGENT = "gate-pbr-fetcher/1.0"

RESOLUTIONS = ("1k", "2k", "4k", "8k")

# 输出文件后缀 → (Poly Haven 通道名, 文件格式)
CHANNELS = {
    "albedo": ("Diffuse", "jpg"),
    "roughmetal": ("arm", "jpg"),
    "height": ("Displacement", "png"),
}

# 缺省拉取清单：16 类表面（全部已核实含 Diffuse / arm / Displacement）
# metal_plate 是唯一的金属 —— 验收 metallic / roughness BRDF 与镜面反射靠它，别删。
MATERIALS = (
    ("square_tiles_03", "方形瓷砖"),
    ("cobblestone_01", "鹅卵石"),
    ("floor_tiles_06", "地面瓷砖"),
    ("laminate_floor_02", "复合地板"),
    ("herringbone_parquet", "人字拼木地板"),
    ("stone_wall_04", "石墙"),
    ("marble_cliff_03", "大理岩崖壁"),
    ("brick_wall_001", "砖墙"),
    ("broken_brick_wall", "破损砖墙"),
    ("painted_plaster_wall", "涂装抹灰墙"),
    ("wooden_floor_01", "木地板"),
    ("rocky_terrain_02", "岩石地面"),
    ("dirt", "泥土"),
    ("lacquered_cherry_wood", "漆面樱桃木"),
    ("coated_pine", "涂装松木"),
    ("metal_plate", "金属板"),
)
DEFAULT_MATERIALS = tuple(m for m, _ in MATERIALS)


def default_out() -> pathlib.Path:
  """默认输出到仓库的 assets/textures/pbr（脚本放在仓库根目录）。"""
  return pathlib.Path(__file__).resolve().parent / "assets" / "textures" / "pbr"


def shown(path: pathlib.Path) -> str:
  """尽量打相对仓库根的短路径；不在仓库内时退回绝对路径。"""
  try:
    return str(path.relative_to(pathlib.Path(__file__).resolve().parent))
  except ValueError:
    return str(path)


def get_json(url: str) -> dict:
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=60) as resp:
    return json.loads(resp.read().decode("utf-8"))


def md5_of(path: pathlib.Path) -> str:
  h = hashlib.md5()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def download(url: str, dest: pathlib.Path) -> int:
  """下载到 `dest.part` 再改名：中断不会留下"存在但损坏"的文件。返回字节数。"""
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_suffix(dest.suffix + ".part")
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  total = 0
  with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as f:
    while True:
      chunk = resp.read(1 << 20)
      if not chunk:
        break
      f.write(chunk)
      total += len(chunk)
  tmp.replace(dest)
  return total


def fetch_material(material: str, res: str, out: pathlib.Path, force: bool) -> tuple[int, int, int, int]:
  """拉单个材质，返回 (下载数, 跳过数, 缺失数, 失败数)。"""
  try:
    files = get_json(f"{API}/files/{material}")
  except urllib.error.HTTPError as e:
    print(f"  [失败] {material}: 查不到该材质 id（HTTP {e.code}）")
    return 0, 0, 0, 1
  except Exception as e:  # noqa: BLE001 - 网络异常不应中断整批
    print(f"  [失败] {material}: 请求出错 {e}")
    return 0, 0, 0, 1

  got = skipped = missing = failed = 0
  for suffix, (channel, fmt) in CHANNELS.items():
    node = files.get(channel, {}).get(res, {}).get(fmt)
    if not node:
      print(f"  [缺失] {material}: 没有 {channel} @{res}/{fmt}")
      missing += 1
      continue

    dest = out / material / f"{material}_{suffix}.{fmt}"
    if dest.exists() and not force and md5_of(dest) == node["md5"]:
      print(f"  [跳过] {shown(dest)}（md5 一致）")
      skipped += 1
      continue

    try:
      size = download(node["url"], dest)
    except Exception as e:  # noqa: BLE001
      print(f"  [失败] {material}/{suffix}: {e}")
      failed += 1
      continue

    if md5_of(dest) != node["md5"]:
      print(f"  [失败] {material}/{suffix}: md5 校验不符（文件已删，请重跑）")
      dest.unlink(missing_ok=True)
      failed += 1
      continue

    print(f"  [完成] {shown(dest)}  {size / 1024:.0f} KB")
    got += 1

  return got, skipped, missing, failed


def main() -> int:
  ap = argparse.ArgumentParser(
    description="批量拉取 Poly Haven 的 PBR 贴图集（CC0）",
    epilog="缺省材质清单：\n  " + "\n  ".join(f"{m:<20}# {n}" for m, n in MATERIALS),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument("materials", nargs="*", help="Poly Haven 材质 id（缺省 = 内置 12 个，见下）")
  ap.add_argument("--res", default="1k", choices=RESOLUTIONS, help="分辨率档（缺省 1k）")
  ap.add_argument("--out", default=str(default_out()), help="输出目录（缺省 assets/textures/pbr）")
  ap.add_argument("--force", action="store_true", help="忽略已存在文件，强制重下")
  args = ap.parse_args()

  materials = args.materials or list(DEFAULT_MATERIALS)
  out = pathlib.Path(args.out)

  print(f"来源：Poly Haven（CC0）  分辨率：{args.res}  共 {len(materials)} 个材质")
  print(f"输出：{out}")
  print()

  totals = [0, 0, 0, 0]
  for i, material in enumerate(materials, 1):
    print(f"[{i}/{len(materials)}] {material}")
    result = fetch_material(material, args.res, out, args.force)
    totals = [a + b for a, b in zip(totals, result)]

  got, skipped, missing, failed = totals
  print()
  print(f"汇总：下载 {got} · 跳过 {skipped} · 缺失 {missing} · 失败 {failed}")
  if failed or missing:
    print("（缺失/失败不影响其余材质；检查材质 id 或换 --res 后重跑）")
  return 0


if __name__ == "__main__":
  sys.exit(main())
