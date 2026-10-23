"""Swap in MyMicroDuck's modified parts and the raised ankle bore.

MyMicroDuck re-designed some prints (MyMicroDuck cad/native/*_modified, exported to
cad/export/*_modified.STL in each part's original Onshape frame, mm). The ones the
training models care about:

* ankle_left / ankle_right: the ankle joint bore sits ANKLE_BORE_RAISE higher in
  the part (measured on the exported meshes: idler bearing seat +4.38 mm, horn screw
  circle +4.46 mm, both sides). The joint axis is fixed by the unchanged leg, so the
  ankle part, foot, sole, foot collision and foot site all hang that much lower
  relative to the axis: legs are 4.4 mm longer. The idler bearing stays on the axis.
* leg, bottom_head_shell, jaw: same frame, small shape changes (collision meshes in
  walk / groundcontact).

Each modified STL is converted to metres into assets/<name>_modified.stl and the
mesh's file is repointed; mesh names (and so every geom) are unchanged. Roller models
are skipped for the same reason as apply_measured_masses.py. Scene keyframes
(non-roller) are lifted by the same 4.4 mm so their ground clearance is unchanged.
Refuses to run twice (the robot XML check stops it before any scene is touched).

    uv run python src/mjlab_microduck/robot/microduck/apply_modified_parts.py
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
EXPORT = HERE.parents[4] / "MyMicroDuck" / "cad" / "export"  # superproject checkout
ROBOT_XMLS = sorted(p for p in HERE.glob("robot_*.xml") if "rollers" not in p.name)
SCENE_XMLS = sorted(p for p in HERE.glob("scene*.xml") if "rollers" not in p.name)

MODIFIED = ("ankle_left", "ankle_right", "leg", "bottom_head_shell", "jaw")
ANKLE_BORE_RAISE = 0.0044  # m, along the ankle body +y (world up at the zero pose)
# Everything rigidly hung under the ankle bore moves; the idler bearing does not.
ANKLE_MOVED = {"ankle_left": ("ankle_left", "foot_left", "sole_left", "left_foot_collision", "left_foot"),
               "ankle_right": ("ankle_right", "foot_right", "sole_right", "right_foot_collision", "right_foot")}


def convert_stl(name: str) -> None:
    src = EXPORT / f"{name}_modified.STL"
    data = src.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise SystemExit(f"{src} is not a binary STL")
    tris = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])).copy()
    tris["v"] *= 0.001  # mm -> m; normals are unit vectors and unchanged
    header = f"MyMicroDuck {name}_modified, m".encode().ljust(80, b" ")
    (ASSETS / f"{name}_modified.stl").write_bytes(header + data[80:84] + tris.tobytes())


def shift(attr: str, dy: float) -> str:
    x, y, z = (float(v) for v in attr.split())
    return f"{x:.6g} {y + dy:.6g} {z:.6g}"


def apply(xml: Path) -> None:
    text = xml.read_text()
    for name in MODIFIED:
        # Unnamed <mesh file="x.stl"/> takes its name from the file: keep the old name.
        old = f'<mesh file="{name}.stl"'
        if old not in text:
            raise SystemExit(f"{xml.name}: {old} not found (already applied?)")
        text = text.replace(old, f'<mesh name="{name}" file="{name}_modified.stl"')
    lines, body = text.splitlines(keepends=True), None
    for i, line in enumerate(lines):
        if (m := re.search(r'<body name="([^"]+)"', line)):
            body = m.group(1)
        elif "</body>" in line:
            body = None  # ankle bodies are leaves: nothing after them belongs to them
        if body not in ANKLE_MOVED:
            continue
        moved = re.search(r'<(geom|site)\b[^>]*?(?:name|mesh)="([^"]+)"', line)
        if line.lstrip().startswith("<inertial") or (moved and moved.group(2) in ANKLE_MOVED[body]):
            lines[i] = re.sub(r'\bpos="([^"]+)"', lambda p: f'pos="{shift(p.group(1), -ANKLE_BORE_RAISE)}"', line, count=1)
    xml.write_text("".join(lines))


KEY_RE = re.compile(r'(?<!<!-- )(<key name="[^"]+" qpos="\s*)(\S+)(\s+)(\S+)(\s+)(\S+)')


def raise_keyframes(scene: Path) -> int:
    """Lift every active keyframe's base z by the leg lengthening: same ground clearance."""
    text, count = KEY_RE.subn(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}"
                                         f"{float(m.group(6)) + ANKLE_BORE_RAISE:.4f}", scene.read_text())
    scene.write_text(text)
    return count


def main() -> None:
    for name in MODIFIED:
        convert_stl(name)
    for xml in ROBOT_XMLS:
        apply(xml)
        print(f"{xml.name}: modified meshes + ankle bore raise {ANKLE_BORE_RAISE * 1000:.1f} mm")
    for scene in SCENE_XMLS:
        print(f"{scene.name}: {raise_keyframes(scene)} keyframes lifted")


if __name__ == "__main__":
    main()
