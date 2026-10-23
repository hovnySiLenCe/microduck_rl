"""Rewrite the robot MJCF masses for MyMicroDuck (Feetech HL2915 servos).

Pollen's Onshape export carries 15 XL330 servos (18 g). MyMicroDuck weighs more:
HL2915 servos (27.6 g each) plus its own prints/electronics. Measurements
(MyMicroDuck tests/records/mass-measurement.template.md, 0.1 g scale, PETG 15%,
NP-F550 battery):

    whole robot 910 g, one HL2915 27.6 g
    trunk 268.6 g, head+neck 338.6 g, left leg 142.8 g, right leg 143.3 g

The two hip-roll servos stayed on the trunk when the legs were weighed, so they are
moved from the trunk to the legs (trunk 213.4 g, legs 170.4 / 170.9 g); every group's
non-servo mass then lands within 0.97-1.06x of pollen's. The 16.7 g the four groups
miss against the whole robot (cables, screws) is spread proportionally to group mass.

Per body: remove the XL330s (18 g boxes 20x34x26 mm at the servo mesh centroids)
from the exported inertial, scale what remains (mass and inertia, CoM unchanged) by
its group factor, add the HL2915s (27.6 g boxes 34x20x23 mm) at the same centroids,
and recombine CoM + full inertia. Bodies not listed (roller wheels) are untouched.

Idempotent per source: it always starts from the pollen values in ORIGINAL_MASS_KG
and refuses to run on a file whose masses no longer match them.

    uv run python src/mjlab_microduck/robot/microduck/apply_measured_masses.py [--check]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
# Roller models are left at pollen's values: their feet are different bodies (wheel
# trucks) that MyMicroDuck has not built or weighed, so a partial rewrite would mix
# measured legs with unmeasured feet.
ROBOT_XMLS = sorted(p for p in HERE.glob("robot_*.xml") if "rollers" not in p.name)

SERVO_MESH = "xl330"
XL330_KG, XL330_BOX = 0.018, (0.020, 0.034, 0.026)
HL2915_KG, HL2915_BOX = 0.0276, (0.034, 0.020, 0.023)

WHOLE_ROBOT_KG = 0.910
GROUPS = {  # measured kg (hip-roll servo moved trunk -> legs), bodies
    "trunk": (0.2686 - 2 * HL2915_KG, ("trunk_base",)),
    "head": (0.3386, ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")),
    "left_leg": (0.1428 + HL2915_KG, ("yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left")),
    "right_leg": (0.1433 + HL2915_KG, ("bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right")),
}
# Pollen's exported masses (robot_walk.xml), the only valid starting point.
ORIGINAL_MASS_KG = {
    "trunk_base": 0.199224, "yaw2roll": 0.0230406, "hip_l": 0.00618934, "upper_leg_left": 0.0482067,
    "leg": 0.0215844, "ankle_left": 0.0300246, "neck": 0.0368414, "neck_pitch": 0.00572,
    "yaw_roll_motion": 0.0486, "jaw_soft": 0.188766, "bearing_roll": 0.0230406, "hip_l_2": 0.00618934,
    "upper_leg_right": 0.0482067, "leg_2": 0.0215844, "ankle_right": 0.0300251,
}
SERVOS_PER_BODY = {"trunk_base": 2, "yaw2roll": 1, "bearing_roll": 1, "upper_leg_left": 2,
                   "upper_leg_right": 2, "leg": 1, "leg_2": 1, "neck": 2, "yaw_roll_motion": 1,
                   "jaw_soft": 2}

INERTIAL_RE = re.compile(r'<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"\s*/>')
BODY_RE = re.compile(r'<body name="([^"]+)"')


def box_inertia(mass: float, dims: tuple[float, float, float]) -> np.ndarray:
    """Isotropic stand-in for a servo of unknown orientation: mean of the box's axes."""
    a, b, c = dims
    return np.eye(3) * mass * (a * a + b * b + c * c) / 18.0


def point_shift(mass: float, r: np.ndarray) -> np.ndarray:
    return mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def to_matrix(full: list[float]) -> np.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = full
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def group_scales() -> dict[str, float]:
    measured = sum(m for m, _ in GROUPS.values())
    scales = {}
    for name, (kg, bodies) in GROUPS.items():
        target = kg * WHOLE_ROBOT_KG / measured
        servos = sum(SERVOS_PER_BODY.get(b, 0) for b in bodies)
        orig = sum(ORIGINAL_MASS_KG[b] for b in bodies)
        scales[name] = (target - servos * HL2915_KG) / (orig - servos * XL330_KG)
    return scales


def servo_centroids(xml: Path) -> dict[str, list[np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, SERVO_MESH)
    out: dict[str, list[np.ndarray]] = {}
    for g in range(model.ngeom):
        # Visual copies only: allcollisions adds a colliding duplicate per servo housing.
        visual = model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        if visual and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g] == mesh:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            # Compiled mesh geoms sit at the mesh centroid (MuJoCo recenters meshes).
            out.setdefault(body, []).append(model.geom_pos[g].copy())
    return out


def rewrite_body(pos, mass, full, servos, scale):
    c, inertia_c = np.array(pos), to_matrix(full)
    about_origin = inertia_c + point_shift(mass, c)
    s_mass = mass - XL330_KG * len(servos)
    s_com = (mass * c - XL330_KG * sum(servos, np.zeros(3))) / s_mass
    s_origin = about_origin - sum((box_inertia(XL330_KG, XL330_BOX) + point_shift(XL330_KG, p)
                                   for p in servos), np.zeros((3, 3)))
    s_inertia = s_origin - point_shift(s_mass, s_com)
    if np.linalg.eigvalsh(s_inertia).min() <= 0:  # export too coarse to subtract: keep shape
        s_inertia = inertia_c * s_mass / mass
    n_mass = scale * s_mass + HL2915_KG * len(servos)
    n_com = (scale * s_mass * s_com + HL2915_KG * sum(servos, np.zeros(3))) / n_mass
    n_inertia = scale * (s_inertia + point_shift(s_mass, s_com - n_com))
    for p in servos:
        n_inertia += box_inertia(HL2915_KG, HL2915_BOX) + point_shift(HL2915_KG, p - n_com)
    assert np.linalg.eigvalsh(n_inertia).min() > 0
    return n_com, n_mass, n_inertia


def fmt(x: float) -> str:
    return f"{x:.6g}"


def apply(xml: Path, check: bool) -> float:
    scale_of = {b: group_scales()[g] for g, (_, bodies) in GROUPS.items() for b in bodies}
    servos = servo_centroids(xml)
    lines, body, total = xml.read_text().splitlines(keepends=True), None, 0.0
    for i, line in enumerate(lines):
        if (m := BODY_RE.search(line)):
            body = m.group(1)
        m = INERTIAL_RE.search(line)
        if not m or body not in scale_of:
            continue
        mass = float(m.group(2))
        if abs(mass - ORIGINAL_MASS_KG[body]) > 1e-6:
            raise SystemExit(f"{xml.name}:{i + 1} {body} mass {mass} is not pollen's "
                             f"{ORIGINAL_MASS_KG[body]}; restore the export first")
        got = servos.get(body, [])
        assert len(got) == SERVOS_PER_BODY.get(body, 0), (xml.name, body, len(got))
        pos = [float(v) for v in m.group(1).split()]
        full = [float(v) for v in m.group(3).split()]
        com, new_mass, inertia = rewrite_body(pos, mass, full, got, scale_of[body])
        total += new_mass
        new = (f'<inertial pos="{" ".join(fmt(v) for v in com)}" mass="{fmt(new_mass)}" '
               f'fullinertia="{" ".join(fmt(v) for v in (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]))}"/>')
        lines[i] = line[:m.start()] + new + line[m.end():]
    if not check:
        xml.write_text("".join(lines))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compute and report, write nothing")
    args = parser.parse_args()
    for name, k in group_scales().items():
        print(f"  {name:10s} non-servo scale {k:.3f}")
    for xml in ROBOT_XMLS:
        print(f"{xml.name:42s} measured bodies -> {apply(xml, args.check) * 1000:.1f} g")


if __name__ == "__main__":
    main()
