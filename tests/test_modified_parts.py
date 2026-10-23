"""MyMicroDuck's modified parts and 4.4 mm ankle bore raise are in every non-roller model.

See src/mjlab_microduck/robot/microduck/apply_modified_parts.py.
"""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
spec = importlib.util.spec_from_file_location("apply_modified_parts", ROBOT_DIR / "apply_modified_parts.py")
amp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amp)

POLLEN_AXIS_TO_SOLE = 0.02503   # m, ankle joint axis to sole bottom at the zero pose
POLLEN_STAND_CLEARANCE = 0.00282  # m, sole bottom above the floor in scene STAND keyframe


def _sole_bottom_and_axis(model, data, side):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mid = model.geom_dataid[g]
    verts = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
    return world[:, 2].min(), data.xpos[body][2]


@pytest.mark.parametrize("xml", [p.name for p in amp.ROBOT_XMLS])
def test_legs_are_longer_by_the_bore_raise(xml):
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        sole, axis = _sole_bottom_and_axis(model, data, side)
        assert axis - sole == pytest.approx(POLLEN_AXIS_TO_SOLE + amp.ANKLE_BORE_RAISE, abs=2e-4)


@pytest.mark.parametrize("xml", [p.name for p in amp.ROBOT_XMLS])
def test_modified_meshes_are_used(xml):
    text = (ROBOT_DIR / xml).read_text()
    for name in amp.MODIFIED:
        assert f'<mesh name="{name}" file="{name}_modified.stl"' in text
        assert (ROBOT_DIR / "assets" / f"{name}_modified.stl").is_file()


@pytest.mark.parametrize("scene", ["scene.xml", "scene_walk.xml", "scene_backlash.xml", "scene_walk_backlash.xml"])
def test_stand_keyframe_keeps_pollen_ground_clearance(scene):
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / scene))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")]
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        sole, _ = _sole_bottom_and_axis(model, data, side)
        assert sole == pytest.approx(POLLEN_STAND_CLEARANCE, abs=2e-4)


def test_rewrite_refuses_to_run_twice():
    with pytest.raises(SystemExit, match="already applied"):
        amp.apply(ROBOT_DIR / "robot_walk.xml")


def test_modified_ankle_bore_is_on_the_joint_axis():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "robot_walk.xml"))
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "ankle_left")
    g = next(i for i in range(model.ngeom) if model.geom_dataid[i] == mid
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    verts = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    body = (verts @ rot.reshape(3, 3).T + model.geom_pos[g]) * 1000  # ankle body frame, mm
    seat = body[(body[:, 2] > -40) & (body[:, 2] < -20)]          # idler bearing seat end
    on_axis = np.sum(np.abs(np.hypot(seat[:, 0], seat[:, 1]) - 7.25) < 0.3)
    raised = np.sum(np.abs(np.hypot(seat[:, 0], seat[:, 1] - 4.4) - 7.25) < 0.3)
    assert on_axis > 3 * max(raised, 1)
