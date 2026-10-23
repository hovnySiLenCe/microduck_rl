"""MyMicroDuck's weighed masses (HL2915 servos) are what every non-roller model carries.

See src/mjlab_microduck/robot/microduck/apply_measured_masses.py for the measurements
and how they were applied.
"""

import importlib.util
from pathlib import Path

import mujoco
import pytest

ROBOT_DIR = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "microduck"
spec = importlib.util.spec_from_file_location("apply_measured_masses", ROBOT_DIR / "apply_measured_masses.py")
amm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amm)

MEASURED_XMLS = [p.name for p in amm.ROBOT_XMLS]


def _masses(xml: str) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / xml))
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): float(model.body_mass[i])
            for i in range(1, model.nbody)}


def test_the_training_models_are_rewritten():
    assert {"robot_walk.xml", "robot_groundcontact.xml", "robot_walk_backlash.xml",
            "robot_groundcontact_backlash.xml", "robot_allcollisions.xml"} <= set(MEASURED_XMLS)
    assert not any("rollers" in x for x in MEASURED_XMLS)


@pytest.mark.parametrize("xml", MEASURED_XMLS)
def test_whole_robot_and_groups_match_the_scale(xml):
    masses = _masses(xml)
    assert sum(masses.values()) == pytest.approx(amm.WHOLE_ROBOT_KG, abs=5e-4)
    measured = sum(kg for kg, _ in amm.GROUPS.values())
    for kg, bodies in amm.GROUPS.values():
        assert sum(masses[b] for b in bodies) == pytest.approx(kg * amm.WHOLE_ROBOT_KG / measured, abs=5e-4)


@pytest.mark.parametrize("xml", MEASURED_XMLS)
def test_every_body_heavier_than_pollen_by_its_servos_at_least(xml):
    masses = _masses(xml)
    for body, orig in amm.ORIGINAL_MASS_KG.items():
        servos = amm.SERVOS_PER_BODY.get(body, 0)
        assert masses[body] > orig - 0.01 + servos * (amm.HL2915_KG - amm.XL330_KG) * 0.9


def test_models_agree_body_by_body():
    reference = _masses("robot_walk.xml")
    for xml in MEASURED_XMLS:
        masses = _masses(xml)
        for body, kg in reference.items():
            assert masses[body] == pytest.approx(kg, abs=1e-6), (xml, body)


def test_rewrite_refuses_to_run_twice():
    with pytest.raises(SystemExit, match="not pollen's"):
        amm.apply(ROBOT_DIR / "robot_walk.xml", check=True)
