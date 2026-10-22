"""The HL2915 actuator the MyMicroDuck policies are trained against.

Locks: the identified params load through BAM with the "hl2915" registration, the
datasheet anchors (stall torque / current) hold, the control law runs batched on the
torch backend exactly as bam.mjlab feeds it, and the 1.5 A firmware cap binds.
"""

import numpy as np
import pytest
import torch
from bam.actuator import TorchBackend, VoltageControlledActuator
from bam.model import load_model

from mjlab_microduck.hl2915 import GOAL_TORQUE, HL2915_PARAMS, KP_FW, HL2915Actuator


@pytest.fixture(scope="module")
def model():
    return load_model(str(HL2915_PARAMS))


def test_params_load_as_hl2915_m6(model):
    assert isinstance(model.actuator, HL2915Actuator)
    assert isinstance(model.actuator, VoltageControlledActuator)  # bam.mjlab requirement
    assert model.name == "m6"
    assert model.actuator.kp == KP_FW == 32.0


def test_datasheet_anchors(model):
    kt, cap = model.kt.value, model.actuator.current_cap()
    assert kt == pytest.approx(0.912)              # 9.3 kg.cm/A
    assert cap == pytest.approx(1.5)               # stall current at goal torque 1000
    assert GOAL_TORQUE == 1000
    assert kt * cap == pytest.approx(1.39, rel=0.03)   # 14.2 kg.cm stall torque
    assert 12.0 / kt == pytest.approx(11.5, rel=0.2)   # 110 RPM no-load, +-10% datasheet


def test_torch_batched_control_matches_numpy(model):
    act = model.actuator
    n, j = 4, 14
    gen = torch.Generator().manual_seed(0)
    q_target = torch.rand(n, j, generator=gen) - 0.5
    q = torch.rand(n, j, generator=gen) - 0.5
    dq = 4 * (torch.rand(n, j, generator=gen) - 0.5)
    vin = torch.tensor([[10.8], [11.4], [12.0], [12.6]])
    kp = KP_FW * torch.tensor([[0.9], [1.0], [1.1], [1.0]])

    act.backend, act.vin, act.kp = TorchBackend(), vin, kp
    volts = act.compute_control(q_target, q, dq, 0.005)
    assert volts.shape == (n, j) and torch.isfinite(volts).all()

    from bam.actuator import NumpyBackend
    act.backend, act.vin, act.kp = NumpyBackend(), vin.numpy(), kp.numpy()
    ref = act.compute_control(q_target.numpy(), q.numpy(), dq.numpy(), 0.005)
    np.testing.assert_allclose(volts.numpy(), ref, rtol=1e-5, atol=1e-6)
    act.vin, act.kp = 12.0, KP_FW


def test_current_cap_binds_at_stall(model):
    act = model.actuator
    act.vin, act.kp = 12.0, KP_FW
    volts = act.compute_control(1.0, 0.0, 0.0, 0.005)   # large error, not moving
    current = volts / model.R.value
    assert current == pytest.approx(act.current_cap())
    torque = act.compute_torque(volts, True, 0.0, 0.0)
    assert torque == pytest.approx(model.kt.value * act.current_cap())
    assert torque < 1.45
