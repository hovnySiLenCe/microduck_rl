"""Feetech HL2915 actuator for BAM, identified on the MyMicroDuck pendulum bench.

Importing this module registers ``"hl2915"`` in ``bam.actuators.actuators`` so
``bam.model.load_model(HL2915_PARAMS)`` works in training (bam.mjlab, torch) and
in the CPU rehearsal (bam.mujoco, numpy).  Kept free of torch/mjlab imports so
``scripts/infer_policy.py`` can load it cheaply.

Control law (same parameter names as the identification model in
MyMicroDuck/software/actuator-id/duck_actid/bam_hl2915.py, so its params JSON
loads unchanged):

    duty = clip((q_target - q) * kp * error_gain * error_gain_ratio)
    duty kept inside the firmware current window |I| <= torque_unit * GOAL_TORQUE
    V    = vin * duty;  tau = kt V / R - kt^2 dq / R

The identification model also slews the target with the HLS goal-block speed and
acceleration limits.  At the runtime goal block (acc 255, speed 3000) the fitted
limits sit far above what the servo reaches (~370 rad/s, ~450 rad/s^2 vs a 13 rad/s
no-load speed), so they are omitted here; re-identify if the runtime goal block
changes.  kt (9.3 kg.cm/A) and the 1.5 A current cap come from the HL-2915-C001
datasheet (bench loads stayed below 0.2 Nm): stall torque 0.912 * 1.5 = 1.37 Nm.
"""

from __future__ import annotations

from pathlib import Path

from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Pendulum

NAME = "hl2915"
HL2915_PARAMS = Path(__file__).resolve().parent / "robot" / "actuators" / "hl2915_m6.json"

KP_FW = 32.0          # HLS P coefficient (register 21) read on the bench and on the robot
GOAL_TORQUE = 1000    # runtime goal-block torque (register 44); current cap = unit * this
STS_ERROR_GAIN = 0.166  # BAM's STS3215 duty per (rad * kp); error_gain_ratio scales it


class HL2915Actuator(VoltageControlledActuator):
    def __init__(self, testbench_class=Pendulum):
        super().__init__(testbench_class, vin=12.0, kp=KP_FW, error_gain=STS_ERROR_GAIN,
                         max_pwm=1.0)

    def initialize(self) -> None:
        m = self.model
        m.kt = Parameter(0.912, 0.2, 3.0)
        m.R = Parameter(8.0, 1.0, 30.0)
        m.armature = Parameter(0.005, 1e-5, 0.05)
        m.error_gain_ratio = Parameter(1.0, 0.05, 20.0)
        # Identified goal-block unit factors; only torque_unit is used here.
        m.speed_unit = Parameter(0.07665, 0.02, 0.3)
        m.acc_unit = Parameter(0.1534, 0.005, 2.0)
        m.torque_unit = Parameter(0.0015, 0.0005, 0.05)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def current_cap(self) -> float:
        # Not named max_current: BAM code assigns actuator.max_current = <A or None>.
        return self.model.torque_unit.value * GOAL_TORQUE

    def compute_control(self, q_target, q, dq, dt):
        m = self.model
        duty = (q_target - q) * self.kp * self.error_gain * m.error_gain_ratio.value
        # Firmware current limit as a duty window (BAM VoltageControlledActuator idiom):
        # I = (duty * vin - kt * dq) / R stays within +-I_max when the battery allows.
        center = m.kt.value * dq / self.vin
        span = m.R.value * self.current_cap() / self.vin
        duty = self.backend.clamp(duty, center - span, center + span)
        duty = self.backend.clamp(duty, -self.max_pwm, self.max_pwm)
        return self.vin * duty


def register() -> None:
    from bam.actuators import actuators

    actuators.setdefault(NAME, lambda: HL2915Actuator(Pendulum))


register()
