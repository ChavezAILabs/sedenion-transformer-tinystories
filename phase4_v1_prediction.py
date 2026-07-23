"""phase4_v1_prediction.py — runs the LOGGED falsifiable prediction from
PHASE4_dial_task_design.md SS5.1: train D0p on the ARCHIVED LEAKY v1
generator at high delta. If the leak diagnosis is right, accuracy should
converge toward the suffix band (~0.92-0.97), not the floor (~0.09) and
not ceiling — an empirical shortcut-detection test.

Precondition (discovered 2026-07-19): the model must be at a config that
learns delta=0 at all; the first calibration attempt at smoke dims sat
at chance everywhere, which tests capacity, not shortcuts.

Mechanism: dial_data.gen dispatches through the module-global
_gen_reference at call time, so pointing that at the archived v1
generator reroutes episode generation for the whole training run.
"""
import sys

import dial_data
import dial_selective_recall_prototype_v1_LEAKY as v1

dial_data._gen_reference = v1.gen          # reroute BEFORE training

import phase4_train

if __name__ == "__main__":
    sys.argv = ["phase4_v1_prediction.py",
                "--variant", "D0p", "--task", "dial", "--delta", "12",
                *sys.argv[1:]]
    print("*** v1-LEAKY generator rerouted; prediction test config ***")
    phase4_train.main()
