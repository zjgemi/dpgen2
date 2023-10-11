from pathlib import Path
from typing import List

import dpdata
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign
from fpop.abacus import AbacusInputs, PrepAbacus, RunAbacus

from ..constants import fp_default_out_data_name


class PrepFpOpAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "type_map": List[str],
                "confs": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        op_in = OPIO({
            "inputs": ip["config"]["inputs"],
            "type_map": ip["type_map"],
            "confs": ip["confs"],
            "prep_image_config": ip["config"].get("prep", {}),
        })
        op = PrepAbacus()
        return op.execute(op_in)


class RunFpOpAbacus(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "labeled_data": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        run_config = ip["config"].get("run", {})
        op_in = OPIO({
            "task_name": ip["task_name"],
            "task_path": ip["task_path"],
            "backward_list": [],
            "run_image_config": run_config,
        })
        op = RunAbacus()
        op_out = op.execute(op_in)
        workdir = op_out["backward_dir"].parent

        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem(str(workdir), fmt="abacus/scf")
        out_name = run_config.get("out", fp_default_out_data_name)
        sys.to("deepmd/npy", workdir / out_name)

        return OPIO({
            "log": workdir / "log",
            "labeled_data": workdir / out_name,
        })
