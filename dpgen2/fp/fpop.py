from pathlib import (
    Path,
)
from typing import (
    List,
)

from dargs import Argument

import dpdata
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)
from fpop.abacus import (
    AbacusInputs,
    PrepAbacus,
    RunAbacus,
)

from ..constants import (
    fp_default_out_data_name,
)


class FpOpAbacusInputs(AbacusInputs):
    @staticmethod
    def args():
        doc_input_file = "A template INPUT file."
        doc_pp_files = 'The pseudopotential files for the elements. '\
            'For example: {"H": "/path/to/H.upf", "O": "/path/to/O.upf"}.'
        doc_element_mass = 'Specify the mass of some elements. '\
            'For example: {"H": 1.0079, "O": 15.9994}.'
        doc_kpt_file = "The KPT file, by default None."
        doc_orb_files = 'The numerical orbital fiels for the elements, '\
            'by default None. '\
            'For example: {"H": "/path/to/H.orb", "O": "/path/to/O.orb"}.'
        doc_deepks_descriptor = "The deepks descriptor file, by default None."
        doc_deepks_model = "The deepks model file, by default None."
        return [
            Argument("input_file", str, optional=False, doc=doc_input_file),
            Argument("pp_files", dict, optional=False, doc=doc_pp_files),
            Argument("element_mass", dict, optional=True, default=None,
                     doc=doc_element_mass),
            Argument("kpt_file", str, optional=True, default=None,
                     doc=doc_kpt_file),
            Argument("orb_files", dict, optional=True, default=None,
                     doc=doc_orb_files),
            Argument("deepks_descriptor", str, optional=True, default=None,
                     doc=doc_deepks_descriptor),
            Argument("deepks_model", str, optional=True, default=None,
                     doc=doc_deepks_model),
        ]


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
        op_in = OPIO(
            {
                "inputs": ip["config"]["inputs"],
                "type_map": ip["type_map"],
                "confs": ip["confs"],
                "prep_image_config": ip["config"].get("prep", {}),
            }
        )
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
        op_in = OPIO(
            {
                "task_name": ip["task_name"],
                "task_path": ip["task_path"],
                "backward_list": [],
                "run_image_config": run_config,
            }
        )
        op = RunAbacus()
        op_out = op.execute(op_in)
        workdir = op_out["backward_dir"].parent

        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem(str(workdir), fmt="abacus/scf")
        out_name = run_config.get("out", fp_default_out_data_name)
        sys.to("deepmd/npy", workdir / out_name)

        return OPIO(
            {
                "log": workdir / "log",
                "labeled_data": workdir / out_name,
            }
        )

    @staticmethod
    def args():
        doc_cmd = "The command of abacus"
        doc_out = "The output dir name of labeled data. "\
            "In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default="abacus",
                     doc=doc_cmd),
            Argument("out", str, optional=True,
                     default=fp_default_out_data_name, doc=doc_out),
        ]
