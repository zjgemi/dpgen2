"""Prep and Run Gaussian tasks."""
from pathlib import (
    Path,
)
from typing import (
    Any,
    List,
    Optional,
    Tuple,
)

import dpdata
import numpy as np
from dargs import (
    Argument,
    dargs,
)

from dpgen2.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
)

from ..utils import (
    BinaryFileInput,
)
from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)

# global static variables
deepmd_input_path = "one_frame_input"

# global static variables
deepmd_temp_path = "one_frame_temp"

# global static variables
deepmd_teacher_model = "teacher_model.pt"

type_map = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
            "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
            "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
            "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
            "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
            "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
            "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
            "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]


class DeepmdPtInputs:
    @staticmethod
    def args() -> List[Argument]:
        return []

    def __init__(self, **kwargs: Any):
        self.data = kwargs


class PrepDeepmdPt(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        inputs,
    ):
        r"""Define how one Deepmd task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs : str or dict
            This parameter is useless in deepmd.
        """
        conf_frame.to("deepmd/npy", deepmd_input_path)


class RunDeepmdPt(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [deepmd_input_path]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
        self,
        teacher_model_path: BinaryFileInput,
        out: str,
        log: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running Deepmd task
        out : str
            The name of the output data file.

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem
            format.
        log_name: str
            The file name of the log.
        """
        log_name = log
        out_name = out

        teacher_model_path.save_as_file(deepmd_teacher_model)
        self._load_model(deepmd_teacher_model)
        self._infer(out_name)

        return out_name, log_name

    def _load_model(self, model: Path):
        from deepmd_pt.infer.deep_eval import DeepPot
        self.dp = DeepPot(model)

    def _evaluate(self,
                  coord: np.ndarray,
                  cell: Optional[np.ndarray],
                  atype: List[int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coord = coord.reshape([1, -1, 3])
        cell = cell.reshape([1, 3, 3])
        atype = atype.reshape([1, -1])
        return self.dp.eval(coord, cell, atype, infer_batch_size=1)

    def _infer(self, out_name):
        k = dpdata.System(deepmd_input_path, fmt="deepmd/npy")
        assert len(k) == 1
        cell = k.data["cells"][0]
        if k.nopbc:
            cell = None
        coord = k.data["coords"][0]
        ori_atype = k.data["atom_types"]
        anames = k.data["atom_names"]
        atype = np.array([type_map.index(anames[j]) for j in ori_atype])
        e, f, v = self._evaluate(coord, cell, atype)
        k.data["energies"] = e.reshape((1,))
        k.data["forces"] = f.reshape((1, -1, 3))
        if v is not None:
            k.data["virials"] = v.reshape((1, 3, 3))
        s = dpdata.LabeledSystem(data=k.data)
        s.to_deepmd_npy(out_name)

    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_deepmd_teacher_model = (
            "The path of teacher model, which can be loaded by "
            "deepmd.infer.DeepPot"
        )
        doc_deepmd_log = "The log file name of dp"
        doc_deepmd_out = "The output dir name of labeled data. In " \
            "`deepmd/npy` format provided by `dpdata`."
        return [
            Argument(
                "teacher_model_path",
                [str, BinaryFileInput],
                optional=False,
                doc=doc_deepmd_teacher_model,
            ),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_deepmd_out,
            ),
            Argument(
                "log",
                str,
                optional=True,
                default=fp_default_log_name,
                doc=doc_deepmd_log,
            ),
        ]
