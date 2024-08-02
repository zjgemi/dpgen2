from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

import dpdata
import numpy as np

from ..deviation import (
    DeviManager,
    DeviManagerStd,
)
from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from dpgen2.exploration.selector import (
        ConfFilters,
    )


class TrajRenderLammps(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
    ):
        self.nopbc = nopbc

    def get_model_devi(
        self,
        files: List[Path],
    ) -> DeviManager:
        ntraj = len(files)

        model_devi = DeviManagerStd()
        for ii in range(ntraj):
            self._load_one_model_devi(files[ii], model_devi)

        return model_devi

    def _load_one_model_devi(self, fname, model_devi):
        dd = np.loadtxt(fname)
        if len(np.shape(dd)) == 1:  # In case model-devi.out is 1-dimensional
            dd = dd.reshape((1, len(dd)))

        # Remove duplicated steps due to a bug of LAMMPS
        if len(set(dd[:, 0])) != len(dd[:, 0]):
            new_dd = []
            steps = []
            for row in dd:
                if row[0] not in steps:
                    new_dd.append(row)
                    steps.append(row[0])
            dd = np.array(new_dd)

        model_devi.add(DeviManager.MAX_DEVI_V, dd[:, 1])
        model_devi.add(DeviManager.MIN_DEVI_V, dd[:, 2])
        model_devi.add(DeviManager.AVG_DEVI_V, dd[:, 3])
        model_devi.add(DeviManager.MAX_DEVI_F, dd[:, 4])
        model_devi.add(DeviManager.MIN_DEVI_F, dd[:, 5])
        model_devi.add(DeviManager.AVG_DEVI_F, dd[:, 6])

    def get_confs(
        self,
        trajs: List[Path],
        id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
    ) -> dpdata.MultiSystems:
        del conf_filters  # by far does not support conf filters
        ntraj = len(trajs)
        traj_fmt = "lammps/dump"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            if len(id_selected[ii]) > 0:
                ss = dpdata.System(trajs[ii], fmt=traj_fmt, type_map=type_map)
                ss.nopbc = self.nopbc
                ss = ss.sub_system(id_selected[ii])
                ms.append(ss)
        return ms
