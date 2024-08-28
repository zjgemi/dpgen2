import json
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
from dpgen2.utils import (
    setup_ele_temp,
)

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
        use_ele_temp: int = 0,
    ):
        self.nopbc = nopbc
        self.use_ele_temp = use_ele_temp

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
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        ntraj = len(trajs)
        ele_temp = None
        if optional_outputs:
            assert ntraj == len(optional_outputs)
            ele_temp = []
            for ii in range(ntraj):
                with open(optional_outputs[ii], "r") as f:
                    data = json.load(f)
                if self.use_ele_temp:
                    ele_temp.append(data["ele_temp"])
            if self.use_ele_temp:
                if self.use_ele_temp == 1:
                    setup_ele_temp(False)
                elif self.use_ele_temp == 2:
                    setup_ele_temp(True)
                else:
                    raise ValueError(
                        "Invalid value for 'use_ele_temp': %s", self.use_ele_temp
                    )

        traj_fmt = "lammps/dump"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            if len(id_selected[ii]) > 0:
                ss = dpdata.System(trajs[ii], fmt=traj_fmt, type_map=type_map)
                ss.nopbc = self.nopbc
                if self.use_ele_temp == 1 and ele_temp:
                    ss.data["fparam"] = np.tile(ele_temp[ii], [len(ss), 1])
                elif self.use_ele_temp == 2 and ele_temp:
                    ss.data["aparam"] = np.tile(
                        ele_temp[ii], [len(ss), ss.get_natoms(), 1]
                    )
                ss = ss.sub_system(id_selected[ii])
                ms.append(ss)
        if conf_filters is not None:
            ms2 = dpdata.MultiSystems(type_map=type_map)
            for s in ms:
                s2 = conf_filters.check(s)
                if len(s2) > 0:
                    ms2.append(s2)
            ms = ms2
        return ms
