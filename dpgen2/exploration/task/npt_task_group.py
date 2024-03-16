import itertools
import random
from typing import (
    List,
    Optional,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    model_name_pattern,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .lmp import (
    make_lmp_input,
)
from .task import (
    ExplorationTask,
)


class NPTTaskGroup(ConfSamplingTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()
        self.md_set = False

    def set_md(
        self,
        numb_models,
        mass_map,
        temps: List[float],
        press: Optional[List[float]] = None,
        ens: str = "npt",
        dt: float = 0.001,
        nsteps: int = 1000,
        trj_freq: int = 10,
        tau_t: float = 0.1,
        tau_p: float = 0.5,
        pka_e: Optional[float] = None,
        neidelay: Optional[int] = None,
        no_pbc: bool = False,
        use_clusters: bool = False,
        relative_f_epsilon: Optional[float] = None,
        relative_v_epsilon: Optional[float] = None,
        ele_temp_f: Optional[float] = None,
        ele_temp_a: Optional[float] = None,
    ):
        """
        Set MD parameters
        """
        self.graphs = [model_name_pattern % ii for ii in range(numb_models)]
        self.mass_map = mass_map
        self.temps = temps
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p
        self.pka_e = pka_e
        self.neidelay = neidelay
        self.no_pbc = no_pbc
        self.use_clusters = use_clusters
        self.relative_f_epsilon = relative_f_epsilon
        self.relative_v_epsilon = relative_v_epsilon
        self.ele_temp_f = ele_temp_f
        self.ele_temp_a = ele_temp_a
        self.md_set = True

    def make_task(
        self,
    ) -> "NPTTaskGroup":
        """
        Make the LAMMPS task group.

        Returns
        -------
        task_grp: ExplorationTaskGroup
            The returned lammps task group. The number of tasks is nconf*nT*nP.
            nconf is set by `n_sample` parameter of `set_conf`.
            nT and nP are lengths of the `temps` and `press` parameters of `set_md`.

        """
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.md_set:
            raise RuntimeError("MD settings are not set")
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        for cc, tt, pp in itertools.product(confs, self.temps, self.press):  # type: ignore
            self.add_task(self._make_lmp_task(cc, tt, pp))
        return self

    def _make_lmp_task(
        self,
        conf: str,
        tt: float,
        pp: Optional[float],
    ) -> ExplorationTask:
        task = ExplorationTask()
        task.add_file(
            lmp_conf_name,
            conf,
        ).add_file(
            lmp_input_name,
            make_lmp_input(
                lmp_conf_name,
                self.ens,
                self.graphs,
                self.nsteps,
                self.dt,
                self.neidelay,
                self.trj_freq,
                self.mass_map,
                tt,
                self.tau_t,
                pp,
                self.tau_p,
                self.use_clusters,
                self.relative_f_epsilon,
                self.relative_v_epsilon,
                self.pka_e,
                self.ele_temp_f,
                self.ele_temp_a,
                self.no_pbc,
                trj_seperate_files=False,
            ),
        )
        return task
