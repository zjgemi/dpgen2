from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from dflow.python import (
    FatalError,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.selector import (
    ConfSelector,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTaskGroup,
)

from .stage_scheduler import (
    StageScheduler,
)


class ExplorationScheduler:
    """
    The exploration scheduler.

    """

    def __init__(
        self,
    ):
        self.stage_schedulers = []
        self.cur_stage = 0
        self.complete_ = False

    def add_stage_scheduler(
        self,
        stage_scheduler: StageScheduler,
    ):
        """
        Add stage scheduler.

        All added schedulers can be treated as a `list` (order matters). Only one stage is converged, the iteration goes to the next iteration.

        Parameters
        ----------
        stage_scheduler : StageScheduler
            The added stage scheduler

        """
        self.stage_schedulers.append(stage_scheduler)
        self.complete_ = False
        return self

    def get_stage(self):
        """
        Get the index of current stage.

        Stage index increases when the previous stage converges. Usually called after `self.plan_next_iteration`.

        """
        return self.cur_stage

    def get_iteration(self):
        """
        Get the index of the current iteration.

        Iteration index increase when `self.plan_next_iteration` returns valid `expl_task_grp` and `conf_selector` for the next iteration.

        """
        tot_iter = -1
        for idx, ii in enumerate(self.stage_schedulers):
            if ii.complete():
                # the last plan is not used because the stage
                # is found converged
                tot_iter += ii.next_iteration() - 1
            else:
                tot_iter += ii.next_iteration()
        return tot_iter

    def complete(self):
        """
        Tell if all stages are converged.

        """
        return self.complete_

    def force_stage_complete(self):
        """
        Force complete the current stage

        """
        self.stage_schedulers[self.cur_stage].force_complete()
        self.cur_stage += 1
        if self.cur_stage < len(self.stage_schedulers):
            # goes to next stage
            self.plan_next_iteration()
        else:
            # all stages complete
            self.complete_ = True

    def plan_next_iteration(
        self,
        report: Optional[ExplorationReport] = None,
        trajs: Optional[List[Path]] = None,
    ) -> Tuple[bool, Optional[ExplorationTaskGroup], Optional[ConfSelector]]:
        """
        Make the plan for the next DPGEN iteration.

        Parameters
        ----------
        report : ExplorationReport
            The exploration report of this iteration.
        trajs : List[Path]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration.

        Returns
        -------
        complete: bool
            If all the DPGEN stages complete.
        task: ExplorationTaskGroup
            A `ExplorationTaskGroup` defining the exploration of the next iteration. Should be `None` if converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if converged.

        """

        try:
            stg_complete, expl_task_grp, conf_selector = self.stage_schedulers[
                self.cur_stage
            ].plan_next_iteration(
                report,
                trajs,
            )
        except FatalError as e:
            raise FatalError(f"stage {self.cur_stage}: " + str(e))

        if stg_complete:
            self.cur_stage += 1
            if self.cur_stage < len(self.stage_schedulers):
                # goes to next stage
                return self.plan_next_iteration()
            else:
                # all stages complete
                self.complete_ = True
                return (
                    True,
                    None,
                    None,
                )
        else:
            return stg_complete, expl_task_grp, conf_selector

    def get_stage_of_iterations(self):
        """
        Get the stage index and the index in the stage of iterations.

        """
        stages = self.stage_schedulers
        n_stage_iters = []
        for ii in range(self.get_stage() + 1):
            if ii < len(stages) and len(stages[ii].reports) > 0:
                n_stage_iters.append(len(stages[ii].reports))
        cumsum_stage_iters = np.cumsum(n_stage_iters)

        max_iter = self.get_iteration()
        if self.complete() or max_iter == -1:
            max_iter += 1
        stage_idx = []
        idx_in_stage = []
        iter_idx = []
        for ii in range(max_iter):
            idx = np.searchsorted(cumsum_stage_iters, ii + 1)
            stage_idx.append(idx)
            if idx > 0:
                idx_in_stage.append(ii - cumsum_stage_iters[idx - 1])
            else:
                idx_in_stage.append(ii)
            iter_idx.append(ii)
        assert len(stage_idx) == max_iter
        assert len(idx_in_stage) == max_iter
        assert len(iter_idx) == max_iter
        return stage_idx, idx_in_stage, iter_idx

    def get_convergence_ratio(self):
        """
        Get the accurate, candidate and failed ratios of the iterations

        Returns
        -------
        accu    np.ndarray
            The accurate ratio. length of array the same as # iterations.
        cand    np.ndarray
            The candidate ratio. length of array the same as # iterations.
        fail    np.ndarray
            The failed ration. length of array the same as # iterations.
        """
        stages = self.stage_schedulers
        stag_idx, idx_in_stag, iter_idx = self.get_stage_of_iterations()
        accu = []
        cand = []
        fail = []
        for ii in range(np.size(iter_idx)):
            accu.append(stages[stag_idx[ii]].reports[idx_in_stag[ii]].accurate_ratio())
            cand.append(stages[stag_idx[ii]].reports[idx_in_stag[ii]].candidate_ratio())
            fail.append(stages[stag_idx[ii]].reports[idx_in_stag[ii]].failed_ratio())
        return np.array(accu), np.array(cand), np.array(fail)

    def _print_prev_summary(self, prev_stg_idx):
        if prev_stg_idx >= 0:
            yes = "YES" if self.stage_schedulers[prev_stg_idx].converged() else "NO "
            rmx = (
                "YES"
                if self.stage_schedulers[prev_stg_idx].reached_max_iteration()
                else "NO "
            )
            return f"# Stage {prev_stg_idx:4d}  converged {yes}  reached max numb iterations {rmx}"
        else:
            return None

    def print_last_iteration(self, print_header=False):
        stages = self.stage_schedulers

        stage_idx, idx_in_stage, iter_idx = self.get_stage_of_iterations()

        if np.size(iter_idx) == 0:
            return "No finished iteration found\n"

        iidx = np.size(iter_idx) - 1

        ret = []
        if print_header:
            ret.append(
                stages[stage_idx[iidx]].reports[idx_in_stage[iidx]].print_header()
            )
        ret.append(
            stages[stage_idx[iidx]]
            .reports[idx_in_stage[iidx]]
            .print(stage_idx[iidx], idx_in_stage[iidx], iidx)
        )

        if self.complete():
            ret.append(f"# All stages converged")
        return "\n".join(ret + [""])

    def print_convergence(self):
        ret = []
        stages = self.stage_schedulers

        stage_idx, idx_in_stage, iter_idx = self.get_stage_of_iterations()

        if np.size(iter_idx) == 0:
            return "No finished iteration found\n"

        prev_stg_idx = -1
        for iidx in range(np.size(iter_idx)):
            if len(ret) == 0:
                ret.append(
                    stages[stage_idx[iidx]].reports[idx_in_stage[iidx]].print_header()
                )
            if stage_idx[iidx] != prev_stg_idx:
                if prev_stg_idx >= 0:
                    _summary = self._print_prev_summary(prev_stg_idx)
                    assert _summary is not None
                    ret.append(_summary)
                ret.append(f"# Stage {stage_idx[iidx]:4d}  " + "-" * 20)
                prev_stg_idx = stage_idx[iidx]
            ret.append(
                stages[stage_idx[iidx]]
                .reports[idx_in_stage[iidx]]
                .print(stage_idx[iidx], idx_in_stage[iidx], iidx)
            )

        if self.complete():
            if prev_stg_idx >= 0:
                _summary = self._print_prev_summary(prev_stg_idx)
                assert _summary is not None
                ret.append(_summary)
                ret.append(f"# All stages converged")
        return "\n".join(ret + [""])
