import json
import os
import pickle
import shutil
import time
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import jsonpickle
import numpy as np
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
)

try:
    from context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from context import (
    default_host,
    default_image,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    upload_python_packages,
)
from mocked_ops import (
    MockedRunLmp,
    mocked_numb_models,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_task_pattern,
    lmp_traj_name,
    model_name_pattern,
    train_log_name,
    train_script_name,
    train_task_pattern,
)
from dpgen2.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTask,
)
from dpgen2.op.prep_lmp import (
    PrepLmp,
)
from dpgen2.superop.prep_run_lmp import (
    PrepRunLmp,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)


def make_task_group_list(ngrp, ntask_per_grp):
    tgrp = BaseExplorationTaskGroup()
    for ii in range(ngrp):
        for jj in range(ntask_per_grp):
            tt = ExplorationTask()
            tt.add_file(lmp_conf_name, f"group{ii} task{jj} conf").add_file(
                lmp_input_name, f"group{ii} task{jj} input"
            )
            tgrp.add_task(tt)
    return tgrp


def check_lmp_tasks(tcase, ngrp, ntask_per_grp):
    cc = 0
    tdirs = []
    for ii in range(ngrp):
        for jj in range(ntask_per_grp):
            tdir = lmp_task_pattern % cc
            tdirs.append(tdir)
            tcase.assertTrue(Path(tdir).is_dir())
            fconf = Path(tdir) / lmp_conf_name
            finpt = Path(tdir) / lmp_input_name
            tcase.assertTrue(fconf.is_file())
            tcase.assertTrue(finpt.is_file())
            tcase.assertEqual(fconf.read_text(), f"group{ii} task{jj} conf")
            tcase.assertEqual(finpt.read_text(), f"group{ii} task{jj} input")
            cc += 1
    return tdirs


class TestPrepLmp(unittest.TestCase):
    def setUp(self):
        self.ngrp = 2
        self.ntask_per_grp = 3
        self.task_group_list = make_task_group_list(self.ngrp, self.ntask_per_grp)

    def tearDown(self):
        for ii in range(self.ngrp * self.ntask_per_grp):
            work_path = Path(lmp_task_pattern % ii)
            if work_path.is_dir():
                shutil.rmtree(work_path)

    def test(self):
        op = PrepLmp()
        out = op.execute(
            OPIO(
                {
                    "lmp_task_grp": self.task_group_list,
                }
            )
        )
        tdirs = check_lmp_tasks(self, self.ngrp, self.ntask_per_grp)
        tdirs = [str(ii) for ii in tdirs]

        self.assertEqual(tdirs, out["task_names"])
        self.assertEqual(tdirs, [str(ii) for ii in out["task_paths"]])


class TestMockedRunLmp(unittest.TestCase):
    def setUp(self):
        self.ntask = 2
        self.nmodels = 3
        self.task_list = []
        self.model_list = []
        for ii in range(self.ntask):
            work_path = Path(lmp_task_pattern % ii)
            work_path.mkdir(exist_ok=True, parents=True)
            (work_path / lmp_conf_name).write_text(f"conf {ii}")
            (work_path / lmp_input_name).write_text(f"input {ii}")
            self.task_list.append(work_path)
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            model.write_text(f"model {ii}")
            self.model_list.append(model)

    def check_run_lmp_output(
        self,
        task_name: str,
        models: List[Path],
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        for ii in [lmp_conf_name, lmp_input_name] + [ii.name for ii in models]:
            fc.append(Path(ii).read_text())
        self.assertEqual(fc, Path(lmp_log_name).read_text().strip().split("\n"))
        self.assertEqual(
            f"traj of {task_name}", Path(lmp_traj_name).read_text().split("\n")[0]
        )
        self.assertEqual(
            f"model_devi of {task_name}", Path(lmp_model_devi_name).read_text()
        )
        os.chdir(cwd)

    def tearDown(self):
        for ii in range(self.ntask):
            work_path = Path(lmp_task_pattern % ii)
            if work_path.is_dir():
                shutil.rmtree(work_path)
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            if model.is_file():
                os.remove(model)

    def test(self):
        self.task_list_str = [str(ii) for ii in self.task_list]
        self.model_list_str = [str(ii) for ii in self.model_list]
        for ii in range(self.ntask):
            ip = OPIO(
                {
                    "task_name": self.task_list_str[ii],
                    "task_path": self.task_list[ii],
                    "models": self.model_list,
                    "config": {},
                }
            )
            op = MockedRunLmp()
            out = op.execute(ip)
            self.assertEqual(out["log"], Path(f"task.{ii:06d}") / lmp_log_name)
            self.assertEqual(out["traj"], Path(f"task.{ii:06d}") / lmp_traj_name)
            self.assertEqual(
                out["model_devi"], Path(f"task.{ii:06d}") / lmp_model_devi_name
            )
            self.assertTrue(out["log"].is_file())
            self.assertTrue(out["traj"].is_file())
            self.assertTrue(out["model_devi"].is_file())
            self.check_run_lmp_output(self.task_list_str[ii], self.model_list)


# @unittest.skip("temp")
@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestPrepRunLmp(unittest.TestCase):
    def setUp(self):
        self.ngrp = 2
        self.ntask_per_grp = 3
        self.task_group_list = make_task_group_list(self.ngrp, self.ntask_per_grp)
        self.nmodels = mocked_numb_models
        self.model_list = []
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            model.write_text(f"model {ii}")
            self.model_list.append(model)
        self.models = upload_artifact(self.model_list)

    def tearDown(self):
        for ii in range(self.nmodels):
            model = Path(f"model{ii}.pb")
            if model.is_file():
                os.remove(model)
        for ii in range(self.ngrp * self.ntask_per_grp):
            work_path = Path(f"task.{ii:06d}")
            if work_path.is_dir():
                shutil.rmtree(work_path)

    def check_run_lmp_output(
        self,
        task_name: str,
        models: List[Path],
    ):
        cwd = os.getcwd()
        os.chdir(task_name)
        fc = []
        idx = int(task_name.split(".")[1])
        ii = idx // self.ntask_per_grp
        jj = idx - ii * self.ntask_per_grp
        fc.append(f"group{ii} task{jj} conf")
        fc.append(f"group{ii} task{jj} input")
        for ii in [ii.name for ii in models]:
            fc.append((Path("..") / Path(ii)).read_text())
        self.assertEqual(fc, Path(lmp_log_name).read_text().strip().split("\n"))
        self.assertEqual(
            f"traj of {task_name}", Path(lmp_traj_name).read_text().split("\n")[0]
        )
        self.assertEqual(
            f"model_devi of {task_name}", Path(lmp_model_devi_name).read_text()
        )
        os.chdir(cwd)

    def test(self):
        steps = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            MockedRunLmp,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        prep_run_step = Step(
            "prep-run-step",
            template=steps,
            parameters={
                "explore_config": {},
                "expl_task_grp": self.task_group_list,
            },
            artifacts={
                "models": self.models,
            },
        )

        wf = Workflow(name="dp-train", host=default_host)
        wf.add(prep_run_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="prep-run-step")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["model_devis"])
        download_artifact(step.outputs.artifacts["trajs"])
        download_artifact(step.outputs.artifacts["logs"])

        for ii in step.outputs.parameters["task_names"].value:
            self.check_run_lmp_output(ii, self.model_list)
