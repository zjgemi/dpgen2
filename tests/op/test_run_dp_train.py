import itertools
import json
import os
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    FatalError,
    OPIOSign,
    TransientError,
)
from fake_data_set import (
    fake_multi_sys,
    fake_system,
)
from mock import (
    call,
    patch,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.constants import (
    train_script_name,
    train_task_pattern,
)
from dpgen2.op.run_dp_train import (
    RunDPTrain,
    _get_data_size_of_all_mult_sys,
    _make_train_command,
)

# isort: on


class TestRunDPTrain(unittest.TestCase):
    def setUp(self):
        self.atom_name = "foo"
        self.nframes_0 = [2, 5, 3]
        self.natoms_0 = [4, 3, 4]
        self.nframes_1 = [3, 4, 2]
        self.natoms_1 = [5, 3, 2]
        ms_0 = fake_multi_sys(self.nframes_0, self.natoms_0, self.atom_name)
        ms_1 = fake_multi_sys(self.nframes_1, self.natoms_1, self.atom_name)
        ms_0.to_deepmd_npy("data-0")
        ms_1.to_deepmd_npy("data-1")
        self.iter_data = [Path("data-0"), Path("data-1")]
        self.iter_data_exp = [
            "data-0/foo3",
            "data-0/foo4",
            "data-1/foo2",
            "data-1/foo3",
            "data-1/foo5",
        ]
        ms_0.to_deepmd_npy_mixed("mixed-data-0")
        ms_1.to_deepmd_npy_mixed("mixed-data-1")
        self.mixed_iter_data = [Path("mixed-data-0"), Path("mixed-data-1")]

        self.init_nframs_0 = 3
        self.init_natoms_0 = 5
        self.init_nframs_1 = 4
        self.init_natoms_1 = 2
        ss_0 = fake_system(self.init_nframs_0, self.init_natoms_0, self.atom_name)
        ss_1 = fake_system(self.init_nframs_1, self.init_natoms_1, self.atom_name)
        ss_0.to_deepmd_npy("init/data-0")
        ss_1.to_deepmd_npy("init/data-1")
        self.init_data = [Path("init/data-0"), Path("init/data-1")]
        self.init_data = sorted(list(self.init_data))

        self.init_model = Path("bar.pb")

        self.config = {
            "init_model_policy": "no",
            "init_model_old_ratio": 0.9,
            "init_model_numb_steps": 400000,
            "init_model_start_lr": 1e-4,
            "init_model_start_pref_e": 0.1,
            "init_model_start_pref_f": 100,
            "init_model_start_pref_v": 0.0,
        }
        self.config = RunDPTrain.normalize_config(self.config)

        self.old_data_size = (
            self.init_nframs_0 + self.init_nframs_1 + sum(self.nframes_0)
        )
        self.task_name = "task-000"
        self.task_path = "input-000"

        self.idict_v2 = {
            "training": {
                "training_data": {
                    "systems": [],
                },
                "validation_data": {
                    "systems": [],
                },
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }
        self.expected_odict_v2 = {
            "training": {
                "training_data": {
                    "systems": [
                        "init/data-0",
                        "init/data-1",
                        "data-0/foo3",
                        "data-0/foo4",
                        "data-1/foo2",
                        "data-1/foo3",
                        "data-1/foo5",
                    ],
                    "batch_size": "auto",
                    "auto_prob": "prob_sys_size",
                },
                "disp_file": "lcurve.out",
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }
        self.expected_init_model_odict_v2 = {
            "training": {
                "training_data": {
                    "systems": [
                        "init/data-0",
                        "init/data-1",
                        "data-0/foo3",
                        "data-0/foo4",
                        "data-1/foo2",
                        "data-1/foo3",
                        "data-1/foo5",
                    ],
                    "batch_size": "auto",
                    "auto_prob": "prob_sys_size; 0:4:0.9; 4:7:0.1",
                },
                "disp_file": "lcurve.out",
                "numb_steps": 400000,
            },
            "learning_rate": {
                "start_lr": 1e-4,
            },
            "loss": {
                "start_pref_e": 0.1,
                "start_pref_f": 100,
                "start_pref_v": 0.0,
            },
        }

        self.idict_v1 = {
            "training": {
                "systems": [],
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }
        self.expected_odict_v1 = {
            "training": {
                "systems": [
                    "init/data-0",
                    "init/data-1",
                    "data-0/foo3",
                    "data-0/foo4",
                    "data-1/foo2",
                    "data-1/foo3",
                    "data-1/foo5",
                ],
                "batch_size": "auto",
                "auto_prob_style": "prob_sys_size",
                "disp_file": "lcurve.out",
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }
        self.expected_init_model_odict_v1 = {
            "training": {
                "systems": [
                    "init/data-0",
                    "init/data-1",
                    "data-0/foo3",
                    "data-0/foo4",
                    "data-1/foo2",
                    "data-1/foo3",
                    "data-1/foo5",
                ],
                "batch_size": "auto",
                "auto_prob_style": "prob_sys_size; 0:4:0.9; 4:7:0.1",
                "disp_file": "lcurve.out",
                "stop_batch": 400000,
            },
            "learning_rate": {
                "start_lr": 1e-4,
            },
            "loss": {
                "start_pref_e": 0.1,
                "start_pref_f": 100,
                "start_pref_v": 0.0,
            },
        }

    def tearDown(self):
        for ii in [
            "init",
            "data-0",
            "data-1",
            "mixed-data-0",
            "mixed-data-1",
            self.task_path,
            self.task_name,
        ]:
            if Path(ii).exists():
                shutil.rmtree(str(ii))

    def test_normalize_config(self):
        config = self.config
        self.assertEqual(config["init_model_policy"], "no")
        self.assertAlmostEqual(config["init_model_old_ratio"], 0.9)
        self.assertEqual(config["init_model_numb_steps"], 400000)
        self.assertAlmostEqual(config["init_model_start_lr"], 1e-4)
        self.assertAlmostEqual(config["init_model_start_pref_e"], 0.1)
        self.assertAlmostEqual(config["init_model_start_pref_f"], 100)
        self.assertAlmostEqual(config["init_model_start_pref_v"], 0.0)

    def test_get_size_of_all_mult_sys(self):
        cc = _get_data_size_of_all_mult_sys(self.iter_data)
        self.assertEqual(cc, sum(self.nframes_0) + sum(self.nframes_1))
        cc = _get_data_size_of_all_mult_sys(self.mixed_iter_data, mixed_type=True)
        self.assertEqual(cc, sum(self.nframes_0) + sum(self.nframes_1))
        # read the mixed type systems as if they were standard system,
        # should give the correct estimate of the data size
        cc = _get_data_size_of_all_mult_sys(self.mixed_iter_data, mixed_type=False)
        self.assertEqual(cc, sum(self.nframes_0) + sum(self.nframes_1))

    def test_decide_init_model_no_model(self):
        do_init_model = RunDPTrain.decide_init_model(
            self.config, None, self.init_data, self.iter_data
        )
        self.assertFalse(do_init_model)

    def test_decide_init_model_none_iter_data(self):
        do_init_model = RunDPTrain.decide_init_model(
            self.config, self.init_model, self.init_data, None
        )
        self.assertFalse(do_init_model)

    def test_decide_init_model_no_iter_data(self):
        do_init_model = RunDPTrain.decide_init_model(
            self.config, self.init_model, self.init_data, []
        )
        self.assertFalse(do_init_model)

    def test_decide_init_model_config_no(self):
        config = self.config.copy()
        config["init_model_policy"] = "no"
        do_init_model = RunDPTrain.decide_init_model(
            config, self.init_model, self.init_data, self.iter_data
        )
        self.assertFalse(do_init_model)

    def test_decide_init_model_config_yes(self):
        config = self.config.copy()
        config["init_model_policy"] = "yes"
        do_init_model = RunDPTrain.decide_init_model(
            config, self.init_model, self.init_data, self.iter_data
        )
        self.assertTrue(do_init_model)

    def test_decide_init_model_config_larger_than_no(self):
        config = self.config.copy()
        config["init_model_policy"] = f"old_data_larger_than:{self.old_data_size}"
        do_init_model = RunDPTrain.decide_init_model(
            config, self.init_model, self.init_data, self.iter_data
        )
        self.assertFalse(do_init_model)

    def test_decide_init_model_config_larger_than_yes(self):
        config = self.config.copy()
        config["init_model_policy"] = f"old_data_larger_than:{self.old_data_size-1}"
        do_init_model = RunDPTrain.decide_init_model(
            config, self.init_model, self.init_data, self.iter_data
        )
        self.assertTrue(do_init_model)

    def test_update_input_dict_v1_init_model(self):
        odict = RunDPTrain.write_data_to_input_script(
            self.idict_v1,
            self.config,
            self.init_data,
            self.iter_data_exp,
            auto_prob_str="prob_sys_size; 0:4:0.9; 4:7:0.1",
            major_version="1",
        )
        config = self.config.copy()
        config["init_model_policy"] = "yes"
        odict = RunDPTrain.write_other_to_input_script(
            odict, config, True, major_version="1"
        )
        self.assertDictEqual(odict, self.expected_init_model_odict_v1)

    def test_update_input_dict_v1(self):
        odict = RunDPTrain.write_data_to_input_script(
            self.idict_v1,
            self.config,
            self.init_data,
            self.iter_data_exp,
            auto_prob_str="prob_sys_size",
            major_version="1",
        )
        config = self.config.copy()
        config["init_model_policy"] = "no"
        odict = RunDPTrain.write_other_to_input_script(
            odict, config, False, major_version="1"
        )
        self.assertDictEqual(odict, self.expected_odict_v1)

    def test_update_input_dict_v2_init_model(self):
        idict = self.idict_v2
        odict = RunDPTrain.write_data_to_input_script(
            idict,
            self.config,
            self.init_data,
            self.iter_data_exp,
            auto_prob_str="prob_sys_size; 0:4:0.9; 4:7:0.1",
            major_version="2",
        )
        config = self.config.copy()
        config["init_model_policy"] = "yes"
        odict = RunDPTrain.write_other_to_input_script(
            odict, config, True, major_version="2"
        )
        self.assertDictEqual(odict, self.expected_init_model_odict_v2)

    def test_update_input_dict_v2(self):
        idict = self.idict_v2
        odict = RunDPTrain.write_data_to_input_script(
            idict,
            self.config,
            self.init_data,
            self.iter_data_exp,
            auto_prob_str="prob_sys_size",
            major_version="2",
        )
        config = self.config.copy()
        config["init_model_policy"] = "no"
        odict = RunDPTrain.write_other_to_input_script(
            odict, config, False, major_version="2"
        )
        self.assertDictEqual(odict, self.expected_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v1(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]

        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v1, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [Path(ii) for ii in self.iter_data],
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(["dp", "train", train_script_name]),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v1)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]

        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [Path(ii) for ii in self.iter_data],
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(["dp", "train", train_script_name]),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_init_model(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]

        config = self.config.copy()
        config["init_model_policy"] = "yes"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [Path(ii) for ii in self.iter_data],
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(
                [
                    "dp",
                    "train",
                    "--init-frz-model",
                    str(self.init_model),
                    train_script_name,
                ]
            ),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_init_model_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_train_error(self, mocked_run):
        mocked_run.side_effect = [(1, "", "foo\n"), (0, "bar\n", "")]

        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        with self.assertRaises(FatalError) as ee:
            out = ptrain.execute(
                OPIO(
                    {
                        "config": config,
                        "task_name": task_name,
                        "task_path": Path(task_path),
                        "init_model": Path(self.init_model),
                        "init_data": [Path(ii) for ii in self.init_data],
                        "iter_data": [Path(ii) for ii in self.iter_data],
                    }
                )
            )

        calls = [
            call(["dp", "train", train_script_name]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        with open(work_dir / train_script_name) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_freeze_error(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (1, "", "bar\n")]

        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        with self.assertRaises(FatalError) as ee:
            out = ptrain.execute(
                OPIO(
                    {
                        "config": config,
                        "task_name": task_name,
                        "task_path": Path(task_path),
                        "init_model": Path(self.init_model),
                        "init_data": [Path(ii) for ii in self.init_data],
                        "iter_data": [Path(ii) for ii in self.iter_data],
                    }
                )
            )

        calls = [
            call(["dp", "train", train_script_name]),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        with open(work_dir / train_script_name) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_finetune_finetune(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]
        config = self.config.copy()
        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [Path(ii) for ii in self.iter_data],
                    "optional_parameter": {
                        "mixed_type": False,
                        "finetune_mode": "finetune",
                    },
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(
                [
                    "dp",
                    "train",
                    train_script_name,
                    "--finetune",
                    str(self.init_model),
                ]
            ),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_finetune_train_init(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]

        config = self.config.copy()
        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [Path(ii) for ii in self.iter_data],
                    "optional_parameter": {
                        "mixed_type": False,
                        "finetune_mode": "train-init",
                    },
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(
                [
                    "dp",
                    "train",
                    "--init-frz-model",
                    str(self.init_model),
                    train_script_name,
                ]
            ),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)


class TestRunDPTrainNullIterData(unittest.TestCase):
    def setUp(self):
        self.atom_name = "foo"
        self.init_nframs_0 = 3
        self.init_natoms_0 = 5
        self.init_nframs_1 = 4
        self.init_natoms_1 = 2
        ss_0 = fake_system(self.init_nframs_0, self.init_natoms_0, self.atom_name)
        ss_1 = fake_system(self.init_nframs_1, self.init_natoms_1, self.atom_name)
        ss_0.to_deepmd_npy("init/data-0")
        ss_1.to_deepmd_npy("init/data-1")
        self.init_data = [Path("init/data-0"), Path("init/data-1")]
        self.init_data = sorted(list(self.init_data))

        self.init_model = Path("bar.pb")

        self.config = {
            "init_model_policy": "no",
            "init_model_old_ratio": 0.9,
            "init_model_numb_steps": 400000,
            "init_model_start_lr": 1e-4,
            "init_model_start_pref_e": 0.1,
            "init_model_start_pref_f": 100,
            "init_model_start_pref_v": 0.0,
        }
        self.config = RunDPTrain.normalize_config(self.config)

        self.task_name = "task-000"
        self.task_path = "input-000"

        self.idict_v2 = {
            "training": {
                "training_data": {
                    "systems": [],
                },
                "validation_data": {
                    "systems": [],
                },
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }
        self.expected_odict_v2 = {
            "training": {
                "training_data": {
                    "systems": ["init/data-0", "init/data-1"],
                    "batch_size": "auto",
                    "auto_prob": "prob_sys_size",
                },
                "disp_file": "lcurve.out",
            },
            "learning_rate": {
                "start_lr": 1.0,
            },
            "loss": {
                "start_pref_e": 1.0,
                "start_pref_f": 1.0,
                "start_pref_v": 1.0,
            },
        }

    def tearDown(self):
        for ii in ["init", self.task_path, self.task_name, "foo"]:
            if Path(ii).exists():
                shutil.rmtree(str(ii))

    def test_update_input_dict_v2_empty_list(self):
        idict = self.idict_v2
        odict = RunDPTrain.write_data_to_input_script(
            idict,
            self.config,
            self.init_data,
            [],
            auto_prob_str="prob_sys_size",
            major_version="2",
        )
        config = self.config.copy()
        config["init_model_policy"] = "no"
        odict = RunDPTrain.write_other_to_input_script(
            odict, config, False, major_version="2"
        )
        self.assertDictEqual(odict, self.expected_odict_v2)

    def test_exec_v2_empty_list(self):
        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)

        self.init_model = self.init_model.absolute()
        self.init_model.write_text("this is init model")

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [],
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], self.init_model)
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            f"We have init model {self.init_model} and "
            f"no iteration training data. "
            f"The training is skipped.\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)
        self.assertEqual(Path(out["model"]).read_text(), "this is init model")

        os.remove(self.init_model)

    @patch("dpgen2.op.run_dp_train.run_command")
    def test_exec_v2_empty_dir(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", ""), (0, "bar\n", "")]

        config = self.config.copy()
        config["init_model_policy"] = "no"

        task_path = self.task_path
        Path(task_path).mkdir(exist_ok=True)
        with open(Path(task_path) / train_script_name, "w") as fp:
            json.dump(self.idict_v2, fp, indent=4)
        task_name = self.task_name
        work_dir = Path(task_name)
        empty_data = Path("foo")
        empty_data.mkdir(exist_ok=True)

        ptrain = RunDPTrain()
        out = ptrain.execute(
            OPIO(
                {
                    "config": config,
                    "task_name": task_name,
                    "task_path": Path(task_path),
                    "init_model": Path(self.init_model),
                    "init_data": [Path(ii) for ii in self.init_data],
                    "iter_data": [empty_data],
                }
            )
        )
        self.assertEqual(out["script"], work_dir / train_script_name)
        self.assertEqual(out["model"], work_dir / "frozen_model.pb")
        self.assertEqual(out["lcurve"], work_dir / "lcurve.out")
        self.assertEqual(out["log"], work_dir / "train.log")

        calls = [
            call(["dp", "train", train_script_name]),
            call(["dp", "freeze", "-o", "frozen_model.pb"]),
        ]
        mocked_run.assert_has_calls(calls)

        self.assertTrue(work_dir.is_dir())
        self.assertTrue(out["log"].is_file())
        self.assertEqual(
            out["log"].read_text(),
            "#=================== train std out ===================\n"
            "foo\n"
            "#=================== train std err ===================\n"
            "#=================== freeze std out ===================\n"
            "bar\n"
            "#=================== freeze std err ===================\n",
        )
        with open(out["script"]) as fp:
            jdata = json.load(fp)
            self.assertDictEqual(jdata, self.expected_odict_v2)
