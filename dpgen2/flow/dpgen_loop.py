import os
import pickle
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import jsonpickle
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OPTemplate,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    if_expression,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.scheduler import (
    ExplorationScheduler,
)
from dpgen2.exploration.selector import (
    ConfSelector,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.superop.block import (
    ConcurrentLearningBlock,
)
from dpgen2.utils import (
    dump_object_to_file,
    load_object_from_file,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

cl_default_optional_parameter = {
    "data_mixed_type": False,
    "finetune_mode": "no",
}


def make_block_optional_parameter(cl_optional_parameter):
    return {
        "data_mixed_type": cl_optional_parameter["data_mixed_type"],
        "finetune_mode": cl_optional_parameter["finetune_mode"],
    }


class SchedulerWrapper(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "exploration_scheduler": BigParameter(ExplorationScheduler),
                "exploration_report": BigParameter(ExplorationReport),
                "trajs": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "converged": bool,
                "exploration_scheduler": BigParameter(ExplorationScheduler),
                "expl_task_grp": BigParameter(ExplorationTaskGroup),
                "conf_selector": ConfSelector,
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        scheduler = ip["exploration_scheduler"]
        report = ip["exploration_report"]
        trajs = ip["trajs"]

        conv, expl_task_grp, selector = scheduler.plan_next_iteration(report, trajs)

        return OPIO(
            {
                "converged": conv,
                "exploration_scheduler": scheduler,
                "expl_task_grp": expl_task_grp,
                "conf_selector": selector,
            }
        )


class MakeBlockId(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "exploration_scheduler": BigParameter(ExplorationScheduler),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "block_id": str,
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        scheduler = ip["exploration_scheduler"]

        stage = scheduler.get_stage()
        iteration = scheduler.get_iteration()

        return OPIO(
            {
                "block_id": f"iter-{iteration:06d}",
            }
        )


class ConcurrentLearningLoop(Steps):
    def __init__(
        self,
        name: str,
        block_op: ConcurrentLearningBlock,
        step_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "conf_selector": InputParameter(),
            "fp_config": InputParameter(),
            "exploration_scheduler": InputParameter(),
            "optional_parameter": InputParameter(type=dict),
            "expl_task_grp": InputParameter(),
        }
        self._input_artifacts = {
            "init_models": InputArtifact(optional=True),
            "init_data": InputArtifact(),
            "iter_data": InputArtifact(),
        }
        self._output_parameters = {
            "exploration_scheduler": OutputParameter(),
        }
        self._output_artifacts = {
            "models": OutputArtifact(),
            "iter_data": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._my_keys = ["block", "scheduler", "id"]
        self._keys = self._my_keys[:1] + block_op.keys + self._my_keys[1:3]
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = "--".join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )

        self = _loop(
            self,
            self.step_keys,
            name,
            block_op,
            step_config=step_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


class ConcurrentLearning(Steps):
    def __init__(
        self,
        name: str,
        block_op: ConcurrentLearningBlock,
        step_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self.loop = ConcurrentLearningLoop(
            name + "-loop",
            block_op,
            step_config=step_config,
            upload_python_packages=upload_python_packages,
        )

        self._input_parameters = {
            "type_map": InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "explore_config": InputParameter(),
            "fp_config": InputParameter(),
            "exploration_scheduler": InputParameter(),
            "optional_parameter": InputParameter(
                type=dict, value=cl_default_optional_parameter
            ),
        }

        self._input_artifacts = {
            "init_models": InputArtifact(optional=True),
            "init_data": InputArtifact(),
            "iter_data": InputArtifact(),
        }
        self._output_parameters = {
            "exploration_scheduler": OutputParameter(),
        }
        self._output_artifacts = {
            "models": OutputArtifact(),
            "iter_data": OutputArtifact(),
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._init_keys = ["scheduler", "id"]
        self.loop_key = "loop"
        self.step_keys = {}
        for ii in self._init_keys:
            self.step_keys[ii] = "--".join(["init", ii])

        self = _dpgen(
            self,
            self.step_keys,
            name,
            self.loop,
            self.loop_key,
            step_config=step_config,
            upload_python_packages=upload_python_packages,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def init_keys(self):
        return self._init_keys

    @property
    def loop_keys(self):
        return [self.loop_key] + self.loop.keys


def _loop(
    steps,
    step_keys,
    name: str,
    block_op: OPTemplate,
    step_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    step_config = deepcopy(step_config)
    step_template_config = step_config.pop("template_config")
    step_executor = init_executor(step_config.pop("executor"))
    block_optional_parameter = make_block_optional_parameter(
        steps.inputs.parameters["optional_parameter"]
    )

    block_common_parameters = {
        "block_id": steps.inputs.parameters["block_id"],
        "type_map": steps.inputs.parameters["type_map"],
        "numb_models": steps.inputs.parameters["numb_models"],
        "template_script": steps.inputs.parameters["template_script"],
        "train_config": steps.inputs.parameters["train_config"],
        "conf_selector": steps.inputs.parameters["conf_selector"],
        "fp_config": steps.inputs.parameters["fp_config"],
        "optional_parameter": block_optional_parameter,
        "explore_config": steps.inputs.parameters["explore_config"],
        "expl_task_grp": steps.inputs.parameters["expl_task_grp"],
    }
    block_step = Step(
        name=name + "-block",
        template=block_op,
        parameters=block_common_parameters,
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
        key=step_keys["block"],
    )
    steps.add(block_step)

    scheduler_step = Step(
        name=name + "-scheduler",
        template=PythonOPTemplate(
            SchedulerWrapper,
            python_packages=upload_python_packages,
            **step_template_config,
        ),
        parameters={
            "exploration_report": block_step.outputs.parameters["exploration_report"],
            "exploration_scheduler": steps.inputs.parameters["exploration_scheduler"],
        },
        artifacts={
            "trajs": block_step.outputs.artifacts["trajs"],
        },
        key=step_keys["scheduler"],
        executor=step_executor,
        **step_config,
    )
    scheduler_step.template.outputs.parameters[
        "exploration_scheduler"
    ].global_name = "exploration_scheduler"
    steps.add(scheduler_step)

    id_step = Step(
        name=name + "-make-block-id",
        template=PythonOPTemplate(
            MakeBlockId,
            python_packages=upload_python_packages,
            **step_template_config,
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters[
                "exploration_scheduler"
            ],
        },
        artifacts={},
        key=step_keys["id"],
        executor=step_executor,
        **step_config,
    )
    steps.add(id_step)

    next_common_parameters = {
        "block_id": id_step.outputs.parameters["block_id"],
        "type_map": steps.inputs.parameters["type_map"],
        "numb_models": steps.inputs.parameters["numb_models"],
        "template_script": steps.inputs.parameters["template_script"],
        "train_config": steps.inputs.parameters["train_config"],
        "explore_config": steps.inputs.parameters["explore_config"],
        "conf_selector": scheduler_step.outputs.parameters["conf_selector"],
        "fp_config": steps.inputs.parameters["fp_config"],
        "exploration_scheduler": scheduler_step.outputs.parameters[
            "exploration_scheduler"
        ],
        "optional_parameter": steps.inputs.parameters["optional_parameter"],
        "expl_task_grp": scheduler_step.outputs.parameters["expl_task_grp"],
    }
    next_step = Step(
        name=name + "-next",
        template=steps,
        parameters=next_common_parameters,
        artifacts={
            "init_models": block_step.outputs.artifacts["models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": block_step.outputs.artifacts["iter_data"],
        },
        when="%s == false" % (scheduler_step.outputs.parameters["converged"]),
    )
    steps.add(next_step)

    steps.outputs.parameters[
        "exploration_scheduler"
    ].value_from_expression = if_expression(
        _if=(scheduler_step.outputs.parameters["converged"] == True),
        _then=scheduler_step.outputs.parameters["exploration_scheduler"],
        _else=next_step.outputs.parameters["exploration_scheduler"],
    )
    steps.outputs.artifacts["models"].from_expression = if_expression(
        _if=(scheduler_step.outputs.parameters["converged"] == True),
        _then=block_step.outputs.artifacts["models"],
        _else=next_step.outputs.artifacts["models"],
    )
    steps.outputs.artifacts["iter_data"].from_expression = if_expression(
        _if=(scheduler_step.outputs.parameters["converged"] == True),
        _then=block_step.outputs.artifacts["iter_data"],
        _else=next_step.outputs.artifacts["iter_data"],
    )

    return steps


def _dpgen(
    steps,
    step_keys,
    name,
    loop_op,
    loop_key,
    step_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    step_config = deepcopy(step_config)
    step_template_config = step_config.pop("template_config")
    step_executor = init_executor(step_config.pop("executor"))

    scheduler_step = Step(
        name=name + "-scheduler",
        template=PythonOPTemplate(
            SchedulerWrapper,
            python_packages=upload_python_packages,
            **step_template_config,
        ),
        parameters={
            "exploration_report": None,
            "exploration_scheduler": steps.inputs.parameters["exploration_scheduler"],
        },
        artifacts={
            "trajs": None,
        },
        key=step_keys["scheduler"],
        executor=step_executor,
        **step_config,
    )
    scheduler_step.template.outputs.parameters[
        "exploration_scheduler"
    ].global_name = "exploration_scheduler"
    steps.add(scheduler_step)

    id_step = Step(
        name=name + "-make-block-id",
        template=PythonOPTemplate(
            MakeBlockId,
            python_packages=upload_python_packages,
            **step_template_config,
        ),
        parameters={
            "exploration_scheduler": scheduler_step.outputs.parameters[
                "exploration_scheduler"
            ],
        },
        artifacts={},
        key=step_keys["id"],
        executor=step_executor,
        **step_config,
    )
    steps.add(id_step)

    common_parameters = {
        "block_id": id_step.outputs.parameters["block_id"],
        "type_map": steps.inputs.parameters["type_map"],
        "numb_models": steps.inputs.parameters["numb_models"],
        "template_script": steps.inputs.parameters["template_script"],
        "train_config": steps.inputs.parameters["train_config"],
        "explore_config": steps.inputs.parameters["explore_config"],
        "conf_selector": scheduler_step.outputs.parameters["conf_selector"],
        "fp_config": steps.inputs.parameters["fp_config"],
        "exploration_scheduler": scheduler_step.outputs.parameters[
            "exploration_scheduler"
        ],
        "optional_parameter": steps.inputs.parameters["optional_parameter"],
        "expl_task_grp": scheduler_step.outputs.parameters["expl_task_grp"],
    }
    loop_step = Step(
        name=name + "-loop",
        template=loop_op,
        parameters=common_parameters,
        artifacts={
            "init_models": steps.inputs.artifacts["init_models"],
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(["%s" % id_step.outputs.parameters["block_id"], loop_key]),
    )
    steps.add(loop_step)

    steps.outputs.parameters[
        "exploration_scheduler"
    ].value_from_parameter = loop_step.outputs.parameters["exploration_scheduler"]
    steps.outputs.artifacts["models"]._from = loop_step.outputs.artifacts["models"]
    steps.outputs.artifacts["iter_data"]._from = loop_step.outputs.artifacts[
        "iter_data"
    ]

    return steps
