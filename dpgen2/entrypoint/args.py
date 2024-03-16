import textwrap
from typing import (
    List,
)

import dargs
from dargs import (
    Argument,
    Variant,
)

import dpgen2
from dpgen2.conf import (
    conf_styles,
)
from dpgen2.constants import (
    default_image,
)
from dpgen2.exploration.report import (
    conv_styles,
)
from dpgen2.fp import (
    fp_styles,
)
from dpgen2.op.run_dp_train import (
    RunDPTrain,
)
from dpgen2.op.run_lmp import (
    RunLmp,
)
from dpgen2.utils import (
    normalize_step_dict,
    step_conf_args,
)


def make_link(content, ref_key):
    raw_anchor = dargs.dargs.RAW_ANCHOR
    return (
        f"`{content} <{ref_key}_>`_" if not raw_anchor else f"`{content} <#{ref_key}>`_"
    )


def dp_dist_train_args():
    doc_config = "Configuration of training"
    doc_template_script = "File names of the template training script. It can be a `List[str]`, the length of which is the same as `numb_models`. Each template script in the list is used to train a model. Can be a `str`, the models share the same template training script. "
    dock_student_model_path = "The path of student model"

    return [
        Argument(
            "config",
            dict,
            RunDPTrain.training_args(),
            optional=True,
            default=RunDPTrain.normalize_config({}),
            doc=doc_config,
        ),
        Argument(
            "template_script", [List[str], str], optional=False, doc=doc_template_script
        ),
        Argument("student_model_path", str, optional=True, doc=dock_student_model_path),
    ]


def dp_train_args():
    doc_numb_models = "Number of models trained for evaluating the model deviation"
    doc_config = "Configuration of training"
    doc_template_script = "File names of the template training script. It can be a `List[str]`, the length of which is the same as `numb_models`. Each template script in the list is used to train a model. Can be a `str`, the models share the same template training script. "
    doc_init_models_paths = "the paths to initial models"

    return [
        Argument(
            "config",
            dict,
            RunDPTrain.training_args(),
            optional=True,
            default=RunDPTrain.normalize_config({}),
            doc=doc_numb_models,
        ),
        Argument("numb_models", int, optional=True, default=4, doc=doc_numb_models),
        Argument(
            "template_script", [List[str], str], optional=False, doc=doc_template_script
        ),
        Argument(
            "init_models_paths",
            List[str],
            optional=True,
            default=None,
            doc=doc_init_models_paths,
            alias=["training_iter0_model_path"],
        ),
    ]


def variant_train():
    doc = "the type of the training"
    return Variant(
        "type",
        [
            Argument("dp", dict, dp_train_args()),
            Argument("dp-dist", dict, dp_dist_train_args()),
        ],
        doc=doc,
    )


def variant_conv():
    doc = "the type of the condidate selection and convergence check method."
    var_list = []
    for kk in conv_styles.keys():
        var_list.append(
            Argument(
                kk,
                dict,
                conv_styles[kk].args(),
                doc=conv_styles[kk].doc(),
            )
        )
    return Variant(
        "type",
        var_list,
        doc=doc,
    )


def variant_conf():
    doc = "the type of the initial configuration generator."
    var_list = []
    for kk in conf_styles.keys():
        var_list.append(
            Argument(
                kk,
                dict,
                conf_styles[kk].args(),
                doc=conf_styles[kk].doc(),
            )
        )
    return Variant(
        "type",
        var_list,
        doc=doc,
    )


def lmp_args():
    doc_config = "Configuration of lmp exploration"
    doc_max_numb_iter = "Maximum number of iterations per stage"
    doc_fatal_at_max = (
        "Fatal when the number of iteration per stage reaches the `max_numb_iter`"
    )
    doc_output_nopbc = "Remove pbc of the output configurations"
    doc_convergence = "The method of convergence check."
    doc_configuration = "A list of initial configurations."
    doc_stages = (
        "The definition of exploration stages of type `List[List[ExplorationTaskGroup]`. "
        "The outer list provides the enumeration of the exploration stages. "
        "Then each stage is defined by a list of exploration task groups. "
        "Each task group is described in :ref:`the task group definition<task_group_sec>` "
    )

    return [
        Argument(
            "config",
            dict,
            RunLmp.lmp_args(),
            optional=True,
            default=RunLmp.normalize_config({}),
            doc=doc_config,
        ),
        Argument(
            "max_numb_iter", int, optional=True, default=10, doc=doc_max_numb_iter
        ),
        Argument(
            "fatal_at_max", bool, optional=True, default=True, doc=doc_fatal_at_max
        ),
        Argument(
            "output_nopbc", bool, optional=True, default=False, doc=doc_output_nopbc
        ),
        Argument(
            "convergence",
            dict,
            [],
            [variant_conv()],
            optional=False,
            doc=doc_convergence,
        ),
        Argument(
            "configurations",
            list,
            [],
            [variant_conf()],
            optional=False,
            repeat=True,
            doc=doc_configuration,
            alias=["configuration"],
        ),
        Argument("stages", List[List[dict]], optional=False, doc=doc_stages),
    ]


def variant_explore():
    # TODO: add calypso_args
    doc = "The type of the exploration"
    doc_lmp = "The exploration by LAMMPS simulations"
    doc_calypso = "The exploration by CALYPSO structure prediction"
    return Variant(
        "type",
        [
            Argument("lmp", dict, lmp_args(), doc=doc_lmp),
            Argument("calypso", dict, lmp_args(), doc=doc_calypso),
        ],
        doc=doc,
    )


def fp_args(inputs, run):
    doc_inputs_config = "Configuration for preparing vasp inputs"
    doc_run_config = "Configuration for running vasp tasks"
    doc_task_max = "Maximum number of vasp tasks for each iteration"

    return [
        Argument(
            "inputs_config",
            dict,
            inputs.args(),
            optional=False,
            doc=doc_inputs_config,
        ),
        Argument(
            "run_config",
            dict,
            run.args(),
            optional=False,
            doc=doc_run_config,
        ),
        Argument("task_max", int, optional=True, default=10, doc=doc_task_max),
    ]


def variant_fp():
    doc = "the type of the fp"

    fp_list = []
    for kk in fp_styles.keys():
        fp_list.append(
            Argument(
                kk,
                dict,
                fp_args(fp_styles[kk]["inputs"], fp_styles[kk]["run"]),
            )
        )

    return Variant("type", fp_list, doc=doc)


def input_args():
    doc_type_map = 'The type map. e.g. ["Al", "Mg"]. Al and Mg will have type 0 and 1, respectively.'
    doc_mass_map = "The mass map. e.g. [27., 24.]. Al and Mg will be set with mass 27. and 24. amu, respectively."
    doc_mixed_type = "Use `deepmd/npy/mixed` format for storing training data."
    doc_do_finetune = (
        "Finetune the pretrained model before the first iteration. If it is set to True, then an additional step, finetune-step, "
        'which is based on a branch of "PrepRunDPTrain," will be added before the dpgen_step. In the '
        'finetune-step, the internal flag finetune_mode is set to "finetune," which means SuperOP "PrepRunDPTrain" '
        'is now used as the "Finetune." In this step, we finetune the pretrained model in the train step and modify '
        'the template after training. After that, in the normal dpgen-step, the flag do_finetune is set as "train-init," '
        'which means we use `--init-frz-model` to train based on models from the previous iteration. The "do_finetune" flag '
        'is set to False by default, while the internal flag finetune_mode is set to "no," which means anything related '
        "to finetuning will not be done."
    )
    doc_do_finetune = textwrap.dedent(doc_do_finetune)
    doc_init_data_prefix = "The prefix of initial data systems"
    doc_init_sys = "The inital data systems"
    doc_multitask = "Do multitask training"
    doc_head = "Head to use in the multitask training"
    doc_multi_init_data = (
        "The inital data for multitask, it should be a dict, whose keys are task names and each value is a dict"
        "containing fields `prefix` and `sys` for initial data of each task"
    )
    doc_valid_data_prefix = "The prefix of validation data systems"
    doc_valid_sys = "The validation data systems"

    return [
        Argument("type_map", List[str], optional=False, doc=doc_type_map),
        Argument("mass_map", List[float], optional=False, doc=doc_mass_map),
        Argument(
            "init_data_prefix",
            str,
            optional=True,
            default=None,
            doc=doc_init_data_prefix,
        ),
        Argument("mixed_type", bool, optional=True, default=False, doc=doc_mixed_type),
        Argument(
            "do_finetune", bool, optional=True, default=False, doc=doc_do_finetune
        ),
        Argument(
            "init_data_sys",
            [List[str], str],
            optional=True,
            default=None,
            doc=doc_init_sys,
        ),
        Argument(
            "multitask",
            bool,
            optional=True,
            default=False,
            doc=doc_multitask,
        ),
        Argument(
            "head",
            str,
            optional=True,
            default=None,
            doc=doc_head,
        ),
        Argument(
            "multi_init_data",
            dict,
            optional=True,
            default=None,
            doc=doc_multi_init_data,
        ),
        Argument(
            "valid_data_prefix",
            str,
            optional=True,
            default=None,
            doc=doc_valid_data_prefix,
        ),
        Argument(
            "valid_data_sys",
            [List[str], str],
            optional=True,
            default=None,
            doc=doc_valid_sys,
        ),
    ]


def dflow_conf_args():
    doc_dflow_config = "The configuration passed to dflow"
    doc_dflow_s3_config = "The S3 configuration passed to dflow"

    return [
        Argument(
            "dflow_config", dict, optional=True, default=None, doc=doc_dflow_config
        ),
        Argument(
            "dflow_s3_config",
            dict,
            optional=True,
            default=None,
            doc=doc_dflow_s3_config,
        ),
    ]


def bohrium_conf_args():
    doc_username = "The username of the Bohrium platform"
    doc_password = "The password of the Bohrium platform"
    doc_project_id = "The project ID of the Bohrium platform"
    doc_host = (
        "The host name of the Bohrium platform. Will overwrite `dflow_config['host']`"
    )
    doc_k8s_api_server = "The k8s server of the Bohrium platform. Will overwrite `dflow_config['k8s_api_server']`"
    doc_repo_key = "The repo key of the Bohrium platform. Will overwrite `dflow_s3_config['repo_key']`"
    doc_storage_client = "The storage client of the Bohrium platform. Will overwrite `dflow_s3_config['storage_client']`"

    return [
        Argument("username", str, optional=False, doc=doc_username),
        Argument("password", str, optional=False, doc=doc_password),
        Argument("project_id", int, optional=False, doc=doc_project_id),
        Argument(
            "host",
            str,
            optional=True,
            default="https://workflows.deepmodeling.com",
            doc=doc_host,
        ),
        Argument(
            "k8s_api_server",
            str,
            optional=True,
            default="https://workflows.deepmodeling.com",
            doc=doc_k8s_api_server,
        ),
        Argument(
            "repo_key", str, optional=True, default="oss-bohrium", doc=doc_repo_key
        ),
        Argument(
            "storage_client",
            str,
            optional=True,
            default="dflow.plugins.bohrium.TiefblueClient",
            doc=doc_storage_client,
        ),
    ]


def default_step_config_args():
    doc_default_step_config = "The default step configuration."

    return [
        Argument(
            "default_step_config",
            dict,
            step_conf_args(),
            optional=True,
            default={},
            doc=doc_default_step_config,
        ),
    ]


def dpgen_step_config_args(default_config):
    doc_prep_train_config = "Configuration for prepare train"
    doc_run_train_config = "Configuration for run train"
    doc_prep_explore_config = "Configuration for prepare exploration"
    doc_run_explore_config = "Configuration for run exploration"
    doc_prep_fp_config = "Configuration for prepare fp"
    doc_run_fp_config = "Configuration for run fp"
    doc_select_confs_config = "Configuration for the select confs"
    doc_collect_data_config = "Configuration for the collect data"
    doc_cl_step_config = "Configuration for the concurrent learning step"

    return [
        Argument(
            "prep_train_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_train_config,
        ),
        Argument(
            "run_train_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_train_config,
        ),
        Argument(
            "prep_explore_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_explore_config,
        ),
        Argument(
            "run_explore_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_explore_config,
        ),
        Argument(
            "prep_fp_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_fp_config,
        ),
        Argument(
            "run_fp_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_fp_config,
        ),
        Argument(
            "select_confs_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_select_confs_config,
        ),
        Argument(
            "collect_data_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_collect_data_config,
        ),
        Argument(
            "cl_step_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_cl_step_config,
        ),
    ]


def submit_args(default_step_config=normalize_step_dict({})):
    doc_bohrium_config = "Configurations for the Bohrium platform."
    doc_step_configs = "Configurations for executing dflow steps"
    doc_upload_python_packages = "Upload python package, for debug purpose"
    doc_inputs = "The input parameter and artifacts for dpgen2"
    doc_train = "The configuration for training"
    doc_explore = "The configuration for exploration"
    doc_fp = "The configuration for FP"
    doc_name = "The workflow name, 'dpgen' for default"

    return (
        dflow_conf_args()
        + default_step_config_args()
        + [
            Argument(
                "bohrium_config",
                dict,
                bohrium_conf_args(),
                optional=True,
                default=None,
                doc=doc_bohrium_config,
            ),
            Argument(
                "step_configs",
                dict,
                dpgen_step_config_args(default_step_config),
                optional=True,
                default={},
                doc=doc_step_configs,
            ),
            Argument(
                "upload_python_packages",
                [List[str], str],
                optional=True,
                default=None,
                doc=doc_upload_python_packages,
                alias=["upload_python_package"],
            ),
            Argument("inputs", dict, input_args(), optional=False, doc=doc_inputs),
            Argument(
                "train", dict, [], [variant_train()], optional=False, doc=doc_train
            ),
            Argument(
                "explore",
                dict,
                [],
                [variant_explore()],
                optional=False,
                doc=doc_explore,
            ),
            Argument("fp", dict, [], [variant_fp()], optional=False, doc=doc_fp),
            Argument("name", str, optional=True, default="dpgen", doc=doc_name),
        ]
    )


def normalize(data):
    default_step_config = normalize_step_dict(data.get("default_step_config", {}))
    defs = submit_args(default_step_config)

    base = Argument("base", dict, defs)
    data = base.normalize_value(data, trim_pattern="_*")
    # not possible to strictly check arguments, dirty hack!
    base.check_value(data, strict=False)
    return data


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sca = submit_args()
    base = Argument("submit", dict, sca)
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    return "\n\n".join(ptr)
