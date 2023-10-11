from .deepmd import (
    DeepmdInputs,
    PrepDeepmd,
    RunDeepmd,
)
from .deepmd_pt import (
    DeepmdPtInputs,
    PrepDeepmdPt,
    RunDeepmdPt,
)
from .fpop import (
    AbacusInputs,
    PrepFpOpAbacus,
    RunFpOpAbacus,
)
from .gaussian import (
    GaussianInputs,
    PrepGaussian,
    RunGaussian,
)
from .vasp import (
    PrepVasp,
    RunVasp,
    VaspInputs,
)

fp_styles = {
    "vasp": {
        "inputs": VaspInputs,
        "prep": PrepVasp,
        "run": RunVasp,
    },
    "gaussian": {
        "inputs": GaussianInputs,
        "prep": PrepGaussian,
        "run": RunGaussian,
    },
    "deepmd": {
        "inputs": DeepmdInputs,
        "prep": PrepDeepmd,
        "run": RunDeepmd,
    },
    "deepmd_pt": {
        "inputs": DeepmdPtInputs,
        "prep": PrepDeepmdPt,
        "run": RunDeepmdPt,
    },
    "fpop_abacus": {
        "inputs": AbacusInputs,
        "prep": PrepFpOpAbacus,
        "run": RunFpOpAbacus,
    },
}
