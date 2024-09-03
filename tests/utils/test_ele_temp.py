import dpdata
import glob
import numpy as np
import os
import shutil
import unittest

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.utils import (
    setup_ele_temp,
)

# isort: on


class TestSetupEleTemp(unittest.TestCase):
    def test_setup_ele_temp(self):
        system = dpdata.LabeledSystem(
            data={
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": np.zeros(1, dtype=int),
                "cells": np.eye(3).reshape(1, 3, 3),
                "coords": np.zeros((1, 1, 3)),
                "energies": np.zeros(1),
                "forces": np.zeros((1, 1, 3)),
                "orig": np.zeros(3),
                "nopbc": True,
            }
        )
        setup_ele_temp(False)
        system.data["fparam"] = np.array([[1.0]])
        system.to_deepmd_npy("ele_temp_data")
        self.assertEqual(len(glob.glob("ele_temp_data/*/fparam.npy")), 1)
        new_system = dpdata.LabeledSystem("ele_temp_data", fmt="deepmd/npy")
        self.assertTrue("fparam" in new_system.data)

    def tearDown(self):
        if os.path.exists("ele_temp_data"):
            shutil.rmtree("ele_temp_data")
