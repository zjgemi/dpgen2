import os
import unittest

import dpdata
import numpy as np

from dpgen2.exploration.selector import (
    DistanceConfFilter,
)

from .context import (
    dpgen2,
)

POSCAR_1 = """ Er
1.0
   7.00390434172054       0.000000000000000E+000  0.000000000000000E+000
  -3.50195193887670        6.06555921954188       0.000000000000000E+000
  4.695904554609645E-007  8.133544916878595E-007   6.21991417332993
 Er    Th       Pd
           3           3           5
Direct
  0.404315576593774       0.000000000000000E+000  0.916328931175151
  0.000000000000000E+000  0.404315576593774       0.916328931175151
  0.595684423406226       0.595684423406226       0.916328931175151
  0.308657693786501       0.000000000000000E+000  0.431543321200265
  0.000000000000000E+000  0.308657693786501       0.431543321200265
  0.691342306213499       0.691342306213499       0.431543321200265
  0.333299994468689       0.666700005531311       0.181639126706554
  0.666700005531311       0.333299994468689       0.181639126706554
  0.333299994468689       0.666700005531311       0.653715146968972
  0.666700005531311       0.333299994468689       0.653715146968972
  0.000000000000000E+000  0.000000000000000E+000  0.767627989523288
"""


POSCAR_2 = """Re3 B2 Au4
1.0
9.9156076828901600e+00 0.0000000000000000e+00 0.0000000000000000e+00
7.9377192881745906e-07 1.0513827981441199e+01 0.0000000000000000e+00
3.0151272178709800e+00 1.9108776119269221e-08 1.2610865642770299e+00
Re B Au
3 2 4
Cartesian
   7.2596148617    1.5118565582    1.1482828020
   7.2596154272    9.0019714580    1.1482828020
  11.9303483902    0.0000000165    1.0888892057
   4.2582035339    5.2569140081    1.1494063597
   5.4733204545    0.0000000019    0.1275328112
  11.5360001153    6.8393419913    1.1990816350
  11.5359998763    3.6744860264    1.1990816350
   5.4768941202    6.8571150073    0.0060127006
   5.4768938786    3.6567129744    0.0060127006
"""


class TestDistanceConfFilter(unittest.TestCase):
    def setUp(self):
        with open("POSCAR_1", "w") as f:
            f.write(POSCAR_1)
        with open("POSCAR_2", "w") as f:
            f.write(POSCAR_2)

    def test_valid(self):
        system = dpdata.System("POSCAR_1", fmt="poscar")
        distance_conf_filter = DistanceConfFilter()
        valid = distance_conf_filter.check(
            system["coords"][0],
            system["cells"][0],
            np.array([system["atom_names"][t] for t in system["atom_types"]]),
            system.nopbc,
        )
        self.assertTrue(valid)

    def test_invalid(self):
        system = dpdata.System("POSCAR_2", fmt="poscar")
        distance_conf_filter = DistanceConfFilter()
        valid = distance_conf_filter.check(
            system["coords"][0],
            system["cells"][0],
            np.array([system["atom_names"][t] for t in system["atom_types"]]),
            system.nopbc,
        )
        self.assertFalse(valid)

    def tearDown(self):
        if os.path.isfile("POSCAR_1"):
            os.remove("POSCAR_1")
        if os.path.isfile("POSCAR_2"):
            os.remove("POSCAR_2")
