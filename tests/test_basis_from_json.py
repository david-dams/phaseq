from phaseq import *

json_file = "sto-3g.1.json"
b = basis_from_json(json_file)
assert len(b["Li"][-1] == 3)
