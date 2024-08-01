from phaseq import *

json_file = "6-31g_st__st_.1.json"
b = basis_from_json(json_file)
assert len(b["Li"][-1] == 6)
