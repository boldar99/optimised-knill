import json
import pprint
import stim
import numpy as np

from qecc import QECCGadgets


def from_json_serializable(data):
    new_data = {}
    for k, v in data.items():
        sub_keys = k.split("+")
        if len(sub_keys) == 1:
            new_k = tuple(map(int, k))
        else:
            new_k = tuple(tuple(map(int, sub_key)) for sub_key in sub_keys)
        new_data[new_k] = v
    return new_data


def load_optimised_se(code):
    with open(f"simplified_circuits/{code}.json", "r") as f:
        data = json.load(f)

    pprint.pprint(data, width=120)

    data["circuit"] = stim.Circuit(data["circuit"])
    data["lookup_table"] = from_json_serializable(data["lookup_table"])
    data["modified_lookup_table"] = from_json_serializable(data["modified_lookup_table"])

    pprint.pprint(data, width=120)






if __name__ == "__main__":

    qecc = "15_7_3"
    qecc_gadgets = QECCGadgets.from_json(f"circuits/{qecc}.json")
    simp = load_optimised_se(qecc)
