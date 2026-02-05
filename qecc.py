import dataclasses
import json

import numpy as np
from dataclasses import dataclass

import stim


@dataclass
class StatePreparationCircuit:
    ket_zero: list[int]
    ket_plus: list[int]
    cnots: list[tuple[int, int]]
    bra_zero: list[int]
    bra_plus: list[int]

    def to_stim(self):
        circ = stim.Circuit()

        for i in self.ket_plus:
            circ.append("H", i)
        for c, n in self.cnots:
            circ.append("CNOT", [c, n])
        for i in self.bra_plus:
            circ.append("H", i)
        for i in self.ket_zero + self.ket_plus:
            circ.append("MR", i)

        return circ




DATA_TYPE = np.uint8

@dataclass
class QECC:
    def __init__(self, n, k, d, H_x, H_z, L_x, L_z, ft_zero, ft_plus, non_ft_zero, non_ft_plus):
        self.n = n
        self.k = k
        self.d = d

    @classmethod
    def from_json(cls, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            n = data['n']
            k = data['k']
            d = data['d']
            if data['is_self_dual'] == 1:
                H_x = H_z = np.asarray(data['H_x'], dtype=DATA_TYPE)
                L_x = L_z = np.asarray(data['L_x'], dtype=DATA_TYPE)
                ft_zero_state_prep = data['fault_tolerant_zero_state_prep']
                non_ft_zero_state_prep = data['non_fault_tolerant_zero_state_prep']

            else:
                H_x = np.asarray(data['H_x'], dtype=DATA_TYPE)
                H_z = np.asarray(data['H_z'], dtype=DATA_TYPE)
                L_x = np.asarray(data['L_x'], dtype=DATA_TYPE)
                L_z = np.asarray(data['L_z'], dtype=DATA_TYPE)
                ft_zero_state_prep = ft_plus_state_prep = data['fault_tolerant_zero_state_prep']
