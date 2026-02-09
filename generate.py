import sys

import stim

from path_cover_opt import CoveredZXGraph
from qecc import QECCGadgets, _layer_cnot_circuit
from pathlib import Path

from verify_fault_tolerance import compute_modified_lookup_table, list_to_str_stabs, build_css_syndrome_table, \
    explode_circuit

cwd = Path.cwd()


def init_circuits_folder():
    Path(f"{cwd}/simplified_circuits").mkdir(parents=True, exist_ok=True)


def circuit_depth(data: dict):
    ops = explode_circuit(data["circuit"])
    cnots = [list(map(lambda x: x.value, op.targets_copy())) for op in ops if op.name in ("CX", "CNOT")]
    return len(_layer_cnot_circuit(cnots))


def depth_minimal_circuit(cov_graph):
    circuit_data = []
    for cv in cov_graph.min_ancilla_boundary_bends():
        data = {
            "circuit": cv.extract_circuit(),
            "H_indices": cv.matrix_transformation_indices(),
            "measurement_indices": cv.measurement_qubit_indices(),
            "flag_indices": cv.flag_qubit_indices(),
        }
        circuit_data.append(data)

    return min(circuit_data, key=circuit_depth)


def generate_simplification(qecc_gadgets):
    diagram = qecc_gadgets.steane_z_syndrome_extraction.to_pyzx()
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)
    cov_graph.basic_FE_rewrites()

    depth_best_circuit_data = depth_minimal_circuit(cov_graph)

    stabs = list_to_str_stabs(qecc_gadgets.code.H_z)
    depth_best_circuit_data["lookup_table"] = build_css_syndrome_table(stabs, qecc_gadgets.code.d)
    depth_best_circuit_data["modified_lookup_table"] = compute_modified_lookup_table(
        depth_best_circuit_data["circuit"],
        qecc_gadgets.code.H_z,
        qecc_gadgets.code.L_x,
        depth_best_circuit_data["lookup_table"],
        depth_best_circuit_data["flag_indices"],
        "X",
        qecc_gadgets.code.d,
        verbose=False,
    )
    return depth_best_circuit_data


if __name__ == "__main__":
    qecc_gadgets = QECCGadgets.from_json("circuits/15_7_3.json")
    print(generate_simplification(qecc_gadgets))
