import itertools

import stim

from code_examples import *
from codes import *


def build_stim_circuit(n_data: int, n_ancillae: int, h1, h2, cnots):
    circ = stim.Circuit()

    for i in h1:
        circ.append("H", i)

    for c, n in cnots:
        circ.append("CNOT", [c, n])

    for i in range(n_data, n_data + n_ancillae):
        if i in h2:
            circ.append("H", i)
        # circ.append("M", i)

    return circ


def steane_code():
    h1, cnots_list, h2 = steane_code_gates()
    return build_stim_circuit(7, 8, h1, h2, cnots_list)


def code_15_7_3():
    h1, cnots_list, h2 = code_15_7_3_gates()
    return build_stim_circuit(15, 17, h1, h2, cnots_list)


def code_8_3_2():
    h1, cnots_list, h2 = code_8_3_2_gates()
    return build_stim_circuit(8, 8, h1, h2, cnots_list)


def build_css_syndrome_table(stabilizers: list[str], d: int):
    """
    Builds a lookup table for CSS codes.
    Only considers pure X error strings and pure Z error strings up to weight t.
    """
    num_qubits = len(stabilizers[0])
    t_faults = (d - 1) // 2

    stab_paulis = [stim.PauliString(s) for s in stabilizers]
    decoder_table = {}

    for w in range(t_faults + 1):
        for qubit_indices in itertools.combinations(range(num_qubits), w):
            for p_type in ["X", "Z"]:
                error = stim.PauliString(num_qubits)
                for q in qubit_indices:
                    error[q] = p_type
                syndrome = tuple(not error.commutes(s) for s in stab_paulis)
                if syndrome not in decoder_table:
                    decoder_table[syndrome] = error.to_numpy()[1]

    return decoder_table


def list_to_str_stabs(stabs):
    xs = [
        "".join("X" if c == 1 else "I" for c in s)
        for s in stabs
    ]
    zs = [
        "".join("Z" if c == 1 else "I" for c in s)
        for s in stabs
    ]
    return xs


def verify_extraction_circuit(
        circuit: stim.Circuit,
        H_matrix: np.ndarray,  # Binary Matrix (num_stabilizers x num_data_qubits)
        L_matrix: np.ndarray,  # Binary Matrix (num_logicals x num_data_qubits)
        correction_table: dict,  # Your pre-computed Phase 1 table
        basis: str,  # "Z" (Measuring Z-stabs) or "X" (Measuring X-stabs)
        d: int
):
    """
    Verifies a syndrome extraction circuit.
    Checks if the 'Residual Syndrome' (Mismatch between Data and Ancilla)
    leads to a logical failure.
    """
    num_data_qubits = len(H_matrix[0])
    t_faults = (d - 1) // 2

    init_str = "" if basis == "Z" else f"H {' '.join(str(q) for q in range(num_data_qubits))}"
    fault_type = "X" if basis == "Z" else "Z"

    ops = explode_circuit(circuit)
    possible_faults = get_fault_locations(ops)

    # Iterate combinations
    for fault_combo in itertools.combinations(possible_faults, t_faults):
        noisy_c = build_noisy_circuit(fault_combo, fault_type, init_str, ops)
        sim = stim.TableauSimulator()
        sim.do_circuit(noisy_c)

        ancilla_record = np.array(sim.current_measurement_record()).astype(int)[:len(H_matrix)]
        flag_record = np.array(sim.current_measurement_record()).astype(int)[len(H_matrix):]
        if basis == "X":
            sim.do_circuit(stim.Circuit(f"H {' '.join(str(q) for q in range(num_data_qubits))}"))
        data_bits = np.array(sim.measure_many(*range(num_data_qubits))).astype(int)
        actual_syndrome = tuple(((H_matrix @ data_bits) % 2).tolist())

        # E. Decode
        if actual_syndrome not in correction_table:
            print(f"FAIL: Syndrome {actual_syndrome} unknown (Hook error spread too far?)")
            # return False
            continue

        correction = correction_table[actual_syndrome]

        # F. Logical Check
        final_state = (data_bits + correction) % 2
        logical_flip = (L_matrix @ final_state) % 2

        # Since we started with Logical 0 (or +), expected logical outcome is 0.
        if np.any(logical_flip) and not np.any(flag_record):
            print(f"LOGICAL FAILURE via Extraction Circuit!")
            print(f"Faults: {fault_combo}")
            print(f"Data has error: {data_bits} (Syndrome {actual_syndrome})")
            print(f"Correction applied: {correction}")
            print(f"Result: Logical Flip")
            print(noisy_c)
            return False

    return True


import stim


def explode_circuit(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """
    Decomposes a circuit into a list of atomic instructions.
    E.g., 'CX 0 1 2 3' becomes ['CX 0 1', 'CX 2 3'].
    This allows injecting faults *between* gates that were originally grouped.
    """
    atomized_ops = []

    # Common 2-qubit gates in CSS codes
    TWO_QUBIT_GATES = {"CX", "CNOT", "CZ", "SWAP", "CY", "XCZ", "YCX"}

    for op in circuit.flattened():
        # Handle 2-Qubit Gates (Target pairs)
        if op.name in TWO_QUBIT_GATES:
            targets = op.targets_copy()
            # Iterate in steps of 2
            for k in range(0, len(targets), 2):
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, targets[k:k + 2], op.gate_args_copy())
                )

        # Handle Annotations (Don't split, just keep)
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}:
            atomized_ops.append(op)

        # Handle 1-Qubit Gates & Measurements (Single targets)
        else:
            # e.g. H, X, Z, M, R, MR
            targets = op.targets_copy()
            for t in targets:
                atomized_ops.append(
                    stim.CircuitInstruction(op.name, [t], op.gate_args_copy())
                )

    return atomized_ops


def get_fault_locations(ops: list) -> list[tuple[int, int]]:
    possible_faults = []
    for i, op in enumerate(ops):
        for target in op.targets_copy():
            if target.is_qubit_target:
                possible_faults.append((i, target.value))
    return possible_faults


def build_noisy_circuit(fault_combo: tuple[tuple[int, int], ...], fault_type: str, init_str: str,
                        ops: list) -> stim.Circuit:
    # A. Build Noisy Circuit
    noisy_c = stim.Circuit()
    noisy_c.append_from_stim_program_text(init_str)  # Initialize clean state

    current_faults = {f[0]: [] for f in fault_combo}
    for f in fault_combo:
        current_faults[f[0]].append(f[1])

    for i, op in enumerate(ops):
        noisy_c.append(op)
        if i in current_faults:
            for q in current_faults[i]:
                noisy_c.append(fault_type, [q])

    # B. Run Simulation
    return noisy_c


if __name__ == "__main__":
    # Example Usage
    stabs = list_to_str_stabs(H_x_15_7_3)
    print(stabs)
    decoder_table = build_css_syndrome_table(stabs, 3)
    print(decoder_table)

    steane_circ = stim.Circuit("""
H 15 17 20 21 23
CX 23 24 9 24 23 18 10 18 3 22 7 19 15 16 19 16 20 16 24 20 22 16 21 22 21 24 16 18 8 20 13 21 18 22 12 22
M 22
CX 2 24 1 16 15 18 14 15 17 18 15 21 24 17 5 17
M 17
CX 18 15 21 16
M 16
CX 4 24 19 15
H 19
M 19
CX 24 20
M 20
CX 24 21
H 24
M 24
CX 11 21
M 21
CX 6 18
M 18
CX 23 15 0 15
M 15
H 23
M 23
    """)

    verify_extraction_circuit(steane_circ, H_x_15_7_3, L_x_15_7_3, decoder_table, "X", 3)

# verify_t_fault_tolerance(steane_code(), list_to_str_stabs(steane_code_stabs()), t_faults=1)
