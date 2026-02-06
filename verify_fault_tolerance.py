import itertools

import numpy as np
import stim


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
        flag_measurements: list[int],
        basis: str,  # "Z" (Measuring Z-stabs) or "X" (Measuring X-stabs)
        d: int,
        verbose=False,
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

    modified_correction_table = dict()
    modified_correction_table_origin = dict()

    # Iterate combinations
    for fault_combo in itertools.combinations(possible_faults, t_faults):
        noisy_c = build_noisy_circuit(fault_combo, fault_type, init_str, ops)
        sim = stim.TableauSimulator()
        sim.do_circuit(noisy_c)

        non_flag_measurements = list(set(range(len(sim.current_measurement_record()))) - set(flag_measurements))
        ancilla_record = np.array(sim.current_measurement_record()).astype(int)[non_flag_measurements]
        flag_record = np.array(sim.current_measurement_record()).astype(int)[flag_measurements]
        if basis == "X":
            sim.do_circuit(stim.Circuit(f"H {' '.join(str(q) for q in range(num_data_qubits))}"))
        data_bits = np.array(sim.measure_many(*range(num_data_qubits))).astype(int)
        actual_syndrome = tuple(((H_matrix @ data_bits) % 2).tolist())

        # E. Decode
        if actual_syndrome not in correction_table:
            return False
            continue

        correction = correction_table[actual_syndrome]

        # F. Logical Check
        final_state = (data_bits + correction) % 2
        logical_flip = (L_matrix @ final_state) % 2

        # Since we started with Logical 0 (or +), expected logical outcome is 0.
        if np.any(flag_record):
            print(f"Syndrome: {actual_syndrome} + Flags: {flag_record} -> {logical_flip}")
            full_syndrome = (actual_syndrome, tuple(flag_record.tolist()))
            if full_syndrome in modified_correction_table and np.any(logical_flip == 1):
                if np.any(modified_correction_table[full_syndrome] != logical_flip):
                    print(f"NON UNIQUE CORRECTION for {full_syndrome} for ")
                    print("Saved correction:", modified_correction_table[full_syndrome])
                    print("New correction:", logical_flip)
                    print("Fault Combos:", modified_correction_table_origin[full_syndrome], fault_type)
            elif np.any(logical_flip == 1):
                modified_correction_table[full_syndrome] = logical_flip
                modified_correction_table_origin[full_syndrome] = fault_combo
            else:
                print("Skipping saving measurement")

        elif np.any(logical_flip):
            if verbose:
                print(f"LOGICAL FAILURE via Extraction Circuit!")
                print(f"Faults: {fault_combo}")
                print(f"Data has error: {data_bits} (Syndrome {actual_syndrome})")
                print(f"Correction applied: {correction}")
                print(f"Result: Logical Flip")
                print(noisy_c)
            return False

    return True


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
    stabs = list_to_str_stabs(steane_code_stabs())
    print(stabs)
    decoder_table = build_css_syndrome_table(stabs, 3)
    print(decoder_table)

    steane_circ = stim.Circuit("""
    H 9
    CX 9 7 6 7 0 10 2 8 7 10 8 7 9 8 3 7 10 8 4 8
    M 8
    CX 5 9 1 10 10 7
    M 7
    CX 10 9
    H 10
    M 10 9
        """)

    verify_extraction_circuit(steane_circ, steane_code_stabs(), np.array([0,0,0,0,1,1,1]), decoder_table, [3], "X", 3)

# verify_t_fault_tolerance(steane_code(), list_to_str_stabs(steane_code_stabs()), t_faults=1)
