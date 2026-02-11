import json
import pprint
import stim
import numpy as np

from qecc import QECCGadgets, SyndromeMeasurementCircuit, NoiseModel


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

    data["circuit"] = SyndromeMeasurementCircuit.from_dict(data["circuit"])
    data["lookup_table"] = from_json_serializable(data["lookup_table"])
    data["modified_lookup_table"] = from_json_serializable(data["modified_lookup_table"])

    return data


def get_best_correction(samples, flag_pattern, syndrome_pattern):
    flag_pattern_matched = True if flag_pattern is None else np.all(flags == flag_pattern, axis=1)
    syndrome_pattern_matched = True if syndrome_pattern is None else np.all(syndromes == syndrome_pattern, axis=1)
    filtered_samples = samples[np.logical_and(flag_pattern_matched, syndrome_pattern_matched)]

    min_ler = 2
    min_correction = [0] * 15
    for i in range(2 ** 15):
        vec_i = np.array(list(map(int, ('0' * 15 + format(i, "b"))[-15:])))
        corrected_measurements = (filtered_samples[:, -15:] + vec_i) % 2
        logical_samples = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
        logical_error = np.any(logical_samples, axis=1)
        logical_error_rate = np.average(logical_error)
        if logical_error_rate < min_ler:
            min_ler = logical_error_rate
            min_correction = vec_i
    return min_correction, min_ler


def error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern, syndrome_pattern=None):
    flag_pattern_matched = True if flag_pattern is None else np.all(flags == flag_pattern, axis=1)
    syndrome_pattern_matched = True if syndrome_pattern is None else np.all(syndromes == syndrome_pattern, axis=1)
    corrected_measurements = corrected_measurements[np.logical_and(flag_pattern_matched, syndrome_pattern_matched)]
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    print(f"LER when {flag_pattern=}" +
          (f" and {syndrome_pattern=}" if syndrome_pattern is not None else "") +
          f": {np.average(measurement_error):.3%}")
    if syndrome_pattern is not None:
        print(f"Correction to be applied {(measurement_error.shape[0])}: ", simp["modified_lookup_table"].get((flag_pattern, syndrome_pattern), simp["lookup_table"][syndrome_pattern]))


if __name__ == "__main__":
    qecc = "15_7_3"
    qecc_gadgets = QECCGadgets.from_json(f"circuits/{qecc}.json")
    simp = load_optimised_se(qecc)

    # circ = qecc_gadgets.ft_z_state_prep.to_stim(noise_model=NoiseModel.Quantinuum_Helios())
    circ = qecc_gadgets.ft_x_state_prep.to_stim()
    circ += simp["circuit"].to_stim(noise_model=NoiseModel.Quantinuum_Helios())
    circ.append("H", range(15))
    circ.append("M", range(15))

    NUM_SAMPLES = 1_000_000
    smplr = circ.compile_sampler()
    samples = smplr.sample(NUM_SAMPLES)
    flags = samples[:, -17:-15]
    syndromes = samples[:, -15:] @ qecc_gadgets.code.H_z.T % 2

    # print(circ)

    correction = []
    modified_correction = []
    for f, s in zip(flags, syndromes):
        syndrome = tuple(s.tolist())
        flag = tuple(f.astype(int).tolist())
        correction.append(simp["lookup_table"][syndrome])
        modified_correction.append(simp["modified_lookup_table"].get((flag, syndrome),  simp["lookup_table"][syndrome]))
    correction = np.array(correction)
    modified_correction = np.array(modified_correction)
    corrected_measurements = samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)


    corrected_measurements = corrected_measurements[~np.any(flags, axis=1)]
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    print(f"Flag post-selected LER: {np.average(measurement_error):.3%}")
    print(f"Acceptance rate: {len(measurement_error) / NUM_SAMPLES:.3%}")
    print()

    corrected_measurements = samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)
    logical_measurements = corrected_measurements @ qecc_gadgets.code.L_z.T % 2
    measurement_error = np.any(logical_measurements, axis=1)
    print(f"With modified decoding LER: {np.average(measurement_error):.3%}")
    print()

    corrected_measurements = (samples[:, -15:] + np.where(np.any(flags, axis=1)[:,np.newaxis], modified_correction, correction)) % 2
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(1,1))
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(0,1))
    error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(1,0))


    for i in range(1, 2 ** 4):
        vec_i = tuple(map(int, ('0' * 4 + format(i, "b"))[-4:]))
        error_rate_when_flag_and_syndrome_is(corrected_measurements, flag_pattern=(0, 1), syndrome_pattern=vec_i)
        print(get_best_correction(samples, flag_pattern=(0, 1), syndrome_pattern=vec_i))
    print()

