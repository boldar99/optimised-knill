import copy
import itertools
from collections import defaultdict
from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import pyzx as zx
import stim

from code_examples import *
from verify_fault_tolerance import list_to_str_stabs, build_css_syndrome_table, verify_extraction_circuit


class CoveredZXGraph:
    TYPE_COLORS = {
        zx.VertexType.Z: "#66cc66",
        zx.VertexType.X: "#ff6666",
        zx.VertexType.BOUNDARY: "black",
    }

    def __init__(self,
                 G: nx.Graph,
                 pos: dict[int, tuple[float, float]],
                 node_types: dict[int, zx.VertexType],
                 paths: dict[int, list[int]]) -> None:
        self.G = G
        self.pos = pos
        self.node_types = node_types
        self.paths = paths
        self.path_reversed = {p: False for p in paths}
        self._num_qubits = len([n for n in node_types.values() if n == zx.VertexType.BOUNDARY]) // 2
        self._measurements = {p[-1]: k - self._num_qubits for k, p in paths.items() if
                              node_types[p[-1]] != zx.VertexType.BOUNDARY}
        self.removed_spiders = dict()

    def path_hash(self) -> int:
        return hash(
            tuple(sorted(tuple((v)) for v in self.paths.values()))
        )

    @classmethod
    def from_zx_diagram(cls, diagram: zx.Graph):
        # Convert to NetworkX
        graph_dict = diagram.to_dict()
        G = nx.Graph()
        pos = {}
        node_types = {}
        qubit_indices = {}

        for v_data in graph_dict['vertices']:
            v_id = v_data['id']
            G.add_node(v_id)
            row, qubit = v_data['pos']
            pos[v_id] = (row, -qubit)
            node_types[v_id] = v_data['t']
            qubit_indices[v_id] = qubit

        for u, v, _ in graph_dict['edges']:
            G.add_edge(u, v)

        # Generate trivial paths
        paths = defaultdict(list)
        for v in G.nodes():
            paths[qubit_indices[v]].append(v)
        for path in paths.values():
            path.sort(key=lambda v: pos[v][0])

        return cls(G, pos, node_types, dict(paths))

    def copy(self):
        cg = CoveredZXGraph(
            self.G.copy(),
            self.pos.copy(),
            self.node_types,
            copy.deepcopy(self.paths),
        )
        cg._num_qubits = self._num_qubits
        cg._measurements = self._measurements.copy()
        cg.removed_spiders = copy.deepcopy(self.removed_spiders)
        cg.path_reversed = copy.deepcopy(self.path_reversed)
        return cg

    def _purge_vertex_information(self, v):
        del self.pos[v]
        del self.node_types[v]
        for id, path in list(self.paths.items()):
            if v in path:
                measurement_key_change = (
                        v == path[-1] and len(path) > 1 and v in self._measurements
                )
                if measurement_key_change:
                    self._measurements[path[-2]] = self._measurements[v]
                    del self._measurements[v]
                path.remove(v)
            if len(path) == 0:
                del self.paths[id]

    def fuse(self, u, v):
        if not (self.G.has_edge(u, v)
                and self.node_types[u] == self.node_types[v]
                and self.node_types[u] in (zx.VertexType.X, zx.VertexType.Z)):
            return False

        self.G.remove_edge(u, v)
        u_neighbors = list(self.G.neighbors(u))
        self.G.remove_node(u)
        for n in u_neighbors:
            self.G.add_edge(n, v)

        self._purge_vertex_information(u)

        return True

    def _remove_id_check_flow(self, v):
        paths_copy = self.paths.copy()
        for id, path in list(self.paths.items()):
            if v in path:
                new_path = [p for p in path if p != v]
                paths_copy[id] = new_path
                return self.check_causal_flow(paths_copy)
        return True

    def remove_id(self, v, flow_preserving=True, parity_measurement_preserving=True):
        if self.G.degree(v) != 2:
            return False

        [n1, n2] = list(self.G.neighbors(v))
        self.G.remove_node(v)
        self.G.add_edge(n1, n2)

        flow_check = flow_preserving and not self._remove_id_check_flow(v)
        parity_spider_check =  (parity_measurement_preserving and
                                (self.node_types[n1] == self.node_types[n2] != self.node_types[v]) and
                                (self.G.degree(n1) != 2 and self.G.degree(n2) != 2)
                                )


        if  flow_check or parity_spider_check:
            self.G.remove_edge(n1, n2)
            self.G.add_node(v)
            self.G.add_edge(n1, v)
            self.G.add_edge(n2, v)
            return False

        path_id = next(pid for pid, path in self.paths.items() if v in path)
        in_path_ix = self.paths[path_id].index(v)
        next_node_in_path = self.paths[path_id][in_path_ix + 1] if in_path_ix < (len(self.paths[path_id]) - 1)  else None
        previous_node_in_path = self.paths[path_id][in_path_ix - 1] if in_path_ix > 0  else None

        removal_record = {
            'id': v,
            'node_type': self.node_types[v],
            'pos': self.pos[v],
            'path_id': path_id,
            'next_node_in_path': next_node_in_path,
            'previous_node_in_path': previous_node_in_path,
        }
        edge_key = self._sorted_pair(n1, n2)
        self.removed_spiders[edge_key] = removal_record

        self._purge_vertex_information(v)
        return True

    def basic_FE_rewrites(self, max_cut=1000):
        for v in list(self.G.nodes()):
            if self.G.degree(v) == 1 and self.node_types[v] != zx.VertexType.BOUNDARY:
                [n] = self.G.neighbors(v)
                self.fuse(v, n)

        for v, _ in zip(list(self.G.nodes()), range(max_cut)):
            self.remove_id(v)

    def visualize(self, figsize=(15, 12)):
        world_line_edges = []
        for p in self.paths.values():
            world_line_edges += list(zip(p, p[1:]))

        node_colors = [self.TYPE_COLORS[self.node_types[n]] for n in self.G.nodes()]

        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=150, edgecolors='black')
        nx.draw_networkx_edges(self.G, self.pos, edgelist=self.G.edges(),
                               edge_color='gray', alpha=0.8)
        nx.draw_networkx_edges(self.G, self.pos, edgelist=world_line_edges,
                               edge_color='#336699', width=2, arrows=True, arrowstyle='->')
        nx.draw_networkx_labels(self.G, self.pos, font_color='white', font_size=8)
        plt.show()

    def _construct_flow_graph(self, paths_dict) -> nx.DiGraph:
        constraint_graph = nx.DiGraph()
        constraint_graph.add_nodes_from(self.G.nodes())

        for path_nodes in paths_dict.values():
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                constraint_graph.add_edge(u, v)

                for neighbor in self.G.neighbors(v):
                    if neighbor != u:
                        constraint_graph.add_edge(u, neighbor)
        return constraint_graph

    def check_causal_flow(self, paths=None) -> bool:
        paths_to_check = paths if paths is not None else self.paths
        constraint_graph = self._construct_flow_graph(paths_to_check)
        return nx.is_directed_acyclic_graph(constraint_graph)

    def greedy_path_opt(self):
        """
        Iteratively improves the path cover by checking 'bell bends' (merging paths).
        """
        new_paths_candidate = {}
        current_paths = self.paths

        while new_paths_candidate is not None:
            new_paths_candidate = None
            min_pcheck = self._num_parity_measurement(current_paths)

            for new_paths_candidate, bend_indices in self.all_causal_boundary_bends(current_paths):
                num_pcheck = self._num_parity_measurement(new_paths_candidate)

                if num_pcheck < min_pcheck:
                    break

            if new_paths_candidate is not None:
                current_paths = new_paths_candidate

        self.paths = current_paths

    def all_causal_boundary_bends(self, paths=None):
        """
        Returns all new paths candidates that are causal
        """
        current_paths = paths or self.paths

        for bend_indices in self._boundary_bends(current_paths):
            for bend, reversed_path_index in self._causal_path_bends(current_paths, *bend_indices):
                yield bend, reversed_path_index

    def insert_empty_spiders(self):
        all_path_edges = []
        firsts, lasts = {}, {}
        for q, p in self.paths.items():
            all_path_edges += [self._sorted_pair(v1, v2) for v1, v2 in zip(p, p[1:])]
            firsts[p[0]] = q
            lasts[p[-1]] = q

        for u, v in list(self.G.edges()):
            pair = self._sorted_pair(u, v)
            is_parity = (self.node_types[u] == self.node_types[v]
                         and pair not in all_path_edges)

            if is_parity:
                record = self.removed_spiders[pair]
                self.restore_spider_in_graph(record, u, v)
                self.restore_spider_in_path(record)

    def restore_spider_in_path(self, record):
        spider_id = record['id']
        next_node = record['next_node_in_path']
        prev_node = record['previous_node_in_path']
        path_id = next(
            (k for k, p in self.paths.items() if next_node in p or prev_node in p),
            None
        )
        if path_id is None:
            path_id = record['path_id']
            self.paths[path_id].append(spider_id)
            return

        path = self.paths[path_id]
        if next_node in path:
            i = path.index(next_node)
            if self.path_reversed[record['path_id']]:
                path.insert(i + 1, spider_id)
            else:
                path.insert(i, spider_id)
            pass
        elif prev_node in path:
            i = path.index(prev_node)
            if i == (len(path) - 1):
                self._measurements[spider_id] = self._measurements[path[i]]
                del self._measurements[path[i]]
            if self.path_reversed[record['path_id']]:
                path.insert(i, spider_id)
            else:
                path.insert(i + 1, spider_id)
            pass

    def restore_spider_in_graph(self, record, u, v):
        spider_id = record['id']
        self.G.remove_edge(u, v)
        self.G.add_node(spider_id)
        self.G.add_edge(u, spider_id)
        self.G.add_edge(spider_id, v)

        # 2. Restore Attributes
        self.node_types[spider_id] = record['node_type']
        self.pos[spider_id] = record['pos']

    def _try_insert_firsts(self, u, firsts, next_free_id):
        if u in firsts:
            self.paths[firsts[u]].insert(0, next_free_id)
            if not self.check_causal_flow():
                self.paths[firsts[u]].pop(0)
                return False
            del firsts[u]
            return True
        return False

    def _try_insert_lasts(self, u, lasts, next_free_id):
        if u in lasts:
            self.paths[lasts[u]].append(next_free_id)
            if not self.check_causal_flow():
                self.paths[lasts[u]].pop(-1)
                return False
            else:
                self._measurements[next_free_id] = self._measurements[u]
                del self._measurements[u]
                del lasts[u]
                return True
        return False

    def _insert_paths_single_empty_spider(self, u, v, firsts: dict, lasts: dict, next_free_id: int):
        if self._try_insert_firsts(u, firsts, next_free_id):
            return
        if self._try_insert_firsts(v, firsts, next_free_id):
            return
        if self._try_insert_lasts(u, lasts, next_free_id):
            return
        if self._try_insert_lasts(v, lasts, next_free_id):
            return

        new_path_key = max(self.paths.keys()) + 1
        self.paths[new_path_key] = [next_free_id]
        raise ValueError('New path')

    def extract_circuit(self) -> stim.Circuit:
        if not self.check_causal_flow():
            raise ValueError("Circuit must have causal flow.")

        ops = self._find_total_ordering()
        measurements = []
        circ = stim.Circuit()
        for txt, qbts in ops:
            if txt in ("M", "MZ", "MR"):
                measurements.append((txt, qbts))
            else:
                circ.append(txt, qbts)
        measurements.sort(key=lambda x: x[1])
        for txt, qbts in measurements:
            circ.append(txt, qbts)

        return circ

    def matrix_transformation_indices(self) -> list:
        lasts = [p[-1] for p in self.paths.values() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY]
        return [self._measurements[v] for v in lasts if self._measurements[v] < self._num_qubits]

    def measurement_qubit_indices(self) -> list:
        lasts = {i: p[-1] for i, p in self.paths.items() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY}
        path_to_qubit = self._get_path_to_qubit()
        return [path_to_qubit[i] - self._num_qubits for i, v in lasts.items() if self._measurements[v] < self._num_qubits]

    def flag_qubit_indices(self) -> list:
        lasts = {i: p[-1] for i, p in self.paths.items() if self.node_types[p[-1]] != zx.VertexType.BOUNDARY}
        path_to_qubit = self._get_path_to_qubit()
        return [path_to_qubit[i] - self._num_qubits for i, v in lasts.items() if self._measurements[v] >= self._num_qubits]


    def realign_pos(self) -> None:
        ys = [self.pos[path[0]][1] for path in self.paths.values()]
        ys.sort()

        for path in self.paths.values():
            xs = [self.pos[n][0] for n in path]
            xs.sort()
            # Spread out x coords if duplicates exist
            for i in range(len(xs) - 1):
                if xs[i] == xs[i + 1]:
                    xs[i + 1] += 1
            for i in range(len(xs) - 1):
                if xs[i] == xs[i + 1]:
                    xs[i + 1] += 1

            y = min([self.pos[n][1] for n in path])

            for p, x in zip(path, xs):
                self.pos[p] = (x, y)

    @staticmethod
    def _sorted_pair(v1, v2):
        return (v1, v2) if v1 < v2 else (v2, v1)

    def _get_uncovered_edges(self, paths):
        all_edges = set(self._sorted_pair(u, v) for u, v in self.G.edges())
        path_edges = []
        for path in paths.values():
            path_edges += list(zip(path[1:], path[:-1]))
        path_edges_set = set(self._sorted_pair(u, v) for u, v in path_edges)
        return all_edges.difference(path_edges_set)

    def _num_parity_measurement(self, paths):
        count = 0
        uncovered = self._get_uncovered_edges(paths)
        for e in uncovered:
            v, w = self._sorted_pair(*e)
            if self.node_types[v] == self.node_types[w]:
                count += 1
        return count

    def _boundary_bends(self, paths):
        vertex_qubit = {}
        for k, path in paths.items():
            for v in path:
                vertex_qubit[v] = k

        for v, w in self.G.edges():
            # Check if start of one path connects to start of another
            v_is_first_or_last = v in (paths[vertex_qubit[v]][0], paths[vertex_qubit[v]][-1])
            w_is_first_or_last = w in (paths[vertex_qubit[w]][0], paths[vertex_qubit[w]][-1])

            if v_is_first_or_last and w_is_first_or_last:
                yield vertex_qubit[v], vertex_qubit[w]

    def _causal_path_bends(self, paths, i, j):
        merged_path = list(reversed(paths[i])) + paths[j]

        new_paths_1 = copy.deepcopy(paths)
        del new_paths_1[i]
        new_paths_1[j] = merged_path

        if self.check_causal_flow(new_paths_1):
            yield new_paths_1, i

        new_paths_2 = copy.deepcopy(paths)
        del new_paths_2[j]
        new_paths_2[i] = list(reversed(merged_path))

        if self.check_causal_flow(new_paths_2):
            yield new_paths_2, j

    def _get_path_to_qubit(self):
        qubits = list(self.paths.keys())
        qubits.sort()
        return {q: i for i, q in enumerate(qubits)}

    def _find_total_ordering(self):
        ordered_operations = []

        qubit_to_qubit = self._get_path_to_qubit()

        node_to_qubit = {}
        lasts = set()

        for q, p in self.paths.items():
            for u in p:
                node_to_qubit[u] = qubit_to_qubit[q]
            if self.node_types[p[0]] == zx.VertexType.Z:
                ordered_operations.append(("H", qubit_to_qubit[q]))
            lasts.add(p[-1])

        path_edges_list = [[self._sorted_pair(v1, v2) for v1, v2 in zip(p, p[1:])]
                           for p in self.paths.values()]
        all_path_edges = list(itertools.chain(*path_edges_list))

        constraint_graph = self._construct_flow_graph(self.paths)

        # Remove boundary nodes from constraint calculation
        for n in list(self.G.nodes()):
            if self.node_types[n] == zx.VertexType.BOUNDARY:
                if constraint_graph.has_node(n):
                    constraint_graph.remove_node(n)

        processed_edges = set()

        # Topological sort processing
        while len(constraint_graph.nodes()) > 0:
            # Find nodes with in-degree 0
            sources = [n for n, d in constraint_graph.in_degree() if d == 0]

            if not sources:
                raise ValueError("No solution found (cycle detected in flow).")

            for s in sources:
                constraint_graph.remove_node(s)

                neighbors = sorted(self.G.neighbors(s),
                                   key=lambda n: int(self.node_types[n] == self.node_types[s]))

                qs = node_to_qubit[s]
                ts = self.node_types[s]

                for n in neighbors:
                    if self.node_types[n] == zx.VertexType.BOUNDARY:
                        continue

                    edge = self._sorted_pair(s, n)
                    if edge in all_path_edges or edge in processed_edges:
                        continue

                    qn = node_to_qubit[n]
                    tn = self.node_types[n]

                    if ts != tn:
                        if ts == zx.VertexType.Z:
                            ordered_operations.append(("CNOT", (qs, qn)))
                        else:
                            ordered_operations.append(("CNOT", (qn, qs)))

                    processed_edges.add(edge)

                if s in lasts:
                    if ts == zx.VertexType.Z:
                        ordered_operations.append(("H", qs))
                    ordered_operations.append(("M", qs))

        return ordered_operations


def build_zx_circuit(n_data: int, n_ancillae: int, h1, h2, cnots):
    circ = zx.Circuit(n_data)

    for i in range(n_data, n_data + n_ancillae):
        if i in h1:
            circ.add_gate("InitAncilla", label=i, basis="X")
        else:
            circ.add_gate("InitAncilla", label=i, basis="Z")

    for c, n in cnots:
        circ.add_gate("CNOT", c, n)

    for i in range(n_data, n_data + n_ancillae):
        if i in h2:
            circ.add_gate("PostSelect", label=i, basis="X")
        else:
            circ.add_gate("PostSelect", label=i, basis="Z")

    return circ


def steane_code():
    h1, cnots_list, h2 = steane_code_gates()
    return build_zx_circuit(7, 8, h1, h2, cnots_list)


def code_15_7_3():
    h1, cnots_list, h2 = code_15_7_3_gates()
    return build_zx_circuit(15, 17, h1, h2, cnots_list)


def code_nonFT_15_7_3():
    h1, cnots_list, h2 = code_nonFT_15_7_3_gates()
    return build_zx_circuit(15, 16, h1, h2, cnots_list)


def code_8_3_2():
    h1, cnots_list, h2 = code_8_3_2_gates()
    return build_zx_circuit(8, 8, h1, h2, cnots_list)

def all_good_FT_opts(
        covered_zx_graph: CoveredZXGraph,
        H_matrix: np.ndarray,  # Binary Matrix (num_stabilizers x num_data_qubits)
        L_matrix: np.ndarray,  # Binary Matrix (num_logicals x num_data_qubits)
        basis: str,  # "Z" (Measuring Z-stabs) or "X" (Measuring X-stabs)
        d: int
) -> Iterator[CoveredZXGraph]:
    stabs = list_to_str_stabs(H_matrix)
    decoder_table = build_css_syndrome_table(stabs, d)

    yield covered_zx_graph
    covered_graphs = [covered_zx_graph]
    seen = {covered_zx_graph.path_hash()}
    while len(covered_graphs) > 0:
        current_graph = covered_graphs.pop(0)

        for path, reversed_path_index in current_graph.all_causal_boundary_bends():
            cov_graph = current_graph.copy()
            cov_graph.paths = copy.deepcopy(path)
            cov_graph.path_reversed[reversed_path_index] ^= True
            # cov_graph.insert_empty_spiders()
            circ = cov_graph.extract_circuit()
            good = verify_extraction_circuit(
                circ,
                H_matrix,
                L_matrix,
                decoder_table,
                cov_graph.flag_qubit_indices(),
                basis,
                d,
                verbose=True,
            )
            p_hash = cov_graph.path_hash()
            if not good:
                print("BAD")
            if p_hash not in seen:
                covered_graphs.append(cov_graph)
                seen.add(p_hash)
                cov_graph.visualize()
                yield cov_graph
            else:
                print("PRUNED")


if __name__ == '__main__':
    from codes import *

    diagram = code_nonFT_15_7_3().to_graph()
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)
    initial_size = len(cov_graph.paths)

    cov_graph.visualize()
    cov_graph.basic_FE_rewrites()
    cov_graph.visualize()

    # cov_graph.insert_empty_spiders()
    # cov_graph.visualize()
    # print(cov_graph.paths)

    for cv in all_good_FT_opts(
            cov_graph,
            H_z_15_7_3,
            L_x_15_7_3,
            "X", 1):
        print("Number of ancilla qubits:", len(cv.paths) - len(H_z_15_7_3[0]))
        print(
            f"H indices: {cv.matrix_transformation_indices()}; "
            f"Measurement indices: {cv.measurement_qubit_indices()}; "
            f"Flag qubits: {cv.flag_qubit_indices()}"
        )
        # print(cv.matrix_transformation_indices())
        # print(cv.measurement_qubit_indices())
        # print(cv.flag_qubit_indices())
        # print(cv.extract_circuit())

