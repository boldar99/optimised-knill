import itertools
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import pyzx as zx
import stim

from pyzx.rewrite_rules import remove_id, fuse

from code_examples import *


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
                 qubit_indices: dict[int, int],
                 paths: dict[int, list[int]]) -> None:
        self.G = G
        self.pos = pos
        self.node_types = node_types
        self.qubit_indices = qubit_indices
        self.paths = paths

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

        return cls(G, pos, node_types, qubit_indices, dict(paths))

    @staticmethod
    def preprocess_diagram(diagram: zx.Graph):
        for v in diagram.vertex_set():
            if diagram.vertex_degree(v) == 1 and diagram.type(v) != zx.VertexType.BOUNDARY:
                [n] = diagram.neighbors(v)
                fuse(diagram, n, v)

        for v in diagram.vertex_set():
            if diagram.vertex_degree(v) == 2 and diagram.type(v) != zx.VertexType.BOUNDARY:
                remove_id(diagram, v)

    def visualize(self):
        world_line_edges = []
        for p in self.paths.values():
            world_line_edges += list(zip(p, p[1:]))

        node_colors = [self.TYPE_COLORS[self.node_types[n]] for n in self.G.nodes()]

        plt.figure(figsize=(18, 15))
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
                v = path_nodes[i+1]
                constraint_graph.add_edge(u, v)

                for neighbor in self.G.neighbors(v):
                    if neighbor != u:
                        constraint_graph.add_edge(u, neighbor)
        return constraint_graph

    def check_causal_flow(self, paths=None) -> bool:
        paths_to_check = paths if paths is not None else self.paths
        constraint_graph = self._construct_flow_graph(paths_to_check)
        return nx.is_directed_acyclic_graph(constraint_graph)

    def greedy_bend(self):
        """
        Iteratively improves the path cover by checking 'bell bends' (merging paths).
        """
        new_paths_candidate = {}
        current_paths = self.paths

        while new_paths_candidate is not None:
            new_paths_candidate = None
            min_pcheck = self._num_parity_measurement(current_paths)

            for bend_at in self._bell_bends(current_paths):
                maybe_candidate = self._try_apply_bell_bend(current_paths, *bend_at)

                if maybe_candidate is not None:
                    new_paths_candidate = maybe_candidate
                    num_pcheck = self._num_parity_measurement(new_paths_candidate)

                    if num_pcheck < min_pcheck:
                        break

            if new_paths_candidate is not None:
                current_paths = new_paths_candidate

        self.paths = current_paths

    def insert_empty_spiders(self):
        next_free_id = max(self.G.nodes()) + 1

        all_path_edges = []
        firsts, lasts = {}, {}
        for q, p in self.paths.items():
            all_path_edges += [self._sorted_pair(v1, v2) for v1, v2 in zip(p, p[1:])]
            firsts[p[0]] = q
            lasts[p[-1]] = q

        edges_to_process = list(self.G.edges())
        for u, v in edges_to_process:
            is_parity = (self.node_types[u] == self.node_types[v]
                         and self._sorted_pair(u, v) not in all_path_edges)

            if is_parity:
                self.G.remove_edge(u, v)
                self.G.add_node(next_free_id)
                self.G.add_edge(u, next_free_id)
                self.G.add_edge(next_free_id, v)

                self.node_types[next_free_id] = (zx.VertexType.Z
                                                 if self.node_types[u] == zx.VertexType.X
                                                 else zx.VertexType.X)

                x_u, y_u = self.pos[u]
                x_v, y_v = self.pos[v]
                self.pos[next_free_id] = ((x_u + x_v) / 2, (y_u + y_v) / 2)

                if u in firsts:
                    self.paths[firsts[u]].insert(0, next_free_id)
                    del firsts[u]
                elif v in firsts:
                    self.paths[firsts[v]].insert(0, next_free_id)
                    del firsts[v]
                elif u in lasts:
                    self.paths[lasts[u]].append(next_free_id)
                    del lasts[u]
                elif v in lasts:
                    self.paths[lasts[v]].append(next_free_id)
                    del lasts[v]
                else:
                    new_path_key = max(self.paths.keys()) + 1
                    self.paths[new_path_key] = [next_free_id]

                next_free_id += 1

    def extract_circuit(self) -> stim.Circuit:
        if not self.check_causal_flow():
            raise ValueError("Circuit must have causal flow.")

        ops = self._find_total_ordering()
        circ = stim.Circuit()
        for txt, qbts in ops:
            circ.append(txt, qbts)
        return circ

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

    def _bell_bends(self, paths):
        vertex_qubit = {}
        for k, path in paths.items():
            for v in path:
                vertex_qubit[v] = k

        for v, w in self.G.edges():
            # Check if start of one path connects to start of another
            if paths[vertex_qubit[v]][0] == v and paths[vertex_qubit[w]][0] == w:
                yield vertex_qubit[v], vertex_qubit[w]

    def _try_apply_bell_bend(self, paths, i, j):
        merged_path = list(reversed(paths[i])) + paths[j]

        new_paths_1 = paths.copy()
        del new_paths_1[i]
        new_paths_1[j] = merged_path

        if self.check_causal_flow(new_paths_1):
            return new_paths_1

        new_paths_2 = paths.copy()
        del new_paths_2[j]
        new_paths_2[i] = list(reversed(merged_path))

        if self.check_causal_flow(new_paths_2):
            return new_paths_2

        return None

    def _find_total_ordering(self):
        ordered_operations = []

        qubits = list(self.paths.keys())
        qubits.sort()
        qubit_to_qubit = {q: i for i, q in enumerate(qubits)}

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


def code_8_3_2():
    h1, cnots_list, h2 = code_8_3_2_gates()
    return build_zx_circuit(8, 8, h1, h2, cnots_list)


if __name__ == '__main__':
    diagram = code_15_7_3().to_graph()
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)
    cov_graph.visualize()

    CoveredZXGraph.preprocess_diagram(diagram)
    cov_graph = CoveredZXGraph.from_zx_diagram(diagram)

    cov_graph.visualize()

    # Run Optimization
    cov_graph.greedy_bend()
    cov_graph.visualize()

    cov_graph.insert_empty_spiders()
    cov_graph.realign_pos()
    cov_graph.visualize()

    # Extract Circuit
    c = cov_graph.extract_circuit()
    svg_content = c.diagram('timeline-svg')

    # png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
    # image = Image.open(io.BytesIO(png_data))
    #
    # plt.figure(figsize=(12, 10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()