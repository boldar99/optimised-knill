import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pyzx as zx

from code_examples import *

type_to_color = {
    zx.VertexType.Z: "#66cc66",  # Lighter green
    zx.VertexType.X: "#ff6666",  # Lighter red
    zx.VertexType.BOUNDARY: "black",
}


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


def basic_FE_opt(diagram):
    for v in diagram.vertex_set():
        if diagram.vertex_degree(v) == 1 and diagram.type(v) != zx.VertexType.BOUNDARY:
            [n] = diagram.neighbors(v)
            zx.basicrules.fuse(diagram, n, v)

    for v in diagram.vertex_set():
        if diagram.vertex_degree(v) == 2 and diagram.type(v) != zx.VertexType.BOUNDARY:
            zx.basicrules.remove_id(diagram, v)


def zx_diagram_to_networkx_graph(graph):
    graph_dict = graph.to_dict()
    G = nx.Graph()
    pos = {}
    node_types = {}
    qubit_indices = {}

    for v_data in graph_dict['vertices']:
        v_id = v_data['id']
        G.add_node(v_id)

        (row, qubit) = v_data['pos']

        pos[v_id] = (row, -qubit)
        node_types[v_id] = v_data['t']
        qubit_indices[v_id] = qubit

    for u, v, _ in graph_dict['edges']:
        G.add_edge(u, v)

    return G, pos, node_types, qubit_indices


def trivial_paths(G, qubit_indices, pos):
    paths = defaultdict(list)
    for v in G.nodes():
        paths[qubit_indices[v]].append(v)
    for path in paths.values():
        path.sort(key=lambda v: pos[v][0])
    return paths


def visualize_path_cover(G, pos, node_types, paths):
    world_line_edges = []
    for q, p in paths.items():
        world_line_edges += list(zip(p, p[1:]))

    node_colors = [type_to_color[node_types[n]] for n in G.nodes()]

    # Plotting
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(),
                           edge_color='gray', alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=world_line_edges,
                           edge_color='#336699', width=2, arrows=True, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=8)
    plt.show()


def check_causal_flow(G, paths_dict):
    """
    Checks if a set of directed paths on an undirected graph forms a valid
    Causal Flow.

    The condition requires that for every directed step u -> v in the paths:
    1. u must precede v (u < v).
    2. u must precede ALL neighbors of v (except u itself).
       (i.e., if v 'corrects' u, then v's interactions must happen after u).
    """
    constraint_graph = nx.DiGraph()
    constraint_graph.add_nodes_from(G.nodes())

    for path_nodes in paths_dict.values():
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i+1]

            constraint_graph.add_edge(u, v)

            v_neighbors = G.neighbors(v)
            for neighbor in v_neighbors:
                if neighbor != u:
                    constraint_graph.add_edge(u, neighbor)

    return nx.is_directed_acyclic_graph(constraint_graph)


def sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def get_uncovered_edges(G, paths_dict):
    all_edges = set(sorted_pair(u, v) for u, v in G.edges())
    path_edges = list()
    for path in paths_dict.values():
        path_edges += list(zip(path[1:], path[:-1]))
    path_edges = set(sorted_pair(u, v) for u, v in path_edges)
    return all_edges.difference(path_edges)


def num_parity_measurement(G, paths_dict, node_types):
    sum = 0
    uncovered_edges = get_uncovered_edges(G, paths_dict)
    for e in uncovered_edges:
        v, w = sorted_pair(*e)
        if node_types[v] == node_types[w]:
            sum += 1
    return sum



def bell_bends(G, paths):
    vertex_qubit = dict()
    for k, path in paths.items():
        for v in path:
            vertex_qubit[v] = k

    for v, w in G.edges():
        if paths[vertex_qubit[v]][0] == v and paths[vertex_qubit[w]][0] == w:
            yield vertex_qubit[v], vertex_qubit[w]
    # for v, w in G.edges():
    #     if paths[vertex_qubit[v]][0] == v and paths[vertex_qubit[w]][-1] == w:
    #         yield vertex_qubit[v], vertex_qubit[w]
    #     if paths[vertex_qubit[v]][-1] == v and paths[vertex_qubit[w]][0] == w:
    #         yield vertex_qubit[v], vertex_qubit[w]
    #     if paths[vertex_qubit[v]][-1] == v and paths[vertex_qubit[w]][-1] == w:
    #         yield vertex_qubit[v], vertex_qubit[w]


def try_apply_bell_bend(G, paths, i, j):
    new_paths = copy.deepcopy(paths)
    merged_path = list(reversed(paths[i])) + paths[j]
    del new_paths[j]
    new_paths[i] = merged_path

    if check_causal_flow(G, new_paths):
        return new_paths

    new_paths[i].reverse()
    if check_causal_flow(G, new_paths):
        return new_paths

    return None


def greedy_bend(G, paths, node_types):
    can_bend = True
    while can_bend:
        can_bend = False
        min_pcheck = num_parity_measurement(G, paths, node_types)

        for bend_at in bell_bends(G, paths):
            new_paths = try_apply_bell_bend(G, paths, *bend_at)

            if new_paths is not None:
                can_bend = True
                new_paths_candidate = new_paths

                num_pcheck = num_parity_measurement(G, new_paths_candidate, node_types)
                if num_pcheck < min_pcheck:
                    break

        if can_bend:
            paths = new_paths_candidate
        visualize_path_cover(G, pos, node_types, paths)

    return paths

if __name__ == '__main__':
    diagram = steane_code().to_graph()
    basic_FE_opt(diagram)
    G, pos, node_types, qubit_indices = zx_diagram_to_networkx_graph(diagram)
    paths = trivial_paths(G, qubit_indices, pos)
    visualize_path_cover(G, pos, node_types, paths)
    new_paths = greedy_bend(G, paths, node_types)
    visualize_path_cover(G, pos, node_types, new_paths)
    print(len(new_paths))

