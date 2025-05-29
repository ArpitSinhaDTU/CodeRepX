def shortest_paths(source, weight_by_edge):
    weight_by_node = {v: float('inf') for u, v in weight_by_edge}
    all_nodes = set()
    for u, v in weight_by_edge:
        all_nodes.add(u)
        all_nodes.add(v)
    weight_by_node[source] = 0

    for i in range(len(all_nodes) - 1):
        for (u, v), weight in weight_by_edge.items():
            if weight_by_node[u] != float('inf') and weight_by_node[u] + weight < weight_by_node[v]:
                weight_by_node[v] = weight_by_node[u] + weight

    for (u, v), weight in weight_by_edge.items():
        if weight_by_node[u] != float('inf') and weight_by_node[u] + weight < weight_by_node[v]:
            raise Exception("Negative cycle detected")

    return weight_by_node
