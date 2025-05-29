def minimum_spanning_tree(weight_by_edge):
    group_by_node = {}
    mst_edges = set()

    for edge in sorted(weight_by_edge, key=weight_by_edge.__getitem__):
        u, v = edge
        group_u = group_by_node.setdefault(u, {u})
        group_v = group_by_node.setdefault(v, {v})
        if group_u != group_v:
            mst_edges.add(edge)
            union = group_u.union(group_v)
            for node in union:
                group_by_node[node] = union

    return mst_edges


 