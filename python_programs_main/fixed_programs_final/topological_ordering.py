def topological_ordering(nodes):
    """
    Topological Sort

    Input:
        nodes: A list of directed graph nodes

    Precondition:
        The input graph is acyclic

    Output:
        A list containing the elements of nodes in an order that puts each node before all the nodes it has edges to
    """
    in_degree = {}
    for node in nodes:
        in_degree[node] = 0

    for node in nodes:
        for neighbor in node.outgoing_nodes:
            if neighbor in in_degree:
                in_degree[neighbor] += 1
            

    ordered_nodes = []
    queue = [node for node in nodes if in_degree[node] == 0]

    while queue:
        node = queue.pop(0)
        ordered_nodes.append(node)

        for neighbor in node.outgoing_nodes:
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
    if len(ordered_nodes) != len(nodes):
        return [] # Cycle exists, topological sort not possible

    return ordered_nodes
