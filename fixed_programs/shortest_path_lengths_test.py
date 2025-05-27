
from collections import defaultdict

def shortest_path_lengths(n, graph):
    """
    Find shortest path lengths between all pairs of nodes in a graph using the Floyd-Warshall algorithm.

    Args:
        n (int): The number of nodes in the graph (numbered 0 to n-1).
        graph (dict): A dictionary representing the graph where keys are tuples (u, v) representing an edge from node u to node v, and values are the edge weights.

    Returns:
        dict: A dictionary where keys are tuples (u, v) representing a pair of nodes, and values are the shortest path lengths between them.
             Returns float('inf') if there's no path between two nodes.
    """

    dist = defaultdict(lambda: defaultdict(lambda: float('inf')))

    # Initialize distances
    for i in range(n):
        dist[i][i] = 0  # Distance from a node to itself is 0

    for (u, v), weight in graph.items():
        dist[u][v] = weight

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Convert the defaultdict to a regular dictionary for the result.  Use tuples as keys.
    result = {}
    for i in range(n):
        for j in range(n):
            result[(i, j)] = dist[i][j]

    return result


"""
Test shortest path lengths
""" 
def main():
    # Case 1: Basic graph input.
    # Output:
    graph = {
        (0, 2): 3,
        (0, 5): 5,
        (2, 1): -2,
        (2, 3): 7,
        (2, 4): 4,
        (3, 4): -5,
        (4, 5): -1
    }
    result =  shortest_path_lengths(6, graph)
    for node_pairs in result:
        print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 2: Linear graph input.
    # Output:
    graph2 = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 3): -2,
        (3, 4): 7
    }
    result =  shortest_path_lengths(5, graph2)
    for node_pairs in result:
        print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 3: Disconnected graphs input.
    # Output:
    graph3 = {
        (0, 1): 3,
        (2, 3): 5
    }
    result =  shortest_path_lengths(4, graph3)
    for node_pairs in result:
        print(node_pairs, result[node_pairs], end=" ")
    print()

    # Case 4: Strongly connected graph input.
    graph4 = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 0): -1
    }
    result =  shortest_path_lengths(3, graph4)
    for node_pairs in result:
        print(node_pairs, result[node_pairs], end=" ")
    print()


if __name__ == "__main__":
    main()
