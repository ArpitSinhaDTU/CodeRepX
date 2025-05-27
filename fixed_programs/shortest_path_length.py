
from heapq import *

def shortest_path_length(length_by_edge, startnode, goalnode):
    unvisited_nodes = [(0, startnode)] # FibHeap containing (node, distance) pairs
    heapify(unvisited_nodes)
    visited_nodes = set()
 
    while len(unvisited_nodes) > 0:
        distance, node = heappop(unvisited_nodes)
        if node is goalnode:
            return distance

        visited_nodes.add(node)

        for nextnode in get_successors(node, length_by_edge):
            if nextnode in visited_nodes:
                continue
            new_distance = distance + length_by_edge[(node, nextnode)]
            if any(n == nextnode and d <= new_distance for d, n in unvisited_nodes):
                continue
            
            heappush(unvisited_nodes, (new_distance, nextnode))

    return float('inf')

def get_successors(node, length_by_edge):
    successors = []
    for (u, v) in length_by_edge:
        if u == node:
            successors.append(v)
    return successors

"""
Shortest Path

dijkstra

Implements Dijkstra's algorithm for finding a shortest path between two nodes in a directed graph.

Input:
   length_by_edge: A dict with every directed graph edge's length keyed by its corresponding ordered pair of nodes
   startnode: A node
   goalnode: A node

Precondition:
    all(length > 0 for length in length_by_edge.values())

Output:
    The length of the shortest path from startnode to goalnode in the input graph
"""
