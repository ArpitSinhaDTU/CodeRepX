from collections import deque as Queue
 
def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while queue:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            if hasattr(node, 'successors') and node.successors is not None:
                try:
                    queue.extend(neighbor for neighbor in node.successors if neighbor not in nodesseen)
                    nodesseen.update(node.successors)
                except TypeError:
                    print(f"Error: successors of node {node} is not iterable")
                    return False

    return False
