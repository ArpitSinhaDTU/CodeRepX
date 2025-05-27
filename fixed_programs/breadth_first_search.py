
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
            try:
                successors = node.successors
                for successor in successors:
                    if successor not in nodesseen:
                        queue.append(successor)
                        nodesseen.add(successor)
            except AttributeError:
                pass # Node has no successors

    return False



"""
Breadth-First Search


Input:
    startnode: A digraph node
    goalnode: A digraph node

Output:
    Whether goalnode is reachable from startnode
"""
