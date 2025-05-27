
def detect_cycle(node):
    if not node:
        return False
    
    hare = node
    tortoise = node

    while hare and hare.next:
        tortoise = tortoise.next
        hare = hare.next.next

        if hare is tortoise:
            return True

    return False



"""
Linked List Cycle Detection
tortoise-hare

Implements the tortoise-and-hare method of cycle detection.

Input:
    node: The head node of a linked list

Output:
    Whether the linked list is cyclic
"""
