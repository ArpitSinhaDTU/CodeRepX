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

