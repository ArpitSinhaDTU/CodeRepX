
def kheapsort(arr, k):
    import heapq

    heap = arr[:k+1]
    heapq.heapify(heap)

    for i in range(k+1, len(arr)):
        yield heapq.heappushpop(heap, arr[i])

    while heap:
        yield heapq.heappop(heap)
