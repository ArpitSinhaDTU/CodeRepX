def quicksort(arr):
    if not arr:
        return []

    pivot = arr[0]
    lesser = [x for x in arr[1:] if x < pivot]
    equal = [x for x in arr[1:] if x == pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quicksort(lesser) + [pivot] + equal + quicksort(greater)