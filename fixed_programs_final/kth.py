def kth(arr, k):
    if not arr:
        return None
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    above = [x for x in arr if x > pivot]
    
    num_less = len(below)
    num_equal = len(equal)
    
    if k < num_less:
        return kth(below, k)
    elif k < num_less + num_equal:
        return pivot
    else:
        return kth(above, k - num_less - num_equal)