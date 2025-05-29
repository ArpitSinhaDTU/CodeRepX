def kheapsort(arr, k):
    new_list = []
    for i in arr:
        if i not in new_list:
            new_list.append(i)
    return new_list
