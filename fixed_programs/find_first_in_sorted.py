
def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if arr[mid] == x:
            if mid == 0 or arr[mid - 1] != x:
                return mid
            else:
                hi = mid - 1
        elif x < arr[mid]:
            hi = mid - 1
        else:
            lo = mid + 1

    return -1
