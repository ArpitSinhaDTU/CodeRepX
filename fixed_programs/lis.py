
def lis(arr):
    ends = {}
    longest = 0

    for i, val in enumerate(arr):
        prefix_lengths = [length for length in ends if arr[ends[length]] < val]
        length = max(prefix_lengths) if prefix_lengths else 0

        if length == longest:
            ends[length + 1] = i
            longest = length + 1
        elif length + 1 in ends and val < arr[ends[length + 1]]:
            ends[length + 1] = i

    return longest
