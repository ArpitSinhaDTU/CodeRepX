
def subsequences(a, b, k):
    if k == 0:
        return [[]]  # Return a list containing an empty list

    if b - a + 1 < k:
        return []  # Not enough numbers to form a subsequence of length k

    ret = []
    for i in range(a, b + 1):
        remaining_length = b - i + 1
        if remaining_length >= k:
            for rest in subsequences(i + 1, b, k - 1):
                ret.append([i] + rest)

    return ret


"""
Subsequences

 
Input:
    a: An int
    b: An int
    k: A positive int

Output:
    A list of all length-k ascending sequences of ints in range(a, b)

Example:
    >>> subsequences(a=1, b=5, k=3)
    [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
"""
