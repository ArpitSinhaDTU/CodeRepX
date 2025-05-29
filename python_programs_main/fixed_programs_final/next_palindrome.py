def next_palindrome(digit_list):
    n = len(digit_list)
    mid = n // 2
    
    # Handle edge case: empty list
    if n == 0:
        return [1]

    # Create a copy to avoid modifying the original list directly
    digit_list = digit_list[:]

    # 1. Copy the left half to the right half
    for i in range(mid):
        digit_list[n - 1 - i] = digit_list[i]

    # 2. Check if the new number is greater than the original number
    if digit_list > list(digit_list):
        return digit_list

    # 3. If not, increment the middle digit(s)
    i = mid - 1
    j = mid if n % 2 == 0 else mid

    while i >= 0 and digit_list[i] == 9:
        digit_list[i] = 0
        digit_list[j] = 0
        i -= 1
        j += 1

    if i < 0:
        # All digits were 9, so we need to add a digit
        return [1] + [0] * n + [1]
    else:
        digit_list[i] += 1
        digit_list[j] += 1
        # Copy the left half to the right half after incrementing
        for k in range(mid):
            digit_list[n - 1 - k] = digit_list[k]
        return digit_list
