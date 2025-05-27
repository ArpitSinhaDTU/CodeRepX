
def next_palindrome(digit_list):
    n = len(digit_list)
    mid = n // 2
    i = mid - 1
    j = mid + 1 if n % 2 == 0 else mid
    
    while i >= 0 and digit_list[i] == digit_list[j]:
        i -= 1
        j += 1

    if i < 0 or digit_list[i] < digit_list[j]:
        i = mid - 1
        j = mid + 1 if n % 2 == 0 else mid
        carry = 1
        if n % 2 == 0:
            j = mid
            i = mid -1

        while i >= 0:
            digit_list[i] += carry
            carry = digit_list[i] // 10
            digit_list[i] %= 10
            digit_list[j] = digit_list[i]
            i -= 1
            j += 1
        if carry:
            digit_list = [1] + digit_list
            digit_list[n] = 1
            return digit_list
        else:
            i = mid - 1
            j = mid + 1 if n % 2 == 0 else mid
            while i >= 0:
                digit_list[j] = digit_list[i]
                i -= 1
                j += 1
            return digit_list
    else:
        i = mid - 1
        j = mid + 1 if n % 2 == 0 else mid
        while i >= 0:
            digit_list[j] = digit_list[i]
            i -= 1
            j += 1
        return digit_list
"""
Finds the next palindromic integer when given the current integer
Integers are stored as arrays of base 10 digits from most significant to least significant

Input:
    digit_list: An array representing the current palindrome

Output:
    An array which represents the next palindrome

Preconditions:
    The initial input array represents a palindrome

Example
    >>> next_palindrome([1,4,9,4,1])
    [1,5,0,5,1]
"""
