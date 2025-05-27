
def sieve(max):
    is_prime = [True] * (max + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(max ** 0.5) + 1):
        if is_prime[p]:
            for i in range(p * p, max + 1, p):
                is_prime[i] = False
    return [p for p in range(2, max + 1) if is_prime[p]]

"""
Sieve of Eratosthenes
prime-sieve

Input:
    max: A positive int representing an upper bound.

Output:
    A list containing all primes up to and including max
"""
