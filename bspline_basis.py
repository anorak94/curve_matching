import numpy as np


def memo(f):
    # Peter Norvig's
    """Memoize the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}

    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args can't be a dict key
            return f(*args)
    _f.cache = cache
    return _f


def bspline_basis(c, n, degree):
    """ bspline basis function
        c        = number of control points.
        n        = number of points on the curve.
        degree   = curve degree
    """
    # Create knot vector and a range of samples on the curve
    kv = np.array([0] * degree + [i for i in range(c - degree + 1)] +
                  [c - degree] * degree, dtype='int')  # knot vector
    u = np.linspace(0, c - degree, n)  # samples range

    # Cox - DeBoor recursive function to calculate basis
    @memo
    def coxDeBoor(k, d):
        # Test for end conditions
        if (d == 0):
            return ((u - kv[k] >= 0) & (u - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((u - kv[k]) / denom1) * coxDeBoor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(u - kv[k + d + 1]) / denom2) * coxDeBoor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    b = np.column_stack([coxDeBoor(k, degree) for k in range(c)])
    b[n - 1][-1] = 1

    return b