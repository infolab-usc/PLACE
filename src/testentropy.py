import numpy as np

from Main_ import shanon_entropy_list, cut_list
import random

N = 20
low = 1
high = 100

count = 0
while False:
    count = count + 1
    x = random.sample(range(low, high), N)
    x_c = cut_list(x, random.randint(low, N))

    entropy_x = shanon_entropy_list(x)
    entropy_x_c = shanon_entropy_list(x_c)
    if entropy_x_c < entropy_x:
        print "----"
        print x, entropy_x
        print x_c, entropy_x_c

    if count % 10000 == 0:
        print count

def logbasechange(a,b):
    """
    There is a one-to-one transformation of the entropy value from
    a log base b to a log base a :

    H_{b}(X)=log_{b}(a)[H_{a}(X)]

    Returns
    -------
    log_{b}(a)
    """
    return np.log(b)/np.log(a)

def shannonentropy(px, logbase=2):
    """
    This is Shannon's entropy

    Parameters
    -----------
    logbase, int or np.e
        The base of the log
    px : 1d or 2d array_like
        Can be a discrete probability distribution, a 2d joint distribution,
        or a sequence of probabilities.

    Returns
    -----
    For log base 2 (bits) given a discrete distribution
        H(p) = sum(px * log2(1/px) = -sum(pk*log2(px)) = E[log2(1/p(X))]

    For log base 2 (bits) given a joint distribution
        H(px,py) = -sum_{k,j}*w_{kj}log2(w_{kj})

    Notes
    -----
    shannonentropy(0) is defined as 0
    """
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError, "px does not define proper distribution"
    entropy = -np.sum(np.nan_to_num(px * np.log2(px)))
    if logbase != 2:
        return logbasechange(2, logbase) * entropy
    else:
        return entropy


def renyientropy(px, alpha=1, logbase=2, measure='R'):
    """
    Renyi's generalized entropy

    Parameters
    ----------
    px : array-like
        Discrete probability distribution of random variable X.  Note that
        px is assumed to be a proper probability distribution.
    logbase : int or np.e, optional
        Default is 2 (bits)
    alpha : float or inf
        The order of the entropy.  The default is 1, which in the limit
        is just Shannon's entropy.  2 is Renyi (Collision) entropy.  If
        the string "inf" or numpy.inf is specified the min-entropy is returned.
    measure : str, optional
        The type of entropy measure desired.  'R' returns Renyi entropy
        measure.  'T' returns the Tsallis entropy measure.

    Returns
    -------
    1/(1-alpha)*log(sum(px**alpha))

    In the limit as alpha -> 1, Shannon's entropy is returned.

    In the limit as alpha -> inf, min-entropy is returned.
    """
    alpha = float(alpha)
    if alpha == 1:
        genent = shannonentropy(px)
        if logbase != 2:
            return logbasechange(2, logbase) * genent
        return genent
    elif 'inf' in str(alpha).lower() or alpha == np.inf:
        return -np.log(np.max(px))

    # gets here if alpha != (1 or inf)
    px = px ** alpha
    genent = np.log(px.sum())
    if logbase == 2:
        return 1 / (1 - alpha) * genent
    else:
        return 1 / (1 - alpha) * logbasechange(2, logbase) * genent

arr1 = np.array([0.5, 0.05, 0.45])
arr2 = np.array([1.0/6,1.0/4,1.0/6,1.0/6,1.0/4])
A = 1
print np.exp(shannonentropy(arr1, logbase=np.e)), np.exp(shannonentropy(arr2, logbase=np.e)), np.exp(shannonentropy(arr1, logbase=np.e))/np.exp(shannonentropy(arr2, logbase=np.e))
print ''
print np.exp(renyientropy(arr1, alpha=0.5, logbase=np.e)), np.exp(renyientropy(arr2, alpha=A, logbase=np.e)), np.exp(renyientropy(arr1, alpha=A, logbase=np.e))/np.exp(renyientropy(arr2, alpha=A, logbase=np.e))

import math

x = 1.1
while x < 10000:
    print abs(-math.log(0.5**x  + 0.05**x + (0.45)**x))/(x-1)
    x += 1