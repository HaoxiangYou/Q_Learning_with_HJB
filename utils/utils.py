import numpy as np
import scipy.linalg
from itertools import permutations
from typing import List, Union

# For torch Dataloader to collate numpy
def np_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def keep_first_element(func):
    """
    Decorator that modifies the return value of a function
    to keep only the first element if it's a tuple.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        else:
            return result
    return wrapper

def solve_continuous_are(A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray, multiple_sol=False) -> Union[List[np.ndarray], np.ndarray]:
    """
    This functions gives a set of possible solutions for ARE: 
        A^T P + P A - P B R^-1 B^T P + Q = 0
    The solution is obtained by Schur decomposition of Hamitonian matrix.
    A good reference is available through: 
        https://scholarworks.gsu.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1045&context=math_theses
    
    Note:
        Not all possible solutions that satisify ARE may be return. 
        For example, let's assume A,B,Q,R are all 2X2 identity matrices,
        then the ARE becomes: 2P^T - P^T P + I = 0.
        assume the P is symmetric: 
            [a , b
            b, c]
        To statisfy ARE, the following equations needs to be statisfied:
            (2 - (a+c)) * b = 0
            2a - a^2 - b^2 + 1 = 0
            2c - c^2 - b^2 + 1 = 0
        There are four solutions if b =0, 
        and there are infinity solutions when a+c = 2 (the corresponding b^2 = 2a - a^2 + 1 = 2c - c^2 +1)

        Not does the all solutions returns statisfy HJB equations for LQR.
        Derivated from HJB equations, one of following must be statisfied:
            2P^T A - P^T B R^-1 B^T P + Q = 0
            A^T P + P^T A - P^T B R^-1 B^T P + Q = 0
            2 A^T P - P^T B R^-1 B^T P + Q = 0
        However, the solutions only statisfy:
            A^T P + P A - P B R^-1 B^T P + Q = 0

    Args:
        A, B: system dynamics
        Q, R: cost function coeff
        multiple_sol: a bool flag, whether to return other possible solutions (might not be positive definite matrix), default: False
    Returns:
        P or Ps: P matrices that satisfy A^T P + P A - P B R^-1 B^T P + Q = 0. If multiple_sol=False, then will only return the unique positive definite matrices
    """
    dim = Q.shape[0]
    S = B @ np.linalg.inv(R) @ B.T
    H = np.vstack(
        [np.hstack([A, -S]),
        np.hstack([-Q, -A.T])]
    )
    
    T, Z, _ = scipy.linalg.schur(H, sort="lhp")

    if not multiple_sol:
        p1 = Z[:dim, np.arange(dim)]
        p2 = Z[dim:, np.arange(dim)]
        P = p2 @ np.linalg.inv(p1)
        return P 

    else:
        Ps = []

        combinations_indices = permutations(np.arange(2 * dim), dim)

        for index in combinations_indices:
            p1 = Z[:dim, index]
            p2 = Z[dim:, index]
            try:
                P = np.round(p2 @ np.linalg.inv(p1), decimals=8)
                is_unique = True
                for matrix in Ps:
                    if np.array_equal(matrix, P):
                        is_unique = False
                        break
                if is_unique:
                    Ps.append(P)
            except:
                print("noninvertible p1 detected,")
                continue

        return Ps
