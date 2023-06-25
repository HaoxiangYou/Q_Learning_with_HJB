import jax
import numpy as np
import scipy.linalg
from itertools import permutations
from flax.core.frozen_dict import FrozenDict
from controller.vhjb import VHJBController
from typing import Tuple

def get_equivalent_matrix_multiplication_for_fully_connected_nn(x: np.ndarray, model_params: FrozenDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an input x and neural network params find a particular weight matrix and bias,
    such that the neural network output = W.T x + b
    
    Currently we assume the neural network achitecture is 
    only fully connected layer followed by relu, except the last layer.
    """

    weight = np.eye(x.shape[0])
    bias = np.zeros_like(x)
    feature = x

    for i, (layer_name, layer_params) in enumerate(model_params.items()):
        layer_weight = np.array(layer_params['kernel'])
        
        if 'bias' in layer_params:
            layer_bias = np.array(layer_params['bias'])
        else:
            layer_bias = np.zeros(layer_weight.shape[1])
        
        feature = layer_weight.T @ feature + layer_bias
        
        if i != len(model_params) - 1:
            negative_indices = np.where(feature < 0)[0]
            layer_weight[:, negative_indices] = 0
            layer_bias[negative_indices] = 0
            feature[negative_indices] = 0
        weight = weight @ layer_weight
        bias = layer_weight.T @ bias + layer_bias

    return weight, bias 

def local_optimal_x(x: np.ndarray, nn_policy: VHJBController, max_iter=10, lr=1e-1, verbose=True, newton_method=True):
    """
    This function aim to find a local minimal value around x,
    the local minimal can further served for debug purpose. 
    """
    nn_policy.train_mode = False

    quick_hess_fn = jax.jit(jax.hessian(nn_policy.value_function_approximator.apply, argnums=1), static_argnames=["train"])
    
    for i in range(max_iter):
        value = nn_policy.value_function_approximator.apply({"params":nn_policy.model_params, **nn_policy.model_states}, x, train=False)
        u, v_gradient, updated_states = nn_policy.get_control_efforts(nn_policy.model_params, nn_policy.model_states, x)
        hess = quick_hess_fn({"params":nn_policy.model_params, **nn_policy.model_states}, x, train=False)
        if verbose:
            print(f"iter:{i}, x: {x}, value:{value:.5f}, \n u:{u} v_gradient:{v_gradient} \n hess:{hess}")
        if np.linalg.det(hess) > 0 and newton_method:
            x -= lr * jax.numpy.linalg.inv(hess) @ v_gradient
        else:
            x -= lr * v_gradient
    return x

def check_controllability(A:np.ndarray, B:np.ndarray, verbose=False):
    """
    Check controllablity for linear system using kalman rank condition
    """
    dim = A.shape[0]
    controllablity_matrix = B
    for i in range(1, dim):
        controllablity_matrix = np.hstack([controllablity_matrix, np.linalg.matrix_power(A, i) @ B])
    if np.linalg.matrix_rank(controllablity_matrix) == dim:
        controllable = True
    else:
        controllable = False
    if verbose:
        return (controllable, controllablity_matrix)
    else:
        return controllable

def check_hjb_condition_for_lqr(P: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, threshold=1e-5, verbose=False):
    """
    This function help to check whether a particular P statisfy HJB equations derivative for lqr: 
        2P^T A - P^T B R^-1 B^T P + Q = 0 or,
        A^T P + P^T A - P^T B R^-1 B^T P + Q = 0 or,
        2 A^T P - P^T B R^-1 B^T P + Q = 0
    Note:
        this functions are not exactly same as algebraic equations:
        A^T P + P A - P B R^-1 B^T P + Q = 0
        through a sysmetric solution P may statisfy both are and hjb equations
    """

    cond_1 = 2 * P.T @ A - P.T @ B @ np.linalg.inv(R) @ B.T @ P + Q
    cond_2 = A.T @ P + P.T @ A - P.T @ B @ np.linalg.inv(R) @ B.T @ P + Q
    cond_3 = 2 * A.T @ P - P.T @ B @ np.linalg.inv(R) @ B.T @ P + Q

    if np.max(np.abs(cond_1)) < threshold or np.max(np.abs(cond_2)) < threshold or np.max(np.abs(cond_3)) < threshold:
        satisfied = True
    else:
        satisfied = False
    
    if verbose:
        return (satisfied, cond_1, cond_2, cond_3)
    else:
        return satisfied

def solve_continuous_are(A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray):
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
    """
    dim = Q.shape[0]
    S = B @ np.linalg.inv(R) @ B.T
    H = np.vstack(
        [np.hstack([A, -S]),
        np.hstack([-Q, -A.T])]
    )
    
    T, Z, _ = scipy.linalg.schur(H, sort="lhp")

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