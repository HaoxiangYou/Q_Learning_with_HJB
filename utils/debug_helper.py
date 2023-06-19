import jax
import numpy as np
from controller.vhjb import VHJBController

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