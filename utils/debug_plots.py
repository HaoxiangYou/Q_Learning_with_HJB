import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Tuple
from flax.core.frozen_dict import FrozenDict
from controller.vhjb import ValueFunctionApproximator
from utils.utils import keep_first_element

def visualize_loss_landscope(model:ValueFunctionApproximator, model_params:FrozenDict, model_states: FrozenDict, key: jax.random.PRNGKeyArray,                            
                            xs:Union[np.ndarray, jnp.ndarray], loss_function: Callable,
                            xmin=-1, xmax=1, xnum=101, ymin=-1, ymax=1, ynum=101,
                            *args, **kwags):
    """
    Draw loss landcope around the given model params. 
    The directions are chosen randomly and normalized according to:
    https://arxiv.org/pdf/1712.09913.pdf


    Args:
        model:
            a nn object, serve to generate random directions.
            it should have same architecture as the model used in loss function
        model_params: network params
        model_states: network states such as BN statistics
        key: random key
        xs: a batch of input data
        loss_function: 
            a callable, takes the model params, model states, xs and addtional args, and output the loss (or with some additional infos).
            we assume the nn model(and its mode) is embedded in these function, e.g. termination_loss and hjb_loss in VHJBController
            TODO issolate the model dependency
        xmin, xmax, xnum: ints for the range of x axis
        ymin, ymax, ynum: ints for the range of y axis
        *args, **kwags: additional params for loss function

    Returns:

    """
    def noramlize_direction_with_weight(direction:jnp.ndarray, weight:jnp.ndarray):
        if direction.ndim <= 1:
            # fill bias term with zeros
            direction = direction.at[:].set(0)
        else:
            for index in range(direction.shape[1]):
                # TODO normalize only weights for next feature, or normalize the entire weights for next layers
                direction = direction.at[:,index].mul(jnp.linalg.norm(weight[:, index]) / (jnp.linalg.norm(direction[:, index])))
        
        return direction
    
    # accelerate with jit function
    loss_function_jit = jax.jit(keep_first_element(loss_function))

    key_1, key_2 = jax.random.split(key)
    _, direction_1 = model.init(key_1, xs, train=True).pop("params")
    _, direction_2 = model.init(key_2, xs, train=True).pop("params")
    direction_1 = jax.tree_util.tree_map(noramlize_direction_with_weight, direction_1, model_params)
    direction_2 = jax.tree_util.tree_map(noramlize_direction_with_weight, direction_2, model_params)

    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)

    X, Y = np.meshgrid(x,y)
    losses = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            delta_params = jax.tree_util.tree_map(lambda d1, d2: d1 * X[i,j] + d2 * Y[i,j], direction_1, direction_2)
            params = jax.tree_util.tree_map(lambda nominal_params, delta_params: nominal_params + delta_params, model_params, delta_params)
            losses[i,j] = loss_function_jit(params, model_states, xs, *args, **kwags)
    
    try:
        loss_name = loss_function.__name__
    except:
        try:
            loss_name = loss_function.__func__.__name__
        except:
            loss_name = loss_function.func.__name__
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, losses)
    ax.set_zlabel(f"{loss_name}")
    ax.set_title(f"{loss_name} landscape")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS = ax.contour(X, Y, losses)
    ax.clabel(CS, inline=True)
    ax.set_title(f"{loss_name} contour")

def visualize_value_landscope(model:ValueFunctionApproximator, model_params:FrozenDict, model_states: FrozenDict, value_functon: Callable,
                        x_0: np.ndarray, x_direction: np.ndarray, y_direction: np.ndarray, x_range=np.linspace(-1, 1, 51), y_range=np.linspace(-1, 1, 51)):
    """
    This function help to draw value function landscope

    Args:
        model: nn network object
        model_params: params for learned nn networks
        model_states: params for neural network states such as bn statistics
        value function: a callable to calculate ground truth, e.g. function computed from value iteration or simply x.T @ P @ x for lqr
        x_0: the center point to visualize value function
        x_direction: x axis direction
        y_direction: y axis direction
        x_range: the ranges for x direction
        y_range: the range for y direction

    Returns:
    
    """
    

    quick_apply = jax.jit(model.apply, static_argnames=["train"])
    x_0 = x_0.astype(np.float32)

    X, Y = np.meshgrid(x_range, y_range)
    v_learned = np.zeros_like(X)
    v_gt = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            x = np.copy(x_0)
            x += X[i,j] * x_direction
            x += Y[i,j] * y_direction
            v_learned[i,j] = quick_apply({"params":model_params, **model_states}, x, train=False)
            v_gt[i,j] = value_functon(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, v_gt)
    ax.set_xlabel(f"direction: {np.round(x_direction, 2)}")
    ax.set_ylabel(f"direction: {np.round(y_direction, 2)}")
    ax.set_zlabel('value')
    plt.title("\"Ground truth\" value function landscope")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, v_learned)
    ax.set_xlabel(f"direction: {np.round(x_direction, 2)}")
    ax.set_ylabel(f"direction: {np.round(y_direction, 2)}")
    ax.set_zlabel('value')
    plt.title("Learned value function landscope")

def visualize_value_landscope_for_lqr(model:ValueFunctionApproximator, model_params:FrozenDict, model_states: FrozenDict, P: np.ndarray, 
                                      x_0=None, x_range=np.linspace(-1, 1, 51), y_range=np.linspace(-1, 1, 51)):
    """
    This function will draw value landscope for both learned function and lqr solutions.
    
    The directions are two eigenvectors with biggest eigenvalues of cost-to-go matrix P.
    """

    if x_0 is None:
        x_0 = np.zeros(P.shape[0])

    eigvals, eigvectors = np.linalg.eig(P)
    sorted_indices = np.flip(np.argsort(eigvals))

    x_direction = eigvectors[:, sorted_indices[0]]
    x_eigval = eigvals[sorted_indices[0]]
    y_direction = eigvectors[:, sorted_indices[1]]
    y_eigval = eigvals[sorted_indices[1]]

    # normalize the direction based on eigvalues
    x_direction = x_direction / x_eigval **0.5
    y_direction = y_direction / y_eigval **0.5

    lqr_value_function = lambda x: x.T @ P @ x

    visualize_value_landscope(model, model_params, model_states, lqr_value_function, x_0, x_direction, y_direction, x_range, y_range)