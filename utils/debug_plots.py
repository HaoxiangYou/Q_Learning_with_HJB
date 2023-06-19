import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable
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