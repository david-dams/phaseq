import jax
import jax.numpy as jnp
from jax import random, grad
from jax.example_libraries import optimizers

from jax.scipy.special import gammainc, gamma, factorial

from jax import random as jran

n_features, n_targets = 1, 1
SEED = 1291
ran_key = jran.PRNGKey(SEED)


from jax.example_libraries import stax

net_init, net_apply = stax.serial(
    stax.Dense(8), stax.Selu,
    stax.Dense(8), stax.Selu,
    stax.Dense(16), stax.Selu,
    stax.Dense(16), stax.Selu,
    stax.Dense(8), stax.Selu,
    stax.Dense(n_targets),
)

ran_key, net_init_key = jran.split(ran_key)
out_shape, net_params = net_init(net_init_key, input_shape=(-1, n_features))

print('First layer shapes: ',jnp.shape(net_params[0][0]),jnp.shape(net_params[0][1]))
print('Final layer shapes: ',jnp.shape(net_params[-1][0]),jnp.shape(net_params[-1][1]))

from jax.example_libraries import optimizers as jax_opt

opt_init, opt_update, get_params = jax_opt.adam(1e-3)

opt_state = opt_init(net_params)

from jax import jit as jjit
from jax import value_and_grad
from jax import numpy as jnp
from jax.nn import sigmoid

@jjit
def target_function(arg):
    index = 1
    return gammainc(index+0.5, arg) * gamma(index+0.5) * 0.5 * jnp.pow(arg,-index-0.5) 


@jjit
def mse_loss(params, loss_data):
    X_tbatch, targets = loss_data
    preds = net_apply(params, X_tbatch)
    diff = preds - targets 
    return jnp.mean(diff * diff)


@jjit
def train_step(step_i, opt_state, loss_data):
    net_params = get_params(opt_state)
    loss, grads = value_and_grad(mse_loss, argnums=0)(net_params, loss_data)
    return loss, opt_update(step_i, grads, opt_state)

from time import time

batch_size = 50
num_batches = 2000
loss_history = []
start = time()
for ibatch in range(num_batches):
    ran_key, batch_key = jran.split(ran_key)
    X_train = jran.uniform(batch_key, shape=(batch_size, n_features), minval=1, maxval=10)
    targets = target_function(X_train)
    loss_data = X_train, targets
    
    loss, opt_state = train_step(ibatch, opt_state, loss_data)
    loss_history.append(float(loss))

end = time()
msg = "training time for {0} iterations = {1:.1f} seconds"
print(msg.format(num_batches, end-start))

import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)
# __=ax.plot(jnp.log10(jnp.array(loss_history)))
# xlabel = ax.set_xlabel(r'${\rm step\ number}$')
# ylabel = ax.set_ylabel(r'$\log_{10}{\rm loss}$')
# title = ax.set_title(r'${\rm training\ history}$')
# plt.show()


fig, ax = plt.subplots(1, 1)
# xlim = ax.set_xlim(-2, 2)
# ylim = ax.set_ylim(-0.1, 1.1)

_x = jnp.linspace(1, 10, 1000)
__=ax.plot(_x, target_function(_x), color='k', label=r'${\rm Boys}$')

_y = net_apply(get_params(opt_state), _x.reshape((-1, 1))).flatten()
__=ax.plot(_x, _y, '--', label=r'${\rm network\ prediction}$')

xlabel = ax.set_xlabel(r'$x$')
ylabel = ax.set_ylabel(r'$y$')
title = ax.set_title(r'${\rm training\ validation}$')
leg = ax.legend()
plt.savefig("boys.pdf")
