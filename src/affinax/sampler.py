# Adapted to JAX from github.com/justinalsing/affine

import jax
import jax.numpy as jnp
from tqdm import trange
import time
from functools import partial

# jitted version of one step of the affine invariant sampler
@partial(jax.jit, static_argnames=("log_prob", "n_walkers", "n_params"))
def affine_step(log_prob, key, current_state1, current_state2, n_walkers, n_params, *args):
    
    # first set of walkers:
    
    logp_current1 = log_prob(current_state1, *args)
    # proposals (z from arxiv.org/pdf/1202.3665)
    key, subkey = jax.random.split(key)
    partners1 = current_state2.at[jax.random.randint(subkey, (n_walkers), 0, n_walkers)].get()
    key, subkey = jax.random.split(key)
    z1 = 1/jnp.sqrt(jax.random.uniform(subkey, (n_walkers)) * 1.5 + 0.5)
    proposed_state1 = partners1 + jnp.transpose(z1*jnp.transpose(current_state1 - partners1))

    # target log prob at proposed points
    logp_proposed1 = log_prob(proposed_state1, *args)
    logp_proposed1 = jnp.where(jnp.isnan(logp_proposed1), jnp.ones_like(logp_proposed1)*jnp.log(0.), logp_proposed1)

    # acceptance probability
    p_accept1 = jnp.minimum(jnp.ones(n_walkers), z1**(n_params-1)*jnp.exp(logp_proposed1 - logp_current1) )

    # accept or not
    key, subkey = jax.random.split(key)
    accept1_ = (jax.random.uniform(subkey, (n_walkers)) <= p_accept1)
    accept1 = jnp.astype(accept1_, jnp.float32)

    # update the state
    current_state1 = jnp.transpose( jnp.transpose(current_state1)*(1-accept1) + jnp.transpose(proposed_state1)*accept1)
    logp_current1 = jnp.where(accept1_, logp_proposed1, logp_current1)

    # second set of walkers:
    
    logp_current2 = log_prob(current_state2, *args)

    # proposals (z from arxiv.org/pdf/1202.3665)
    key, subkey = jax.random.split(key)
    partners2 = current_state2.at[jax.random.randint(subkey, (n_walkers), 0, n_walkers)].get()
    key, subkey = jax.random.split(key)
    z2 = 1/jnp.sqrt(jax.random.uniform(subkey, (n_walkers)) * 1.5 + 0.5)
    proposed_state2 = partners2 + jnp.transpose(z2*jnp.transpose(current_state2 - partners2))

    # target log prob at proposed points
    logp_proposed2 = log_prob(proposed_state2, *args)
    logp_proposed2 = jnp.where(jnp.isnan(logp_proposed2), jnp.ones_like(logp_proposed2)*jnp.log(0.), logp_proposed2)

    # acceptance probability
    p_accept2 = jnp.minimum(jnp.ones(n_walkers), z2**(n_params-1)*jnp.exp(logp_proposed2 - logp_current2) )

    # accept or not
    key, subkey = jax.random.split(key)
    accept2_ = (jax.random.uniform(subkey, (n_walkers)) <= p_accept2)
    accept2 = jnp.astype(accept2_, jnp.float32)

    # update the state
    current_state2 = jnp.transpose( jnp.transpose(current_state2)*(1-accept2) + jnp.transpose(proposed_state2)*accept2)
    logp_current2 = jnp.where(accept2_, logp_proposed2, logp_current2)

    # return current state for both sets of walkers
    return jnp.concatenate([current_state1, current_state2], axis=0), key


def affine_sample(log_prob, n_steps, p0, args=[], key=None, progressbar=True):
    
    if key is None:
        key = jax.random.PRNGKey(time.time_ns())
    
    # split the current state in two sets of walkers
    current_state1, current_state2 = p0
    
    # infer number of parameters and walkers from state
    n_walkers, n_params = current_state1.shape

    # array to store whole chain
    chain = jnp.zeros([n_steps, 2*n_walkers, n_params])
    chain = chain.at[0].set(jnp.concatenate([current_state1, current_state2], axis=0))
    
    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for epoch in loop(1, n_steps):
        
        # generate new keys
        key, subkey = jax.random.split(key)
        
        # take a step
        state, key = affine_step(log_prob, subkey, chain[epoch-1, :n_walkers], chain[epoch-1, n_walkers:], n_walkers, n_params, *args)
        
        # append the new state to the chain
        chain = chain.at[epoch].set(state)
    
    # return the chain
    return chain

