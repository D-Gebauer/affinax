import jax
import jax.numpy as jnp
from tqdm import trange
import time

def affine_sample(log_prob, n_steps, p0, args=[], key=None, progressbar=True):
    
    if not key:
        key = jax.random.PRNGKey(time.time_ns())
    
    # split the current state
    current_state1, current_state2 = p0
    
    # pull out the number of parameters and walkers
    n_walkers, n_params = current_state1.shape

    # initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1, *args)
    logp_current2 = log_prob(current_state2, *args)
    logp_current1 = jnp.where(jnp.isnan(logp_current1), jnp.ones_like(logp_current1)*jnp.log(0.), logp_current1)
    logp_current2 = jnp.where(jnp.isnan(logp_current2), jnp.ones_like(logp_current2)*jnp.log(0.), logp_current2)

    # holder for the whole chain
    chain = [jnp.expand_dims(jnp.concatenate([current_state1, current_state2], axis=0), axis=0)]
    
    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for epoch in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = jax.lax.gather(current_state2, jax.random.randint(key, (n_walkers), 0, n_walkers))
        #partners1 = current_state2.at[jax.random.randint(0, n_walkers, n_walkers)].get()
        z1 = 0.5*(jax.random.uniform(key, (n_walkers))+1)**2
        proposed_state1 = partners1 + jnp.transpose(z1*jnp.transpose(current_state1 - partners1))

        # target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1, *args)
        logp_proposed1 = jnp.where(jnp.isnan(logp_proposed1), jnp.ones_like(logp_proposed1)*jnp.log(0.), logp_proposed1)

        # acceptance probability
        p_accept1 = jnp.min(jnp.ones(n_walkers), z1**(n_params-1)*jnp.exp(logp_proposed1 - logp_current1) )

        # accept or not
        accept1_ = (jax.random.uniform(key, (n_walkers)) <= p_accept1)
        accept1 = jnp.astype(accept1_, jnp.float32)

        # update the state
        current_state1 = jnp.transpose( jnp.transpose(current_state1)*(1-accept1) + jnp.transpose(proposed_state1)*accept1)
        logp_current1 = jnp.where(accept1_, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals
        partners2 = jax.lax.gather(current_state2, jax.random.randint(key, (n_walkers), 0, n_walkers))
        z2 = 0.5*(jax.random.uniform(key, (n_walkers))+1)**2
        proposed_state2 = partners2 + jnp.transpose(z2*jnp.transpose(current_state2 - partners2))

        # target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2, *args)
        logp_proposed2 = jnp.where(jnp.isnan(logp_proposed2), jnp.ones_like(logp_proposed2)*jnp.log(0.), logp_proposed2)

        # acceptance probability
        p_accept2 = jnp.min(jnp.ones(n_walkers), z2**(n_params-1)*jnp.exp(logp_proposed2 - logp_current2) )

        # accept or not
        accept2_ = (jax.random.uniform(key, (n_walkers)) <= p_accept2)
        accept2 = jnp.astype(accept2_, jnp.float32)

        # update the state
        current_state2 = jnp.transpose( jnp.transpose(current_state2)*(1-accept2) + jnp.transpose(proposed_state2)*accept2)
        logp_current2 = jnp.where(accept2_, logp_proposed2, logp_current2)

        # append to chain
        chain.append(jnp.expand_dims(jnp.concatenate([current_state1, current_state2], axis=0), axis=0))

    # stack up the chain and return    
    return jnp.concatenate(chain, axis=0)
