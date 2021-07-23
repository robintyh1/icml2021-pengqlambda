import numpy as np
import os
import tensorflow as tf
import gym
import time
import core
from core import get_vars
import wrappers
from spinup.utils.logx import EpochLogger
import gym


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """
    def __init__(self, obs_dim, act_dim, size, nstep):
        self.obs1_bufs, self.obs2_bufs = [], []
        self.acts_bufs, self.rews_bufs, self.done_bufs = [], [], []
        for i in range(nstep):
            self.obs1_bufs.append(np.zeros([size, obs_dim], dtype=np.float32))
            self.obs2_bufs.append(np.zeros([size, obs_dim], dtype=np.float32))
            self.acts_bufs.append(np.zeros([size, act_dim], dtype=np.float32))
            self.rews_bufs.append(np.zeros(size, dtype=np.float32))
            self.done_bufs.append(np.zeros(size, dtype=np.float32))
        self.nstep = nstep
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, acts, rews, next_obs, done):
        assert len(obs) == len(next_obs) == len(acts) == len(rews) == len(done) == self.nstep
        for i in range(self.nstep):
            self.obs1_bufs[i][self.ptr] = obs[i]
            self.obs2_bufs[i][self.ptr] = next_obs[i]
            self.acts_bufs[i][self.ptr] = acts[i]
            self.rews_bufs[i][self.ptr] = rews[i]
            self.done_bufs[i][self.ptr] = done[i]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        info = {}
        for i in range(self.nstep):
            info['obs1_{}'.format(i)] = self.obs1_bufs[i][idxs]
            info['obs2_{}'.format(i)] = self.obs2_bufs[i][idxs]
            info['done_{}'.format(i)] = self.done_bufs[i][idxs]
            info['acts_{}'.format(i)] = self.acts_bufs[i][idxs]
            info['rews_{}'.format(i)] = self.rews_bufs[i][idxs]
        return info


def td3(env_fn, env_fn_test, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, nstep=None, lambda_=None):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to td3_peng.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
            
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        nstep (int): The bootstrapping horizon of n-step. Note that even in Peng's Q
            where the operator is defined with infinite horizon, in practice we still
            compute updates based on finite partial trajectory of length n

        lambda_ (float): The lambda parameter used for trading-off fixed point bias
            and contraction rate, defined in Peng's Q operator
    """
    logger = EpochLogger(**logger_kwargs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn_test()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, d_ph = core.placeholders(obs_dim, act_dim, None)

    # placeholders for retrace
    if nstep >= 2:
        x_phs, a_phs, r_phs = [tf.placeholder(tf.float32, [None, obs_dim]) for _ in range(nstep-1)], \
                              [tf.placeholder(tf.float32, [None, act_dim]) for _ in range(nstep-1)], \
                              [tf.placeholder(tf.float32, [None]) for _ in range(nstep-1)]
    else:
        raise NotImplementedError

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _  = actor_critic(x_phs[-1], a_phs[-1], **ac_kwargs)
    
    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ_pi_final, q2_targ_pi_final, _ = actor_critic(x_phs[-1], a2, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, nstep=nstep)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # ====== Below we compute q values for two networls because we adopt the double Q-learning trick ======
    # first q value
    q1_lambda = tf.stop_gradient(r_phs[-1] + gamma*(1-d_ph)*q1_targ_pi_final)

    # recursive update
    for i in range(2, nstep):
        with tf.variable_scope('target', reuse=True):
            pi_targ_i, q1_targ_a_i, _, _  = actor_critic(x_phs[-i], a_phs[-i], **ac_kwargs)
            _, q1_targ_pi_i, _, _ = actor_critic(x_phs[-i], pi_targ_i, **ac_kwargs)
        # recurvively compute the target values
        q1_lambda = r_phs[-i] + gamma * (1-lambda_) * q1_targ_pi_i + gamma * lambda_ * q1_lambda

    # second q value
    q2_lambda = tf.stop_gradient(r_phs[-1] + gamma*(1-d_ph)*q2_targ_pi_final)

    # recursive update
    for i in range(2, nstep):
        with tf.variable_scope('target', reuse=True):
            pi_targ_i, _, q2_targ_a_i, _  = actor_critic(x_phs[-i], a_phs[-i], **ac_kwargs)
            _, _, q2_targ_pi_i, _ = actor_critic(x_phs[-i], pi_targ_i, **ac_kwargs)
        # recurvively compute the target values
        q2_lambda = r_phs[-i] + gamma * (1-lambda_) * q2_targ_pi_i + gamma * lambda_ * q2_lambda

    # Combine two q values
    backup = tf.stop_gradient(tf.minimum(q1_lambda, q2_lambda))

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ep_rets = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            ep_rets.append(ep_ret)
        return ep_rets

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # record training
    ep_ret_record = []
    time_step_record = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        if 'nstep_data_{}'.format(nstep) in info.keys():
            step_data = info['nstep_data_{}'.format(nstep)]
            replay_buffer.store(*step_data)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # Some additional processing so that we can save the transition data
            # properly in the n-stepn buffer
            info = env.get_last_info()
            # record last time step
            if 'nstep_data_{}'.format(nstep) in info.keys():
                info['nstep_data_{}'.format(nstep)][-1][-1] = d # done signal
            # Store experience to replay buffer
            if 'nstep_data_{}'.format(nstep) in info.keys():
                step_data = info['nstep_data_{}'.format(nstep)]
                replay_buffer.store(*step_data)    
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1_0'],
                             a_ph: batch['acts_0'],
                             d_ph: batch['done_{}'.format(nstep-1)]}
                for i in range(nstep-1):
                    feed_dict_i = {x_phs[i]: batch['obs1_{}'.format(i+1)],
                                   a_phs[i]: batch['acts_{}'.format(i+1)],
                                   r_phs[i]: batch['rews_{}'.format(i)],
                                }
                    feed_dict.update(feed_dict_i)
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            ep_rets = test_agent()
            ep_ret_record.append(np.mean(ep_rets))
            time_step_record.append(t)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # save the recorded time step and tested returns
            np.save(logger_kwargs['output_dir'] + '/ep_rets', ep_ret_record)
            np.save(logger_kwargs['output_dir'] + '/timesteps', time_step_record)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v1')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--nstep', type=int, default=1)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=0.0)
    parser.add_argument('--logdir', type=str)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    env_name = args.env
    logdir = os.path.join(
        args.logdir,
        'td3_peng_{}'.format(env_name),
        'seed_{}nstep_{}gamma_{}hid_{}l_{}lambda_{}delay_{}'.format(
            args.seed,
            args.nstep,
            args.gamma,
            args.hid,
            args.l,
            args.lambda_,
            args.delay
        )
    )
    logger_kwargs['output_dir'] = logdir

    # We augment the nstep input by 1, simply because later in the function call td3
    # we assume the nstep parameter is effectively 'nstep+1'
    nstep = args.nstep + 1

    def env_fn():
        env = gym.make(args.env)
        env = wrappers.DelayedRewardEnv(env, nstep=args.delay)
        return wrappers.NstepDataWrapper(env, max_nstep=nstep, gamma=args.gamma, all_nstep=[nstep])

    def env_fn_test():
        return gym.make(args.env)

    td3(
        env_fn,
        env_fn_test,
        actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        nstep=nstep,
        lambda_=args.lambda_
    )
