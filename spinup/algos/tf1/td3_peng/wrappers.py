import gym
import copy
from collections import deque
import numpy as np


class DelayedRewardEnv(gym.Wrapper):
    """
    A simple wrapper for creating env with delayed rewards
    """

	def __init__(self, env, nstep):
		super(DelayedRewardEnv, self).__init__(env)
		self.env = env
		self.nstep = nstep
		self.reset_buffer()

	def reset_buffer(self):
		self.rewards = []

	def reset(self):
		self.reset_buffer()
		o = self.env.reset()
		return o

	def step(self, action):
		o, r, done, info = self.env.step(action)
		# record
		self.rewards.append(r)
		if len(self.rewards) == self.nstep or done is True:
			r = np.sum(self.rewards)
			self.reset_buffer()
		else:
			r = 0.0
		return o, r, done, info


class NstepDataWrapper(gym.Wrapper):
    """
    A simple wrapper for recording n-step transition
    """

	def __init__(self, env, max_nstep, gamma, all_nstep=None):
		super(NstepDataWrapper, self).__init__(env)
		self.env = env
		self.max_nstep = max_nstep
		self.gamma = gamma
		if all_nstep is None:
			self.all_nstep = [1, max_nstep]
		else:
			assert isinstance(all_nstep, list) and max_nstep in all_nstep
			self.all_nstep = all_nstep
		self.reset_buffer()

	def reset_buffer(self):
		self.obs_all = []
		self.obs2_all = []
		self.acts_all = []
		self.rews_all = []
		self.dones_all = []
		for nstep in self.all_nstep:
			self.obs_all.append(deque(maxlen=nstep))
			self.obs2_all.append(deque(maxlen=nstep))
			self.acts_all.append(deque(maxlen=nstep))
			self.rews_all.append(deque(maxlen=nstep))
			self.dones_all.append(deque(maxlen=nstep))

	def reset(self):
		self.reset_buffer()
		o = self.env.reset()
		for obs in self.obs_all:
			obs.append(o)
		for done in self.dones_all:
			done.append(False)
		return o

	def get_info(self):
		info = {}
		for idx in range(len(self.all_nstep)):
			if len(self.obs2_all[idx]) == self.all_nstep[idx]:
				nstep_data = [self.obs_all[idx].copy(), self.acts_all[idx].copy(), self.rews_all[idx].copy(), self.obs2_all[idx].copy(), self.dones_all[idx].copy()]
				info.update({'nstep_data_{}'.format(self.all_nstep[idx]): copy.deepcopy(nstep_data)})
		return info

	def get_last_info(self):
		o = self.env.reset()
		action = self.env.action_space.sample()
		r = 0.0
		for idx in range(len(self.all_nstep)):
			self.obs2_all[idx].append(o)
			self.rews_all[idx].append(r)
			self.acts_all[idx].append(action)
		info = {}
		for idx in range(len(self.all_nstep)):
			if len(self.obs2_all[idx]) == self.all_nstep[idx]:
				nstep_data = [self.obs_all[idx].copy(), self.acts_all[idx].copy(), self.rews_all[idx].copy(), self.obs2_all[idx].copy(), self.dones_all[idx].copy()]
				info.update({'nstep_data_{}'.format(self.all_nstep[idx]): copy.deepcopy(nstep_data)})
		return info		

	def step(self, action):
		o, r, done, info = self.env.step(action)
		# record
		for idx in range(len(self.all_nstep)):
			self.obs2_all[idx].append(o)
			self.rews_all[idx].append(r)
			self.acts_all[idx].append(action)
		# add to info if necessary
		info = self.get_info()
		# record obs1
		for obs in self.obs_all:
			obs.append(o)
		for ds in self.dones_all:
			ds.append(done)
		return o, r, done, info