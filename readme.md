# Revisiting Peng's Q(lambda) for Modern Reinforcement Learning @ ICML 2021

This is the open source implementation of a few important multi-step deep RL algorithms discussed in the ICML 2021 paper. We implement these algorithms in combination mainly with [TD3](https://arxiv.org/abs/1802.09477.pdf), an actor-critic algorithm for continuous control.

The code is based on the deep RL library of [SpinningUp](https://github.com/openai/spinningup). We greatly appreciate the open source efforts of the library!

This code base implements a few multi-step algorithms, including

- [Peng's Q(lambda)](https://link.springer.com/content/pdf/10.1023/A:1018076709321.pdf)
- [Uncorrected n-step](https://arxiv.org/pdf/1710.02298.pdf)
- [Retrace](https://arxiv.org/abs/1606.02647)
- [Tree-backup](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs)

Installation
------------------
Follow the instructions for installing [SpinningUp](https://github.com/openai/spinningup), you might also need environment libraries such as [Gym](https://github.com/openai/gym) and [MuJoCo](https://github.com/openai/mujoco-py).

You might also want to check out [DeepMind control suite](https://github.com/deepmind/dm_control) and [Pybullet](https://pybullet.org) to train with other environments.

Introduction to the code structure
------------------
The code is located under the sub-directory `spinup/algos/tf1/td3_peng`. Two main files implement the algorithms.

- The file `td3_peng.py` implements Peng's Q(lambda) and uncorrected n-step, with a deterministic policy.
- The file `td3_retrace.py` implements Retrace and tree-backup, with a stochastic policy. 

A few important aspects of the implementation.

- We use n-step replay buffer that collects and samples partial trajectories of length `n`. We implement n-step transition collection by an environment wrapper in `wrapper.py`. The buffer is implemented in the main files.
- We compute Q-function targets with two critics to reduce over-estimation. Targets are computed with Peng's Q(lambda), n-step, Retrace or tree-backup, in a recursive manner.
- We share hyper-parmaeters (including architectures, batch size, optimizer, learning rate, etc) as the original baseline as much as possible.

Running the code
------------------
To run Peng's Q(lambda) with delayed environment (with `k=3`), n-step buffer with `n=5`, run the following
```bash
python td3_peng.py --env HalfCheetah-v1 --seed 100 --delay 3 --nstep 5 --lambda_ 0.7
```

To run n-step with delayed environment (with `k=3`), n-step buffer with `n=5`, run the following
```bash
python td3_peng.py --env HalfCheetah-v1 --seed 100 --delay 3 --nstep 5 --lambda_ 1.0
```

To run Retrace with delayed environment (with `k=3`), n-step buffer with `n=5`, just set `lambda=1.0` and run the following
```bash
python td3_retrace.py --update-mode retrace --env HalfCheetah-v1 --seed 100 --delay 3 --nstep 5 --lambda_ 1.0
```

Examine the results
------------------
The main files log diagnostics and statistics during training to the terminal. Each run of the main file also saves the evaluated returns and training time steps to a newly created sub-directory.

Citation
------------------
If you find this code base useful, you are encouraged to cite the following paper

```
@article{kozuno2021revisiting,
  title={Revisiting Peng's Q ($$\backslash$lambda $) for Modern Reinforcement Learning},
  author={Kozuno, Tadashi and Tang, Yunhao and Rowland, Mark and Munos, R{\'e}mi and Kapturowski, Steven and Dabney, Will and Valko, Michal and Abel, David},
  journal={arXiv preprint arXiv:2103.00107},
  year={2021}
}
```
