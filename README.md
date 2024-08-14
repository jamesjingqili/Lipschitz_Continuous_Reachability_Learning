This is a repo for learning trustworthy reachability sets of high-dimensional nonlinear systems using deep RL

We proposed a new reach-avoid value function which is Lipschitz continuous and its Bellman equation is a contraction mapping without any need of approximation, which is commonly used in prior works. We also introduce a post-learning reach-avoid set certification for verifying if a set of interest is within the ground truth reach-avoid set.

We adapt max-min DDPG and max-min SAC to learn our new reach-avoid value function. Our implementation builds upon the infrastructure of tianshou (version 0.5.1), which leverages Jax and Torch for efficient deep RL training.  

Brief tutorial: we use experiment_script/run_training_ddpg.py to learn our reach-avoid value function. We visualize the learned reach-avoid sets, Lipschitz certified, and SOCP certified sets in experiment_script/droneracing_post_training_DDPG_eval_new.ipynb. 



Install instruction:

1. git clone the repo

2. cd to the folder of this repo, where you should be able to see the "setup.py"

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg

