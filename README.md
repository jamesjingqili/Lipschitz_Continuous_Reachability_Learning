## Lipschitz Continuous Reachability Learning (LCRL)

# Introduction

This is a repo for learning trustworthy reachability sets of high-dimensional nonlinear systems using deep RL

We proposed a new reach-avoid value function, which is Lipschitz continuous, and its Bellman equation is a contraction mapping. Our method does not need to anneal the time discount factor to 1, which is commonly used in prior works. We also introduce two post-learning reach-avoid set certification methods for verifying if a set of states is within the ground truth reach-avoid set.

Both of our methods provide deterministic guarantees of whether a set of states can safely reach the target sets, under all potential disturbances within a prespecified bound. Our first certification method uses the learned control policy and the Lipschitz constant of the dynamics to construct a theoretical lower bound of the ground truth value function. The super zero-level set of this constructed lower bounding function recovers a subset of the ground truth reach-avoid set; Moreover, our second certification method formulates efficient second-order cone programmings to evaluate the constraint violations and the target set reaching, against the worst-case disturbance. 

Both of our method can be computed in real-time to certify if a neighboring set of the current state is within the ground truth reach-avoid set. The computational complexity of evaluating our two certifications scale polynomially with respect to the state dimensions.

In addition, our method can be used offline for a comprehensive certification, i.e., certifying if a large set of states is within the ground truth reach-avoid set.



# Implementation details and installation guide:

We adapt max-min DDPG and max-min SAC to learn our new reach-avoid value function. Our implementation builds upon the deep RL infrastructure of tianshou (version 0.5.1).  

Brief tutorial: we use experiment_script/run_training_ddpg.py to learn our reach-avoid value function. We visualize the learned reach-avoid sets, Lipschitz certified, and SOCP certified sets in experiment_script/droneracing_post_training_DDPG_eval_new.ipynb. 


Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py"

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg



