## Lipschitz Continuous Reachability Learning (LCRL)

This is a repo for learning trustworthy reachability sets of high-dimensional nonlinear systems using a newly proposed Lipschitz continuous reachability value function. 


# Introduction

We propose a new reach-avoid value function, which is Lipschitz continuous, and its Bellman equation is a contraction mapping. Our method does not need to anneal the time discount factor to 1, which is commonly used in prior works. In theory, assuming the learning quality is reasonably good, our training process could require only 1/N of the total training time in prior works, where N represents the number of times of annealing the time discount factor in previous methods.  We also propose two post-learning reach-avoid set certification methods for verifying if a set of states is within the ground truth reach-avoid set.


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



# Limitation and future directions:
In the certification part of our work, we begin by exploring the new idea of using Lipschitz continuity and the learned control policy to construct a theoretical lower bound for the ground truth reach-avoid value function, which is typically unknown and difficult to compute for high-dimensional systems. However, we believe there are several avenues to improve this theoretical lower bound. For instance, the critic loss in max-min DDPG approximately reflects the magnitude of the value function fitting error. This raises a natural question: can we utilize critic loss information to construct a high-confidence lower bound for the ground truth value function? While this may require a detailed sample complexity analysis, the Lipschitz continuity of the value function could be advantageous. We believe this approach holds promise for developing a tighter lower bound on the ground truth value function, ultimately enabling the recovery of trustworthy reach-avoid sets in high-dimensional problems. Deriving a tight deterministic lower bound of the ground truth value function would be even more compelling, though this might necessitate substantial prior knowledge, such as the controllability and stability of the dynamical systems involved in reach-avoid problems.


Moreover, our real-time SOCP certification heavily relies on the Clarabel QP solver, which, fortunately, is very fast. The current implementation does not exploit the structure of these SOCPs when evaluating worst-case constraint violations and target set reachability. Therefore, we see significant opportunities to enhance our SOCP trajectory-level certification methods for faster computation and reduced conservatism. For example, could agents leverage decentralized SOCP solvers to collaboratively verify their joint safety in complex traffic scenarios? Additionally, how can we harness the sparsity structure of dynamics to simplify SOCP computations?


Overall, we are optimistic that in the near future, we will be able to compute trustworthy reach-avoid sets for high-dimensional, real-world systems. Our hardware drone racing experiment is one example that suggests this goal is within reach. If you have any questions or suggestions, please feel free to contact the correspondence author Jingqi Li at jingqili@berkeley.edu.

