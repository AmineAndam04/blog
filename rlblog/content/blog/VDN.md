+++
title = 'VDN:  Value Decomposition Networks for cooperative multi-agent learning'
date = 2024-07-03T19:52:42+01:00
+++
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


The VDN paper presents an algorithm to solve cooperative multi-agent reinforcement learning  with common rewards (team reward). In this settings, several agents are working together to maximise the (discounted) future rewards. Each agent receive a local obervation ( partial-observabiliy) that helps him take apropriate actions. It's worth mentioning that the VDN algorithm considers only discrete actions. 

The cooperative MARL problem can be solved using single-agent RL algorithms such as DQN or PPO by considering a single-agent whoses state space is the contatenated state of the agents and an action space that is the combination of the action spaces of the agents. Alternatively, each agent can be seen as an idependant agent runing his own PPO algorithm for example. 


The two aformentioned algorithms can be problematic. The first one can suffer from state space and action space explosion, as there are exponential with increasing number of agents. The second approach suffers from non-stationarity and credit assignement. 


We consider that we have a finate set of agents $$ I = {1, ...,n} $$ each agent receives an obsevation $O_i$ and a reward $R_i$. In our case, all agents recieve the same reward $ R_i = R $. Our goal is to maximize the expected cumulative discounted reward $\sum_{\tau=t} \gamma^{\tau - t} r_{\tau}$

We formalize the Dec-POMDP problem as 
A Dec-POMDP is formally defined by the tuple $(N,\{A_i\}_{i=1, \ldots,  N}, \{O_i\}_{i=1, \ldots,  N}, R)$, where:

- $N$: The number of agents in the environment.
- $\{A_i\}$: The set of actions available to agent $i$. $a = \{a_i\}_{i=1, \ldots,  N}$ is the joint action. 
- $\{O_i\}$: The set of observations available to agent $i$. Each agent receives observations that provide partial information about the state. $o = \{o_i\}_{i=1, \ldots,  N}$ is the joint observation.
- $R$: The reward function $R(o,a)$, which provides a scalar reward based on the current observation and the actions of all agents. This reward is  shared among all agents.

To solve this problem we use the Q-Learning paradigm. There is two approach in solving this problem:

The first approach is to use single-agent reinforcement learning algorithm. This consist in considering that there is an agent who receives the joint observation (e.g. the concatenation of individual observation of each agent) and its action is the joint actions of all the agents. This can be done using Q-learnig methods where the loss function is the following: 

$$
L(\theta) = \frac{1}{|B|} \sum_{(o_t, a_t, r_t, o'_t) \in B} (y_t - Q(o_t, a_t; \theta))
$$

$$
y_t = \begin{cases}
    r_t & \text{if } o_{t+1} \text{ is terminal} \\
    r_t + \gamma \max_{a'} Q(o_{t+1}, a'; \theta) & \text{otherwise}
\end{cases}
$$

The second approach is to use independent Q-learning. This means that each agent will instantiate his own Q-learning algorithm relying on his local observation $o_i$ and the joint reward. This means that we have $N$ loss function. For each agent we have the following loss function:



$$
L_i(\theta_i) = \frac{1}{|B|} \sum_{(o_i^t, a_i^t, r_i^t, o_{i}^{'t}) \in B} (y_i^t - Q_i(o_i^t, a_i^t; \theta))
$$

where:
$$
y_i^t = \begin{cases}
    r_t & \text{if } o_i^{t+1} \text{ is terminal} \\
    r_t + \gamma \max_{a'_i} Q(o_i^{t+1}, a'_i; \theta_i) & \text{otherwise}
\end{cases}
$$

The Q-network in the single agent case takes as an input the concatenated observations and actions of each agent. This can be very problematic as we have to deal with very large inputs. Moreover, the input will grow exponentially with the numbe of agents. However, when back propagating we use the team reward $R$ which reflects and strongly correlate with the joint action $a$. This is not the case with the independant Q-learning losses as each agent backpropagate $R$, but we don't have any guarantee that $R$ reflects the quality of the individual action $a_i$ as all agent contributed to have $R$. But the independent Q-learning enables us to use just the local observation $o_i$, thus we can train those networks more effeciently avoiding state explosion problems. it also helps when we want to deploy the networks.

The idea of VDN paper is two combine the two approaches in order to take advantage of both approaches. First, we want to train our networks using only local observations $o_i$ avoiding large inputs, thus we train these networks $Q(o_i^{t+1}, a'_i; \theta_i)$. Second we want to backpropagate using the right reward signal. This can be done only when working with $Q(o_t, a_t; \theta)$. So we make the following assumption :

$$
Q(o_t, a_t; \theta) = \sum_{i \in I} Q(o_t^i, a_t^i; \theta_i)
$$

This means that we don't propagate each individual Q-network, but instead we backpropagate through the sum the individual Q-networks. 

The loss function of the VDN algorithm is:
$$
L(\theta) = \frac{1}{|B|} \sum_{(h_t, a_t, r_t, h_{t+1}) \in B} \left( r_t + \gamma \max_{a \in A} Q(h_{t+1}, a; \bar{\theta}) - Q(h_t, a_t; \theta) \right)^2 \tag{9.43}
$$

with

$$
Q(h_t, a_t; \theta) = \sum_{i \in I} Q(h_t^i, a_t^i; \theta_i) \tag{9.44}
$$

and

$$
\max_{a \in A} Q(h_{t+1}, a; \bar{\theta}) = \sum_{i \in I} \max_{a_i \in A_i} Q(h_{t+1}^i, a_i; \bar{\theta}_i)
$$


