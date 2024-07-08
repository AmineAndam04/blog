+++
title = 'VDN:  Value Decomposition Networks for cooperative multi-agent learning'
date = 2024-07-03T19:52:42+01:00
+++
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## To DO:
- use utility function instead of q function
The VDN paper introduces an algorithm designed to address cooperative multi-agent reinforcement learning with common rewards, where all agents receive the same reward. In this setting, multiple agents work together to maximize the discounted future rewards. Each agent receives a local observation that helps them take appropriate actions. It's worth noting that the VDN algorithm considers only discrete actions.



The cooperative multi-agent reinforcement learning problem can be formally defined using a Dec-POMDP. We need the following notation: 


- $I$: The set of agents where $|I| = N$.
- $\{A_i\}$: The set of actions available to agent $i$. $a = ( a_1, \ldots ,  a_N)$ is the joint action. 
- $\{O_i\}$: The set of observations available to agent $i$. $o =  (o_1, \ldots ,  o_N)$ is the joint observation.
- $R(o,a)$: The reward function. This reward is shared among all agents.

Each agent $i\in I$ receives a local observation $o_i$ from the environment, based on the received observations, each agent sends back an action $a_i$ to which the environment responds with a scalar reward reflecting how good or bad the agents' behavior is. In the cooperative context, the agents have an interest in working together to achieve a common goal.

As in single-agent RL, multi-agent reinforcement learning (MARL) can be categorized into value-based, policy-based, and actor-critic methods. VDN is a value-based approach that adopts the Q-learning paradigm as it learns an action-state value function to find an optimal policy. 

Let's forget about the VDN for now and focus on how Q-learning can solve the cooperative MARL problem. There are two approaches we can use.

The first approach is to use a single-agent reinforcement learning algorithm. This consists of considering that there is an agent who receives the joint observation (e.g. the concatenation of individual observation of each agent) and its action is the joint action of all the agents. Then our goal is to estimate $Q(o,a:\theta)$ as in DQN. In this case, the loss of the Q-learning algorithm is:  

$$
L(\theta) = \frac{1}{|B|} \sum_{(o_t, a_t, r_t, o'_t) \in B} (y_t - Q(o_t, a_t; \theta))^2 \tag{1}
$$

$$
y_t = \begin{cases}
 r_t & \text{if } o_{t+1} \text{ is terminal} \\
 r_t + \gamma \max_{a'} Q(o_{t+1}, a'; \theta) & \text{otherwise}
\end{cases}
\tag{2}
$$

The second approach is to use independent Q-learning. This means that each agent will train its own Q-learning algorithm relying only on its local observation $o_i$ and local actions $a_i$ . This means that we have $N$ loss functions. For each agent, we have the following loss function:



$$
L_i(\theta_i) = \frac{1}{|B|} \sum_{(o_i^t, a_i^t, r_i^t, o_{i}^{'t}) \in B} (y_i^t - Q_i(o_i^t, a_i^t; \theta_i))^2 \tag{3}
$$

Where:
$$
y_i^t = \begin{cases}
 r_t & \text{if } o_i^{t+1} \text{ is terminal} \\
 r_t + \gamma \max_{a'_i} Q(o_i^{t+1}, a'_i; \theta_i) & \text{otherwise}
\end{cases}
\tag{4}
$$

Each of these two approaches has its pros and cons. Let's start with the single-agent RL approach.

- Cons: The Q-network takes as input the joint observation $o$ and joint action $a$ of each agent. This can be very problematic as we have to deal with extremely large inputs. Additionally, the input size will grow exponentially with the number of agents.
- Pros: The loss function is backpropagated using the team reward which is strongly related to the input of the Q-network: the joint action $a$. This is not the case with IQL, where every individual loss is backpropagated with the team reward (not an individual reward) which may not reflect the contribution of the individual action $a_i$ since the team reward depends on the joint action.

For the second approach:

- Cons: As previously explained, the individual Q-networks are updated with a global signal. The team reward may be positive even if an individual action is bad.
- Pros: The Q-networks are trained using local observations $o_i$ and individual actions $a_i$, allowing us to train these networks more efficiently and avoid large inputs. It also becomes easier to deploy these networks.

The idea of the VDN paper is to combine the two approaches in order to take advantage of both of them. First, we want to train our network using only local observations $o_i$ and local actions $a_i$ avoiding large inputs, thus we train local Q-networks $Q(o_i^{t+1}, a'_i; \theta_i)$. 

Second, we want to backpropagate the loss using the right reward signal. This can be done only when working with $Q(o_t, a_t; \theta)$. To achieve this we make the following assumption :

$$
Q(o_t, a_t; \theta) = \sum_{i \in I} Q(o_t^i, a_t^i; \theta_i) \tag{5}
$$
So training individual Q-networks is equivalent to training the joint Q-network.

This means that we don't propagate each individual Q-network separately, but instead, we backpropagate through the sum of the individual Q-networks. 

Thus the loss function of the VDN algorithm is:
$$
L(\theta) = \frac{1}{|B|} \sum_{(h_t, a_t, r_t, h_{t+1}) \in B} \left( r_t + \gamma \max_{a \in A} Q(h_{t+1}, a; \bar{\theta}) - Q(h_t, a_t; \theta) \right)^2 \tag{6}
$$

with

$$
Q(h_t, a_t; \theta) = \sum_{i \in I} Q(h_t^i, a_t^i; \theta_i) \tag{7}
$$

and

$$
\max_{a \in A} Q(h_{t+1}, a; \bar{\theta}) = \sum_{i \in I} \max_{a_i \in A_i} Q(h_{t+1}^i, a_i; \bar{\theta}_i) \tag{8}
$$


### Algorithm 21: Value Decomposition Networks (VDN)

1. Initialize \( n \) utility networks with random parameters \( \theta_1, \ldots, \theta_n \)
2. Initialize \( n \) target networks with parameters \( \theta_1' = \theta_1, \ldots, \theta_n' = \theta_n \)
3. Initialize a shared replay buffer \( D \)
4. For time step \( t = 0, 1, 2, \ldots \) do
    1. Collect current observations \( o_t^1, \ldots, o_t^n \)
    2. For agent \( i = 1, \ldots, n \) do
        - With probability \( \epsilon \): choose random action \( a_t^i \)
        - Otherwise: choose \( a_t^i \in \arg \max_{a^i} Q(h_t^i, a^i; \theta_i) \)
    3. Apply actions; collect shared reward \( r_t \) and next observations \( o_{t+1}^1, \ldots, o_{t+1}^n \)
    4. Store transition \( (h_t, a_t, r_t, h_{t+1}) \) in shared replay buffer \( D \)
    5. Sample mini-batch of \( B \) transitions \( (h_k, a_k, r_k, h_{k+1}) \) from \( D \)
    6. If \( s_{k+1} \) is terminal then
        - Targets \( y_k \leftarrow r_k \)
    7. Else
        - Targets \( y_k \leftarrow r_k + \gamma \sum_{i \in I} \max_{a'_i \in A_i} Q(h_{k+1}^i, a'_i; \theta_i) \)
    8. Loss \( L(\theta) \leftarrow \frac{1}{B} \sum_{k=1}^{B} \left( y_k - \sum_{i \in I} Q(h_k^i, a_k^i; \theta_i) \right)^2 \)
    9. Update parameters \( \theta \) by minimizing the loss \( L(\theta) \)
    10. In a set interval, update target network parameters \( \theta_i' \) for each agent \( i \)

In order to implement the algorithm in pytorch, let's go line by line to see what functions and hyper-parameters we need

```python
class Hyperparameters:
    # empty
class VDN():
    def __init__(self):
        pass
```
- From line 1 to line 2: We need a function to initialize a utility functions and a target functions

```python
import torch
import torch.nn as nn
class Hyperparameters:
    # empty
class Qnetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
class VDN():
    def __init__(self,network):
        self.initialize_networks(network)
    def initialize_networks(self,network):
        pass
```
- Line 3: We need a replay buffer. First we need a choose the size of the replay buffer, the size of the replay buffer is a hyperparameter. We need a function to store transition in the replay buffer and we need to initialize our replay buffer with some random transition, the number of initial samples will also be considered as a hyperparameter:

```python
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
class Hyperparameters:
    buffer_size: int = 100000
    init_num: int = 10000

hyperparameters = Hyperparameters()

class Qnetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
class VDN():
    def __init__(self,hyperparameters,
                      network):

        ## Hyperparameters
        self.buffer_size = hyperparameters.buffer_size
        self.init_num = hyperparameters.init_num
        ## Initialize the network
        self.initialize_networks(network)
        ## The replay buffer
        self.replay_buffer = self.create_buffer()
        self.initialize_buffer()
    def initialize_networks(self, network):
        pass
    def create_buffer(self):
        pass
    def store_transition(self):
        pass
    def initialize_buffer(self):
        pass
```

- Line 4: we need a training function. We need a parameter that will specify the number iterations of the training loop:

```python
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
class Hyperparameters:
    buffer_size: int = 100000
    init_num: int = 10000
    total_time_steps: int = 800000

hyperparameters = Hyperparameters()

class Qnetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
class VDN():
    def __init__(self,hyperparameters,
                      network,
                    ):

        ## Hyperparameters
        self.buffer_size = hyperparameters.buffer_size
        self.init_num = hyperparameters.init_num
        self.total_time_steps = hyperparameters.total_time_steps
        ## Initialize the network
        self.initialize_networks(network)
        ## The replay buffer
        self.replay_buffer = self.create_buffer()
        self.initialize_buffer()

    def train(self):
        
    def initialize_networks(self, network):
        pass
    def create_buffer(self):
        pass
    def store_transition(self):
        pass
    def initialize_buffer(self):
        pass
```

- Line 4.1 : can be handled with env.reset() or env.step()

```python

    
class VDN():
    def __init__(self,hyperparameters,
                      network,
                      env
                    ):

        ...

    def train(self):
        obs, _ = self.env.reset()
        for time_step in range(self.total_time_steps):
            ## self.env.step(actions)

```
- Line 4.2: We need a function for action selection based on the output of the utility network so it will take the current observation as an input. This function will select a random action with a probability epsilon. This epsilon is a hyperparameter:

```python
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
class Hyperparameters:
    ...
    epsilon: float = 0.01

hyperparameters = Hyperparameters()


class VDN():
    def __init__(self,hyperparameters,
                      network):

        ## Hyperparameters
        ...
        self.epsilon = hyperparameters.epsilon
        

    def act(self,obs):
        pass
```
- Line 4.3 and 4.4: inside the training loop we execute the action outputed by the act function and we store the transition in the replay buffer

```python

class VDN():
    def __init__(self,hyperparameters,
                      network):

        ## Hyperparameters
        ...
        self.epsilon = hyperparameters.epsilon
    def train(self):
        obs, _ = self.env.reset()
        for time_step in range(self.total_time_steps):
            actions = self.act(obs)
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            is_done = any(terminations.values()) or any(truncations.values())
            self.store_transition( obs, actions, rewards, next_obs, is_done) ## some processing is needed before storing the transitions
            obs = next_obs

```
- Frol line 4.5 to line 4.8: We need to sample a batch of a specific size: batch_size. Using the sampled batch we need to compute the loss function. We implement a function to compute the loss:

```python
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
class Hyperparameters:
    ...
    buffer_size: int = 32

hyperparameters = Hyperparameters()


class VDN():
    def __init__(self,hyperparameters,
                      network):

        ## Hyperparameters
        ...
        self.buffer_size = hyperparameters.buffer_size
        

    def train(self):
        for time_step in range(self.total_time_steps):
            ## Sample and compute loss  
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
    def compute_loss(self,batch):
        pass
```

- Line 4.9 and 4.10: We need a function to update the utility function using the computed loss. We use a hyperparameter for the interval of the updates. This also means that we need to specify an optimizer and a learning rate.


```python
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
class Hyperparameters:
    ...
    update_target: int = 200
    optimizer: str = "Adam"
    learning_rate: float = 0.01
hyperparameters = Hyperparameters()
class VDN():
    def __init__(self,hyperparameters,
                      network):
        ## Hyperparameters
        ...
        self.update_target = hyperparameters.update_target
        self.optimizer = hyperparameters.optimizer
        self.learning_rate = hyperparameters.learning_rate
    def train(self):
        for time_step in range(self.total_time_steps):
            ## Sample and compute loss  
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
            ## Update the network
            self.update_networks(loss)
    def update_networks(self,loss):
        pass
```

Now we are going to implement each function. Before that we need to talk a bit about the environment.
We will use [Pursuit](https://pettingzoo.farama.org/environments/sisl/pursuit/) environment.For simplicity, we will not use the default environment. We are going to use a less number of agents and targets. Below is a simple code to interact with the environment: 

```python
from pettingzoo.sisl import pursuit_v4
env = pursuit_v4.parallel_env(render_mode="human",shared_reward=False,x_size=10, y_size=10,obs_range=4,n_evaders=18,n_pursuers=5,tag_reward=0.01,catch_reward=5.0)
observations, infos = env.reset()
for iteration in range(20):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    next_observations, rewards, terminations, truncations, infos = env.step(actions)
```