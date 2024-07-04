+++
title = 'VDN:  Value Decomposition Networks for cooperative multi-agent learning'
date = 2024-07-03T19:52:42+01:00
+++

The VDN paper presents an algorithm to solve cooperative multi-agent reinforcement learning  with common rewards (team reward). In this settings, several agents are working together to maximise the (discounted) future rewards. Each agent receive a local obervation ( partial-observabiliy) that helps him take apropriate actions. It's worth mentioning that the VDN algorithm considers only discrete actions. 

The cooperative MARL problem can be solved using single-agent RL algorithms such as DQN or PPO by considering a single-agent whoses state space is the contatenated state of the agents and an action space that is the combination of the action spaces of the agents. Alternatively, each agent can be seen as an idependant agent runing his own PPO algorithm for example. 


The two aformentioned algorithms can be problematic. The first one can suffer from state space and action space explosion, as there are exponential with increasing number of agents. The second approach suffers from non-stationarity and credit assignement. 

