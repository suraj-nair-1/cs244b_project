# Distributed Multi-Agent Consensus for Fault Tolerant Decision Making

### 

## Overview 

* `agent.py` contains the main class used to implement the LF+AF, LF, AF, and PBFT consensus protocols. We assume that every agent acts as both the client and server. 
* `faulty_agents.py` contains wrapper classes that introduce standard faulty Byzantine agents as well as malicious agents with faulty observations. 
* `run_agent.py` is called by `agent.py` and initializes a regular or faulty agent. 

## Experiments 

### (1) Standard Faults

* `run_robustness.py` runs the standard fault experiments with varying numbers of agents and faulty agents. `f` random standard faulty agents are sampled 20 times.

### (2) Faulty Observations

* `run_experiment.py` evaluates observation correctness using malicious agents who send faulty observations. 

### (3) Gridworld

* `multiagent_gridworld` contains all the code pertaining to the gridworld environment. `multiagent_gridworld/multiagent_gridworld/gridworld.py` contains the class for the gridworld environment. 
* `run_control.py` runs the gridworld experiments and contains code that integrates the consensus protocol with taking actions in gridworld. 

