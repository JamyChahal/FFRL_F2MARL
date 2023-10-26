# FFRL_F2MARL

## Publication

If you use FFRL in a scientific publication, we would appreciate using the following citations:

```
@InProceedings{Chahal2022,
author="Chahal, Jamy
and Seghrouchni, Amal El Fallah
and Belbachir, Assia",
editor="Camacho, David
and Rosaci, Domenico
and Sarn{\'e}, Giuseppe M. L.
and Versaci, Mario",
title="A Force Field Reinforcement Learning Approach for the Observation Problem",
booktitle="Intelligent Distributed Computing XIV",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="89--99",
isbn="978-3-030-96627-0"
}
```

If you use F2MARL, please cite the thesis of Jamy Chahal, named "Multi-drones patrolling and observation of mobile targets". Bibtex coming soon. 

## Installation

`pip3 install -r requirements.txt`

## Training of model

### FFRL

In the **training** folder, execute *pbt_tuner_FFRL.py*. The parameters for the training are specified in the file *params_pbt_FFRL.list* . The model will be saved in ~/ray_result/{name_of_training}/ once one instance of the pbt reached the condition specified at line 157.

The arguments used, specified directly in the code, or by command line, are : 

* **reward **(-r) : 'ind' for the individual reward for the FFRL
* **model** (-m) : The architecture of the neural network. The list contains the number of neurons for each hidden layer. For example [128, 128, 128] means 128 neurons for 3 hidden layers.
* **samples** (-s) : Number of parallel samples performed by the PBT algorithm. 
* **num_workers** (-w) : Number of parallel workers by samples. 
* **params** (-p) : The file name containing the simulation parameter, such as the number of agents
* **name** (-n) : The name of the training. Will be used then in ~/ray_results/name_of_your_training/

The number of CPU used by the training is : samples * num_workers + 1

### F2MARL

In the **training** folder, execute *pbt_tuner_F2MARL.py*. The parameters for the training are specified in the file *params_pbt_F2MARL.list* . The model will be saved in ~/ray_result/{name_of_training}/ once one instance of the pbt reached the condition specified at line 251.

The arguments used, specified directly in the code, or by command line, are the same as the FFRL, except : 

* **reward** (-r) : 'glo' for global reward of the F2MARL, or individual reward for the FFRL
* **model** (-m) : 
  * 'global_global' : for an environment point of view for the actor and critic model
  * 'local_local' : for an agent centric point of view for the actor and critic model
  * 'local_global' : for an agent centric point of view for the actor model, and environment point of vie for the critic model 

The model used by the F2MARL are described in each file of the folder training/model

## Execution of the environment

### Simple run of the environment 

The evaluator folder contains the code to run the simulation with a visual. The file *params.list* is used to modify all the parameters of the simulation.

* *run_pop.py* : used to have a simple view of the simulation

* *run_pop_show_obs.py* : used to have an image feedback of the observation used by the F2MARL
* *run_pop_policy_ffrl.py* : used to see the FFRL behavior of the agents
  * Specify at line 74 the checkpoint directory, which looks like usually ~/ray_results/your_training_name/PPO.../checkpoint_xxxx

* *run_pop_policy_f2marl.py* : used to see the F2MARL behavior of the agents
  * Specify at line 121 the model used (global_global, local_local, local_global)
  * Specify at line 123 the checkpoint directory, which looks like usually ~/ray_results/your_training_name/PPO.../checkpoint_xxxx




## Project Structure

Here is the structure of the repository: 

* **behaviors** : contains all the strategies needed for this project, such as the A-CMOMMT, I-CMOMMT, FFRL, F2MARL etc. 
* **environment** : the PPO environment, implemented on PettingZoo (based on Gym)
* **evaluation** : used to run the environment, with random strategies or trained one 
* **training** : containing all the script to perform the training of the FFRL or F2MARL
