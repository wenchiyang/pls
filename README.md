This repository is an implementation of [Safe Reinforcement Learning via Probabilistic Logic Shields](https://www.ijcai.org/proceedings/2023/0637.pdf).

```bibtex
@inproceedings{Yang2023,
  title     = {Safe Reinforcement Learning via Probabilistic Logic Shields},
  author    = {Yang, Wen-Chi and Marra, Giuseppe and Rens, Gavin and De Raedt, Luc},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {5739--5749},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/637},
  url       = {https://doi.org/10.24963/ijcai.2023/637},
}
```

## Dependency
- [pacman_gym](https://github.com/wenchiyang/pacman_gym) 
- [carracing-gym](https://github.com/wenchiyang/carracing-gym) 
- [problog](https://github.com/ML-KULeuven/problog)


## Installation

```shell script
pip install -e .
```

## Quick Start
### Example 1
Train a shielded reinforcement learning agent. For the full example,
please see `pls/examples/train_a_policy/quickstart.py`.

```python
# import env-related classes
from env_specific_classes.carracing.env_classes import (
    Carracing_Monitor,
    Carracing_Callback,
    Carracing_FeaturesExtractor,
    Carracing_Observation_Net,
)

# specify the location of the config file
cwd = os.path.join(os.path.dirname(__file__))
config_file = os.path.join(cwd, "carracing/no_shield/seed1/config.json")

# load the config
with open(config_file) as json_data_file:
    config = json.load(json_data_file)

# call the main algorithm with appropriate classes 
learn_ppo(
    config_folder= os.path.dirname(config_file),
    config=config,
    model_cls=PPO_shielded,
    get_sensor_value_ground_truth=get_ground_truth_of_grass,
    custom_callback_cls=Carracing_Callback,
    monitor_cls=Carracing_Monitor,
    features_extractor_cls=Carracing_FeaturesExtractor,
    observation_net_cls=Carracing_Observation_Net,
)

```


### Example 2
Pretrain your own observation net. For the full example, please see 
`pls/examples/pretrain_sensors_cr/pretrain_sensors.py`.

```python
# location of a pretrained agent to be loaded
cwd = os.path.dirname(os.path.realpath(__file__))
policy_folder = os.path.join(cwd, "..", "train_a_policy/carracing/no_shield/seed1")
# location to save the generated images
img_folder = os.path.join(cwd, "data/")
# location of the labels of the generated imgaes
csv_file = os.path.join(img_folder, "labels.csv")

# generate the images and the corresponding labels
generate_random_images_cr(
    policy_folder=policy_folder,
    model_at_step=600000,
    img_folder=img_folder,
    csv_file=csv_file,
    sample_frequency=50,
    num_imgs=600,
)

# pretrain the observation net
pretrain_observation(
    csv_file=csv_file,
    img_folder=img_folder,
    observation_net_folder=cwd,
    image_dim=48,
    downsampling_size=1,
    num_training_examples=500,
    epochs=10,
    net_class=Carracing_Observation_Net,
    labels=["grass(in_front)", "grass(on_the_left)", "grass(on_the_right)"],
    pretrain_w_extra_labels=False,
    num_test_examples=100,
)
```

### Example 3
Evaluate a trained policy. For the full example, please see `pls/examples/evaluate_a_policy/evaluate_a_policy.py`.

```python
# location of a pretrained agent to be loaded
cwd = os.path.join(os.path.dirname(__file__))
config_file = os.path.join(cwd, "../train_a_policy/carracing/no_shield/seed1/config.json")

mean_reward, std_reward = evaluate(config_file, model_at_step="end", n_test_episodes=10)
print(f"{mean_reward=}, {std_reward=}")
```

## Configure a learning agent
The config file contains all required parameters to run the PPO_shielded algorithm. 
`pls/examples/train_a_policy/` contains example config files.


- `base_policy`: The base reinforcement learning algorithm used. Currently, only "ppo" (Proximal Policy Optimization) is supported.
- `env`: Name of the environment. [Gym environments](https://www.gymlibrary.dev/index.html) are supported.
- `env_features`, `eval_env_features`: Parameters for the training or evaluation environment.
- `monitor_features`: Parameters for the monitor. 
- `policy_params`: Parameters for the learning algorithm (`ppo`). Most parameters are passed to [stablebaselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#).
    - `net_arch_shared`, `net_arch_pi`, `net_arch_vf`: Configuration of Neural network architecture (`net_arch` in stablebaselines3).
    - `alpha`: Coefficient of the safety loss.
- `shield_params`: Parameters for the shield. "Null" if no shield is used.
    - `num_sensors`: Number of sensors.
    - `num_actions`: Number of available discrete actions.
    - `differentiable`: Boolean indicating whether the shield is differentiable.
    - `shield_program`: File location of the shield specification.
    - `ghost_distance`: Detection range of agent sensors. This is a domain-specific parameter for Pacman.
- `policy_safety_params`: Parameters for the policy safety calculator for the safety loss. Same structure as `shield_params`. 
- `observation_params`: Parameters for the observation. 
      - `observation_type`: Observation type ("ground truth" or "pretrained"). If the value is "pretrained", 
then `observation_net`, `net_input_dim` and `noisy_observations` must be provided.
      - `observation_net`: File location of the pretrained observation network.
      - `net_input_dim`: the dimension of the image input.
      - `noisy_observations`: Boolean indicating whether noisy observations are used.


## Configure a shield
A shield is a [ProbLog program](https://problog.readthedocs.io/en/latest/modeling_basic.html#problog). 
It has three main parts. We will use an example to explain how to specify a shield. 

### Actions
A probabilistic distribution over available actions. 
During training, `action(int)` will be replaced by probabilities produced by the policy network.
The clauses are separated by `;`, meaning that the sum of `action(int)` must be 1 at all times.

```prolog
action(0)::action(do_nothing);
action(1)::action(accelerate);
action(2)::action(brake);
action(3)::action(turn_left);
action(4)::action(turn_right).
```

### Sensors
The sensor readings. 
During training, `sensor_value(int)` will be replaced by probabilities produced by the observation network (if `observation_type` is "pretrained"),
or the ground truth sensor values (if `observation_type` is "ground truth"). 
```prolog
sensor_value(0)::grass(in_front).
sensor_value(1)::grass(on_the_left).
sensor_value(2)::grass(on_the_right).
```

### Definition of safety

```prolog
unsafe_next :-                                                        % Grass on the left but not on the right means
    grass(on_the_left), \+ grass(on_the_right), action(turn_left).    % that the agent is on the left border of the road
unsafe_next :-                                                        % thus it is unsafe to turn left or accelerate 
    grass(on_the_left), \+ grass(on_the_right), action(accelerate).   % at this point.
    
unsafe_next :-                                                         
    \+ grass(on_the_left), grass(on_the_right), action(turn_right).    
unsafe_next :-                                                         
    \+ grass(on_the_left), grass(on_the_right), action(accelerate).    

safe_next:- \+unsafe_next.                                            % Being safe is defined as "not unsafe" 
safe_action(A):- action(A).                                           % safe_action will be queried during training
```


## Use a Gym Environment