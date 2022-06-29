# NeSyProject
This code is based on http://www.cs.ru.nl/personal/nilsjansen/subs/shield_rl/ 


## Dependency [TODO add links to packages]
pacman_gym

carracing 

sokoban 

## Installation 

Install the pip package

```shell script
pip install -e .
```

## Config an Experiment
```json
{
  "workflow_name": "ppo", # base RL algorithm
  "raw_logger": "ppo_raw",
  "info_logger": "ppo_info",
  "env_type": "GoalFinding-v0", 

  "env_features": {
    "layout": "grid7x7_5_ghosts", # only needed in GoalFinding; the name of the layout file 
    "reward_goal": 10, # reward structure
    "reward_crash": 0, # reward structure
    "reward_food": 0, # reward structure
    "reward_time": -0.1, # reward structure
    "render": false, # whether or not to render the
    "max_steps": 200, # max steps of an episode
    "num_maps": 100, # the number of randomly sampled maps used for training 
    "seed": 567,
    "render_mode": "gray" # GoalFinding has "human", "tinygrid" and "gray"
  },

  "model_features": {
    "name": "no_shielding", # This is actually not used
    "encoder_params": {
      "height": 240, # the height of images given to the agent
      "width": 240, # the width of images given to the agent
      "downsampling_size": 7 # the agent will downsample the given imgaes, larger values mean more blurry, 1 for no downsamping, 
    },
    "params": { 
      "log_interval": 1, # frequency of logging to log.txt
      "batch_size": 128, 
      "n_epochs": 40,
      "n_steps": 1024,
      "learning_rate": 0.001,
      "seed": 567,
      "clip_range": 0.2,
      "gamma": 0.99,
      "step_limit": 1000000,
      "program_type": "relative_loc_simple", # not needed by no_shielding 
      "debug_program_type": "relative_loc_simple", # needed by all settings, to debug
      "alpha": 0, # between 0 and 1. 0 means no shielding, 1 means hard_shielding
      "n_ghost_locs": 4, # this field is specifically for GoalFinidng
      "differentiable_shield": true, # true for PLS and false for VSRL, not applicable for no shielding
      "net_arch_shared": [], # TODO copy stable baselines doc
      "net_arch_pi": [128], # TODO copy stable baselines doc
      "net_arch_vf": [128], # TODO copy stable baselines doc
      "sensor_noise": 0, # deprecated
      "max_num_rejected_samples": 100000, # only for VSRL. If the agent has sampleed this much actions, the next action will be accept no matter how (un)safe it is
      "use_learned_observations": null, # whether to use pretrained observation functions
      "noisy_observations": null, # true for noisy observations, false for deterministic observations; only applicable for use_learned_observations=true 
      "observation_type": null # the filename of the pretrained observation; only applicable for use_learned_observations=true 
    }
  }
}
```

## Run code
```shell script
python src/run.py
```

