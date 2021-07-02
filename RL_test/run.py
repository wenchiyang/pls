from RL_test.pacman_policy_gradient_dpl import main
from datetime import datetime

if __name__ == '__main__':
    setting1 = {
        'layout': 'grid2x2',  # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
        'symbolic_grid_width': 2,
        'symbolic_grid_height': 2,
        'learning_rate': 1e-3,
        'reward_goal': 10,
        'reward_crash': 0,
        'reward_food': 0,
        'reward_time': -1,
        'step_limit': 40000,
        'seed': 567,
        'gamma': 0.99,
        'render': False,
        'timestamp': datetime.now().strftime('%Y%m%d_%H:%M')
    }

    setting2 = {
        'layout': 'grid2x2',  # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
        'symbolic_grid_width': 2,
        'symbolic_grid_height': 2,
        'learning_rate': 1e-3,
        'reward_goal': 10,
        'reward_crash': -10,
        'reward_food': 0,
        'reward_time': -1,
        'step_limit': 40000,
        'seed': 567,
        'gamma': 0.99,
        'render': False,
        'timestamp': datetime.now().strftime('%Y%m%d_%H:%M')
    }

    setting3 = {
        'layout': 'grid2x3',  # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
        'symbolic_grid_width': 3,
        'symbolic_grid_height': 2,
        'learning_rate': 1e-3,
        'reward_goal': 10,
        'reward_crash': 0,
        'reward_food': 0,
        'reward_time': -1,
        'step_limit': 90000,
        'seed': 567,
        'gamma': 0.99,
        'render': False,
        'timestamp': datetime.now().strftime('%Y%m%d_%H:%M')
    }

    setting4 = {
        'layout': 'grid2x3',  # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
        'symbolic_grid_width': 3,
        'symbolic_grid_height': 2,
        'learning_rate': 1e-3,
        'reward_goal': 10,
        'reward_crash': -10,
        'reward_food': 0,
        'reward_time': -1,
        'step_limit': 90000,
        'seed': 567,
        'gamma': 0.99,
        'render': False,
        'timestamp': datetime.now().strftime('%Y%m%d_%H:%M')
    }

    shared_args = setting4

    pg_args = {
        'layout': shared_args['layout'],
        'symbolic_grid_width': shared_args['symbolic_grid_width'],
        'symbolic_grid_height': shared_args['symbolic_grid_height'],
        'learning_rate': shared_args['learning_rate'],
        'shield': False,
        'object_detection': None,
        'reward_goal': shared_args['reward_goal'],
        'reward_crash': shared_args['reward_crash'],
        'reward_food': shared_args['reward_food'],
        'reward_time': shared_args['reward_time'],
        'step_limit': shared_args['step_limit'],
        'logger_name': 'pg',
        'seed': shared_args['seed'],
        'gamma': shared_args['gamma'],
        'render': shared_args['render'],
        'timestamp': shared_args['timestamp']
    }
    pg_dpl_nodetect_args = {
        'layout': shared_args['layout'],
        'symbolic_grid_width': shared_args['symbolic_grid_width'],
        'symbolic_grid_height': shared_args['symbolic_grid_height'],
        'learning_rate': shared_args['learning_rate'],
        'shield': True,
        'object_detection': False,
        'reward_goal': shared_args['reward_goal'],
        'reward_crash': shared_args['reward_crash'],
        'reward_food': shared_args['reward_food'],
        'reward_time': shared_args['reward_time'],
        'step_limit': shared_args['step_limit'],
        'logger_name': "pg_dpl_nodetect",
        'seed': shared_args['seed'],
        'gamma': shared_args['gamma'],
        'render': shared_args['render'],
        'timestamp': shared_args['timestamp']
    }
    pg_dpl_detect_args = {
        'layout': shared_args['layout'],
        'symbolic_grid_width': shared_args['symbolic_grid_width'],
        'symbolic_grid_height': shared_args['symbolic_grid_height'],
        'learning_rate': shared_args['learning_rate'],
        'shield': True,
        'object_detection': True,
        'reward_goal': shared_args['reward_goal'],
        'reward_crash': shared_args['reward_crash'],
        'reward_food': shared_args['reward_food'],
        'reward_time': shared_args['reward_time'],
        'step_limit': shared_args['step_limit'],
        'logger_name': "pg_dpl_detect",
        'seed': shared_args['seed'],
        'gamma': shared_args['gamma'],
        'render': shared_args['render'],
        'timestamp': shared_args['timestamp']
    }

    main(pg_args)
    # main(pg_dpl_nodetect_args)
    # main(pg_dpl_detect_args)



