from gym.envs.registration import register

register(
    id='pacman-v0',
    entry_point='Envs.envs:PacmanEnv',
)
register(
    id='warehouse-v0',
    entry_point='Envs.envs:WarehouseEnv',
)