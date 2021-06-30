from RL_test.pacman_policy_gradient import main as no_dpl
from RL_test.pacman_policy_gradient_dpl import main as dpl

if __name__ == '__main__':
    step_limit = 40000
    no_dpl(step_limit=step_limit)
    dpl(step_limit=step_limit)
