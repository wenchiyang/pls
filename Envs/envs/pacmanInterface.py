import gym
from .pacman.pacman import readCommand

class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions, symX, symY):
        """"""
        # n = SimpleNamespace(**args)

        # import __main__
        # __main__.__dict__['_display'] = display
        #
        # rules = pacman.ClassicGameRules(timeout)
        # games = []
        # stat_games = []
        # last_n_games = []
        # average_scores = []
        # win_rates = []
        #
        # game = rules.newGame(layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions, symX, symY)

    def _step(self, action):
        """
        Parameters
        ----------
        action :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # self._take_action(action)
        # self.status = self.env.step()
        # reward = self._get_reward()
        # ob = self.env.getState()
        # episode_over = self.status != hfo_py.IN_GAME
        # return ob, reward, episode_over, {}

    def _reset(self):
        pass
    def _render(self, mode='human', close=False):
        pass
    def _take_action(self, action):
        pass
    def _get_reward(self):
        """ Reward is given for XY. """
        # if self.status == FOOBAR:
        #     return 1
        # elif self.status == ABC:
        #     return self.somestate ** 2
        # else:
        #     return 0