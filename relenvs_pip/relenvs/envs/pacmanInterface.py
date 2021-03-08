import gym
from .pacman.pacman import readCommand, ClassicGameRules
from gym.utils import seeding


class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,layout, pacman, ghosts, display, numGames, record, numTraining = 0, numGhostTraining = 0, withoutShield = 0, catchExceptions=False, timeout=60, symX=False, symY=False):
        """"""

        # set OpenAI gym variables
        self._seed = 123
        self.A = ['up', 'down', 'left', 'right', 'stay']
        # self.O = [...]
        # self.state_size = (...)
        self.steps = 0
        self.history = []


        # port input values to fields
        self.layout = layout
        self.pacman = pacman
        self.ghosts = ghosts
        self.display = display
        self.numGames = numGames
        self.record = record
        self.numTraining = numTraining
        self.numGhostTraining = numGhostTraining
        self.withoutShield = withoutShield
        self.catchExceptions = catchExceptions
        self.timeout = timeout
        self.symX = symX
        self.symY = symY


        # set games stats veriables
        self.rules = ClassicGameRules(timeout)
        self.games = []
        self.stat_games = []
        self.last_n_games = []
        self.average_scores = []
        self.win_rates = []
        self.game_index = -1
        import __main__
        __main__.__dict__['_display'] = self.display


        self._reset()




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
        self._take_action(action)
        # self.status = self.env.step()
        # reward = self._get_reward()
        # ob = self.env.getState()
        # episode_over = self.status != hfo_py.IN_GAME
        # return ob, reward, episode_over, {}

    def _reset(self):
        # # update the games stats TODO where to put these end game actions?
        # if not self.beQuiet:
        #     self.games.append(self.env)
        # self.stat_games.append(self.env)
        # self.last_n_games.append(self.env)

        # start a new game
        self.game_index += 1
        # self.beQuiet = self.game_index < self.numTraining + self.numGhostTraining
        self.beQuiet = False # For now always visualize the game
        if self.beQuiet:
            # Suppress output and graphics
            from .pacman import textDisplay
            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.gameDisplay = self.display

            self.rules.quiet = False

        self.env = self.rules.newGame(self.layout, self.pacman, self.ghosts, self.gameDisplay, self.beQuiet,
                                       self.catchExceptions, self.symX, self.symY)

        self.env.start_game()


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
