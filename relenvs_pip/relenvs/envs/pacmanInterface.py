import gym
from .pacman.pacman import readCommand, ClassicGameRules

class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,layout, pacman, ghosts, display, numGames, record, numTraining = 0, numGhostTraining = 0, withoutShield = 0, catchExceptions=False, timeout=60, symX=False, symY=False):
        """"""

        # set OpenAI gym variables
        self._seed = 123
        self.A = ['North', 'South', 'West', 'East', 'Stop']
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
        self.rules = ClassicGameRules(timeout)

        import __main__
        __main__.__dict__['_display'] = self.display

        self.reset()

    def step(self, action):
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
        agentIndex = 0
        action = 'Stop' if action not in self.get_legal_actions(0) else action
        # perform "doAction" for the pacman
        self.game.agents[agentIndex].doAction(self.game.state, action)
        # if callable(getattr(self.game.agents[agentIndex], "doAction", None)):
        self.game.take_action(agentIndex, action)
        self.render()

        for agentIndex in range(1, len(self.game.agents)):
            state = self.game.get_observation(agentIndex)
            action = self.game.calculate_action(agentIndex, state)
            self.game.take_action(agentIndex, action)
            self.render()

        return self.game.state, self.game.state.data.scoreChange, self.game.gameOver, dict()

    def reset(self):
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

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts, self.gameDisplay, self.beQuiet,
                                       self.catchExceptions, self.symX, self.symY)
        self.game.start_game()

    def render(self, mode='human', close=False):
        self.game.render()

    def get_legal_actions(self, agentIndex):
        return self.game.state.getLegalActions(agentIndex)




