import gym
from gym.spaces import Box, Discrete

from .pacman.pacman import readCommand, ClassicGameRules
import numpy as np

import random as rd


def sample_layout(layout):
    agent_positions, food_positions = sample_positions(layout)
    new_agentPositions = []
    for i, agent_position in enumerate(agent_positions):
        if i == 0:  # add pacman position
            new_agentPositions.append((True, agent_position))
        else:  # add ghost positions
            new_agentPositions.append((False, agent_position))
    layout.agentPositions = new_agentPositions

    for h in range(1, layout.height - 1):
        for w in range(1, layout.width - 1):
            layout.food.data[w][h] = (w, h) in food_positions
    return layout


def sample_positions(layout):
    all_valid_positions = []
    for h in range(layout.height):
        for w in range(layout.width):
            if not layout.walls[w][h]:
                all_valid_positions.append((w, h))

    pos_num = len(layout.agentPositions)
    food_num = np.count_nonzero(np.array(layout.food.data) == True)
    positions = rd.sample(all_valid_positions, pos_num + food_num)

    return positions[:pos_num], positions[pos_num:]


class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, layout, seed, reward_goal, reward_crash, reward_food, reward_time
    ):
        """"""

        args = readCommand(
            [
                "--layout",
                layout,
                "--reward-goal",
                str(reward_goal),
                "--reward-crash",
                str(reward_crash),
                "--reward-food",
                str(reward_food),
                "--reward-time",
                str(reward_time),
            ]
        )

        # set OpenAI gym variables
        self._seed = seed
        self.A = ["Stop", "North", "South", "West", "East"]
        self.steps = 0
        self.history = []

        # port input values to fields
        self.layout = args["layout"]
        self.pacman = args["pacman"]
        self.ghosts = args["ghosts"]
        self.display = args["display"]
        self.numGames = args["numGames"]
        self.record = args["record"]
        self.numTraining = args["numTraining"]
        self.numGhostTraining = args["numGhostTraining"]
        self.withoutShield = args["withoutShield"]
        self.catchExceptions = args["catchExceptions"]
        self.timeout = args["timeout"]
        self.symX = args["symX"]
        self.symY = args["symY"]

        self.reward_goal = args["reward_goal"]
        self.reward_crash = args["reward_crash"]
        self.reward_food = args["reward_food"]
        self.reward_time = args["reward_time"]

        self.rules = ClassicGameRules(
            args["timeout"],
            self.reward_goal,
            self.reward_crash,
            self.reward_food,
            self.reward_time,
        )

        ######

        self.grid_size = 1

        import __main__

        __main__.__dict__["_display"] = self.display

        self.observation_space = Box(
            0,
            1,
            (
                self.layout.height * self.grid_size,
                self.layout.width * self.grid_size
            )
        )
        self.action_space = Discrete(5) # default datatype is np.int64
        self.np_random = rd.seed(self._seed)
        self.reward_range = (0, 10)

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

        if isinstance(action, np.int64) or isinstance(action, int):
            action = self.A[action]

        action = "Stop" if action not in self.get_legal_actions(0) else action

        # perform "doAction" for the pacman
        self.game.agents[agentIndex].doAction(self.game.state, action)
        self.game.take_action(agentIndex, action)
        self.render()


        reward = self.game.state.data.scoreChange

        if self.game.gameOver:
            eps_info = {"last_r": reward}
        else:
            eps_info = dict()
        # move the ghosts
        # if not self.game.gameOver:
        #     for agentIndex in range(1, len(self.game.agents)):
        #         state = self.game.get_observation(agentIndex)
        #         action = self.game.calculate_action(agentIndex, state)
        #         self.game.take_action(agentIndex, action)
        #         self.render()
        #         reward += self.game.state.data.scoreChange

        # return self.game.state, reward, self.game.gameOver, dict()
        return self.my_render(), reward, self.game.gameOver, eps_info

    def reset(self):
        # self.beQuiet = self.game_index < self.numTraining + self.numGhostTraining
        self.beQuiet = True
        if self.beQuiet:
            # Suppress output and graphics
            from .pacman import textDisplay

            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.gameDisplay = self.display
            self.rules.quiet = False

        sampled_layout = sample_layout(self.layout)

        self.game = self.rules.newGame(
            sampled_layout,
            self.pacman,
            self.ghosts,
            self.gameDisplay,
            self.beQuiet,
            self.catchExceptions,
            self.symX,
            self.symY,
        )
        self.game.start_game()

        return self.my_render()

    def render(self, mode="human", close=False):
        self.game.render()
        # if mode == "rgb_array":
        #     image = self.display.get_image()
        #     return image

    def my_render(self):
        return self.game.my_render(grid_size=self.grid_size)

    def get_legal_actions(self, agentIndex):
        return self.game.state.getLegalActions(agentIndex)

    def get_action_meanings(self):
        return self.A

    @staticmethod
    def constraint_func(self):
        return
