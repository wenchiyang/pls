from torch import nn
from deepproblog.light import DeepProbLogLayer
import torch as th
from util import myformat, get_ground_wall
import torch.nn.functional as F


WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1


class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size)
        )

    def forward(self, x, T=1):
        xx = th.flatten(x, 1)
        action_scores = self.network(xx)
        return F.softmax(action_scores, dim=1)

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 n_actions,
                 shield,
                 detect_ghosts,
                 detect_walls,
                 program_path,
                 logger):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.shield = shield
        self.detect_ghosts = detect_ghosts
        self.detect_walls = detect_walls
        self.n_actions = n_actions
        self.program_path = program_path
        self.logger = logger

    def forward(self, x):
        xx = th.flatten(x, 1)
        return xx

class DPLSafePolicy(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder
        self.logger = self.image_encoder.logger
        self.input_size = self.image_encoder.input_size
        self.shield = self.image_encoder.shield
        self.detect_ghosts = self.image_encoder.detect_ghosts
        self.detect_walls = self.image_encoder.detect_walls
        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path

        self.base_policy_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Softmax(),
        )

        if self.detect_ghosts:
            self.ghost_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),  # TODO : add a flag
            )
        if self.detect_walls:
            self.wall_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),
            )

        if self.shield:
            with open(self.program_path) as f:
                self.program = f.read()

            self.queries = [
                "safe_action(stay)",
                "safe_action(up)",
                "safe_action(down)",
                "safe_action(left)",
                "safe_action(right)",
                "safe_next",
            ]
            self.dpl_layer = DeepProbLogLayer(
                program=self.program, queries=self.queries
            )

    def forward(self, x):
        h = self.image_encoder(x)
        base_actions = self.base_policy_layer(h)

        if not self.shield and not self.object_detection:
            actions = base_actions
            return self.normalize(actions)

        # ghosts_ground = (x[0][1:1 + self.relevant_grid_height, 1:1 + self.relevant_grid_width] == GHOST_COLOR)\
        #     .float().view(1,-1)
        ghosts_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, GHOST_COLOR)
        wall_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, WALL_COLOR)

        if self.detect_ghosts:
            ghosts = self.ghost_layer(h)
        else:
            ghosts = ghosts_ground_relative

        if self.detect_walls:
            walls = self.wall_layer(h)
        else:
            walls = wall_ground_relative

        weights = {"ghost": ghosts, "wall": walls, "action": base_actions}

        results = self.dpl_layer(weights)

        if self.shield:
            actions = results["safe_action"]
            safe_next = results["safe_next"]
            actions = actions / safe_next
            with th.no_grad():
                self.logger.debug(f"Shielded probs: {myformat(actions.data)}")
                self.logger.debug(f"Base probs:     {myformat(base_actions.data)}")
                self.logger.debug(f"Ghost probs:    {myformat(ghosts.data)}")
                self.logger.debug(
                    f"Ghost truth:    {myformat(ghosts_ground_relative.data)}"
                )
                self.logger.debug(f"Wall probs:     {myformat(walls.data)}")
                self.logger.debug(
                    f"Wall truth:     {myformat(wall_ground_relative.data)}"
                )
                # self.logger.debug(f"Safe current:   {myformat(safe_current.data)}")
                self.logger.debug(f"Safe next:      {myformat(safe_next.data)}")
        else:
            safe_next = results["safe_next"]
            actions = results["safe_action"]
            with th.no_grad():
                self.logger.debug(f"Shielded probs: {myformat(actions.data)}")
                self.logger.debug(f"Base probs:     {myformat(base_actions.data)}")
                self.logger.debug(f"Ghost probs:    {myformat(ghosts.data)}")
                self.logger.debug(
                    f"Ghost truth:    {myformat(ghosts_ground_relative.data)}"
                )
                self.logger.debug(f"Wall probs:     {myformat(walls.data)}")
                self.logger.debug(
                    f"Wall truth:     {myformat(wall_ground_relative.data)}"
                )
                # self.logger.debug(f"Safe current:   {myformat(safe_current.data)}")
                self.logger.debug(f"Safe next:      {myformat(safe_next.data)}")

        return self.normalize(actions)

    def normalize(self, probs):
        small_indices = [
            i for i in (probs[0].abs() < 1e-6).nonzero(as_tuple=True)[0].numpy()
        ]
        neg_indices = [i for i in (probs[0] < 0).nonzero(as_tuple=True)[0].numpy()]
        for i in small_indices + neg_indices:
            probs[0][i] = 0
        return probs
