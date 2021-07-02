import torch
from torch import nn
from deepproblog.light import DeepProbLogLayer
import torch as th

program_2x2 = """
wall(-1, 0).
wall(-1, 1).
wall(0, -1).
wall(0, 2).
wall(1, -1).
wall(1, 2).
wall(2, 0).
wall(2, 1).


action(0)::action(stay); 
action(1)::action(up); 
action(2)::action(down); 
action(3)::action(left); 
action(4)::action(right).


% Ghost and Pacman positions are mutually exclusive for now (i.e. only one ghost and one pacman).
ghost(0)::ghost(0,1);
ghost(1)::ghost(1,1);
ghost(2)::ghost(0,0);
ghost(3)::ghost(1,0).

pacman(0)::pacman(0,1);
pacman(1)::pacman(1,1);
pacman(2)::pacman(0,0);
pacman(3)::pacman(1,0).


% Transition
transition(X,Y,stay,X,Y).
transition(X,Y,left,X1,Y) :- X1 is X - 1.
transition(X,Y,right,X1,Y) :- X1 is X + 1.
transition(X,Y,up,X,Y1) :- Y1 is Y + 1.
transition(X,Y,down,X,Y1) :- Y1 is Y - 1.


transition_with_wall(X,Y,A,X,Y) :- 
    transition(X,Y,A,X1,Y1),
    wall(X1,Y1).
transition_with_wall(X,Y,A,X1,Y1) :-
    transition(X,Y,A,X1,Y1),
    \+ wall(X1,Y1).


unsafe :- pacman(X,Y), action(A), transition_with_wall(X,Y,A,X1,Y1), ghost(X1,Y1).
safe :- \+ unsafe.

evidence(safe).

"""
program_2x3 = """
wall(-1, 0).
wall(-1, 1).
wall(0, -1).
wall(0, 2).
wall(1, -1).
wall(1, 2).
wall(2, -1).
wall(2, 2).
wall(3, 0).
wall(3, 1).


action(0)::action(stay);
action(1)::action(up);
action(2)::action(down);
action(3)::action(left);
action(4)::action(right).


% Ghost and Pacman positions are mutually exclusive for now (i.e. only one ghost and one pacman).
ghost(0)::ghost(0,1);
ghost(1)::ghost(1,1);
ghost(2)::ghost(2,1);
ghost(3)::ghost(0,0);
ghost(4)::ghost(1,0);
ghost(5)::ghost(2,0).

pacman(0)::pacman(0,1);
pacman(1)::pacman(1,1);
pacman(2)::pacman(2,1);
pacman(3)::pacman(0,0).
pacman(4)::pacman(1,0);
pacman(5)::pacman(2,0).


% Transition
transition(X,Y,stay,X,Y).
transition(X,Y,left,X1,Y) :- X1 is X - 1.
transition(X,Y,right,X1,Y) :- X1 is X + 1.
transition(X,Y,up,X,Y1) :- Y1 is Y + 1.
transition(X,Y,down,X,Y1) :- Y1 is Y - 1.


transition_with_wall(X,Y,A,X,Y) :-
    transition(X,Y,A,X1,Y1),
    wall(X1,Y1).
transition_with_wall(X,Y,A,X1,Y1) :-
    transition(X,Y,A,X1,Y1),
    \+ wall(X1,Y1).


unsafe :- pacman(X,Y), action(A), transition_with_wall(X,Y,A,X1,Y1), ghost(X1,Y1).
safe :- \+ unsafe.

evidence(safe).

"""

layout_program = {
    'grid2x2': program_2x2,
    'grid2x3': program_2x3
}

WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1

class DPLSafePolicy(nn.Module):

    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder
        self.logger = self.image_encoder.logger
        self.input_size = self.image_encoder.input_size
        self.relevant_grid_width = self.image_encoder.relevant_grid_width
        self.relevant_grid_height = self.image_encoder.relevant_grid_height
        self.object_detection = self.image_encoder.object_detection
        self.n_actions = self.image_encoder.n_actions



        self.n_relevant_cells = self.relevant_grid_width * self.relevant_grid_height


        self.ghost_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_relevant_cells),
            nn.Softmax()
        )
        self.pacman_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_relevant_cells),
            nn.Softmax()
        )
        self.base_policy_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
            nn.Softmax()
        )
        self.queries = [ "action(stay)", "action(up)",
                         "action(down)", "action(left)",
                         "action(right)"]

        key = f"grid{self.relevant_grid_height}x{self.relevant_grid_width}"
        self.program = layout_program[key]

        self.dpl_layer = DeepProbLogLayer(program=self.program, queries=self.queries)



    def forward(self, x):
        h = self.image_encoder(x)
        if self.object_detection:
            ghosts = self.ghost_layer(h)
            pacman = self.pacman_layer(h)
        else:
            ghosts = (x[0][1:1+self.relevant_grid_height, 1:1+self.relevant_grid_width]==GHOST_COLOR).float().view(1,-1)
            pacman = (x[0][1:1+self.relevant_grid_height, 1:1+self.relevant_grid_width]==PACMAN_COLOR).float().view(1,-1)
        base_actions = self.base_policy_layer(h)

        weights = {"ghost": ghosts,
                   "pacman": pacman,
                   "action": base_actions}

        actions = self.dpl_layer(weights)["action"]

        self.logger.debug(f"Shielded probs: {actions}")
        self.logger.debug(f"Base probs:     {base_actions}")
        self.logger.debug(f"Ghost probs:    {ghosts}")
        self.logger.debug(f"Pacman probs:   {pacman}")
        return self.normalize(actions)

    def normalize(self, probs):
        indices = [int(index[0]) for index in (probs.abs() < 1e-6).nonzero(as_tuple=True) if index.numel()]
        for i in indices:
            probs[0][i] = 0
        # with torch.no_grad():
        #     sum_probs = probs.sum(dim=-1, keepdim=True)
        # probs = probs / sum_probs
        return probs

