import torch
from torch import nn
from deepproblog.light import DeepProbLogLayer
import torch as th

program = """
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

# program = """
# wall(-1, 0).
# wall(-1, 1).
# wall(0, -1).
# wall(0, 2).
# wall(1, -1).
# wall(1, 2).
# wall(2, -1).
# wall(2, 2).
# wall(3, 0).
# wall(3, 1).
#
#
# action(0)::action(stay);
# action(1)::action(up);
# action(2)::action(down);
# action(3)::action(left);
# action(4)::action(right).
#
#
# % Ghost and Pacman positions are mutually exclusive for now (i.e. only one ghost and one pacman).
# ghost(0)::ghost(0,1);
# ghost(1)::ghost(1,1);
# ghost(2)::ghost(2,1);
# ghost(3)::ghost(0,0);
# ghost(4)::ghost(1,0);
# ghost(5)::ghost(2,0).
#
# pacman(0)::pacman(0,1);
# pacman(1)::pacman(1,1);
# pacman(2)::pacman(2,1);
# pacman(3)::pacman(0,0).
# pacman(4)::pacman(1,0);
# pacman(5)::pacman(2,0).
#
#
# % Transition
# transition(X,Y,stay,X,Y).
# transition(X,Y,left,X1,Y) :- X1 is X - 1.
# transition(X,Y,right,X1,Y) :- X1 is X + 1.
# transition(X,Y,up,X,Y1) :- Y1 is Y + 1.
# transition(X,Y,down,X,Y1) :- Y1 is Y - 1.
#
#
# transition_with_wall(X,Y,A,X,Y) :-
#     transition(X,Y,A,X1,Y1),
#     wall(X1,Y1).
# transition_with_wall(X,Y,A,X1,Y1) :-
#     transition(X,Y,A,X1,Y1),
#     \+ wall(X1,Y1).
#
#
# unsafe :- pacman(X,Y), action(A), transition_with_wall(X,Y,A,X1,Y1), ghost(X1,Y1).
# safe :- \+ unsafe.
#
# evidence(safe).
#
# """
WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1

class DPLSafePolicy(nn.Module):

    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder
        # self.hidden_size = self.image_encoder.hidden_size
        self.input_size = self.image_encoder.input_size
        self.grid_size_x = self.image_encoder.grid_size_x
        self.grid_size_y = self.image_encoder.grid_size_y
        self.num_actions = 5
        self.grid_size = self.grid_size_x * self.grid_size_y
        # self.ghost_layer = nn.Sequential(nn.Linear(self.hidden_size, self.grid_size), nn.Softmax())
        # self.pacman_layer = nn.Sequential(nn.Linear(self.hidden_size, self.grid_size), nn.Softmax())
        # self.base_policy_layer = nn.Sequential(nn.Linear(self.hidden_size, self.num_actions), nn.Softmax())
        self.ghost_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.grid_size),
            nn.Softmax()
        )
        self.pacman_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.grid_size),
            nn.Softmax()
        )
        self.base_policy_layer = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Softmax()
        )
        self.queries = [ "action(stay)", "action(up)",
                         "action(down)", "action(left)",
                         "action(right)"]
        self.program = program
        self.dpl_layer = DeepProbLogLayer(program=self.program, queries=self.queries)



    def forward(self, x):

        h = self.image_encoder(x)
        ghosts = self.ghost_layer(h)
        pacman = self.pacman_layer(h)
        # ghosts = (x[0][1:1+self.grid_size_y, 1:1+self.grid_size_x]==GHOST_COLOR).float().view(1,-1)
        # pacman = (x[0][1:1+self.grid_size_y, 1:1+self.grid_size_x]==PACMAN_COLOR).float().view(1,-1)
        base_actions = self.base_policy_layer(h)

        weights = {"ghost": ghosts,
                   "pacman": pacman,
                   "action": base_actions}

        actions = self.dpl_layer(weights)["action"]


        return actions, base_actions, ghosts, pacman

