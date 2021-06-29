import torch
from torch import nn
from deepproblog.light import DeepProbLogLayer


program = """


wall(X,Y) :- (X<0;X>1;Y<0;Y>1).


action(0)::action(stay); 
action(1)::action(up); 
action(2)::action(down); 
action(3)::action(left); 
action(4)::action(right).




% Ghost and Pacman positions are mutually exclusive for now (i.e. only one ghost and one pacman).
ghost(0)::ghost(0,0);
ghost(1)::ghost(0,1);
ghost(2)::ghost(1,0);
ghost(3)::ghost(1,1).

pacman(0)::pacman(0,0);
pacman(1)::pacman(0,1);
pacman(2)::pacman(1,0);
pacman(3)::pacman(1,1).


% Transition
transition(X,Y,west,X1,Y) :- X1 is X - 1.
transition(X,Y,east,X1,Y) :- X1 is X + 1.
transition(X,Y,stay,X,Y).
transition(X,Y,up,X,Y1) :- Y1 is Y + 1.
transition(X,Y,down,X,Y1) :- Y1 is Y - 1.


transition_with_wall(X,Y,A,X1,Y1) :- transition(X,Y,A,X1,Y1), wall(X1,Y1), X2=X, Y2=Y.
transition_with_wall(X,Y,A,X1,Y1) :- transition(X,Y,A,X1,Y1), \+ wall(X1,Y1), X2=X1, Y2=Y1.


unsafe :- pacman(X,Y), action(A), transition(X,Y,A,X1,Y1), ghost(X1,Y1).
safe :- \+ unsafe.

evidence(safe).

"""



class DPLSafePolicy(nn.Module):

    def __init__(self, image_encoder):
        super().__init__()

        self.image_encoder = image_encoder
        self.hidden_size = self.image_encoder.hidden_size
        self.num_actions = 5
        self.grid_size = 2 * 2
        self.ghost_layer = nn.Sequential(nn.Linear(self.hidden_size, self.grid_size), nn.Softmax())
        self.pacman_layer = nn.Sequential(nn.Linear(self.hidden_size, self.grid_size), nn.Softmax())
        self.base_policy_layer = nn.Sequential(nn.Linear(self.hidden_size, self.num_actions), nn.Softmax())
        self.queries = [ "action(stay)", "action(up)",
                         "action(down)", "action(left)",
                         "action(right)"]
        self.program = program
        self.dpl_layer = DeepProbLogLayer(program=self.program, queries = self.queries)


    def forward(self, x):

        h = self.image_encoder(x)
        ghosts = self.ghost_layer(h)
        pacman = self.pacman_layer(h)
        base_actions = self.base_policy_layer(h)


        # Map from functors to the probability of their groundings.
        # TODO(giuseppe): We need to fix an order here
        weights = {"ghost": ghosts,
                   "pacman": pacman,
                   "action": base_actions}

        actions = self.dpl_layer(weights)["action"]

        return actions


h = torch.tensor([[10., 2., 7.]])
image_encoder = lambda x: x
image_encoder.hidden_size = 3
dpl_policy = DPLSafePolicy(image_encoder=image_encoder)


print(dpl_policy(h))



