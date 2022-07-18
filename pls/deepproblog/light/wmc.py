from torch import nn
from collections import defaultdict
from problog.formula import LogicFormula, LogicDAG
from problog.ddnnf_formula import DDNNF
from deepproblog.light.semiring import GraphSemiring
import torch as th

class DeepProbLogLayer_Approx(nn.Module):
    def __init__(self, program, queries=[], evidences=[], input_struct=None, query_struct=None,
                 single_output=None):
        super().__init__()
        query = "\n".join(["query(%s)." % q for q in queries])
        evidence = "\n".join(["evidence(%s)." % e for e in evidences])
        self.program = "\n\n".join([program, evidence, query])
        self.lf = LogicFormula.create_from(self.program)
        self.dag = LogicDAG.create_from(self.lf)
        self.ddnnf = DDNNF.create_from(self.dag)
        self.semiring = GraphSemiring()
        self.single_output = single_output

        self.input_struct = input_struct
        self.n_facts = sum(len(input_struct[entry]) for entry in input_struct if "action" not in entry)
        self.n_actions = len(input_struct["action"])
        self.n_ads = len([entry for entry in input_struct if "action" in entry])
        self.n_queries = len(queries)
        self.query_struct = query_struct
        self._init()


    def _init(self):
        nf = self.n_facts
        # initialize fact part of w
        w_facts = th.zeros((2 ** nf, nf))
        for i in range(2 ** nf):
            bin_i = f"{bin(i)[2:]}".zfill(nf)
            for j in range(nf):
                w_facts[i][j] = float(bin_i[j])
        w_facts = w_facts.repeat_interleave(self.n_actions ** self.n_ads, dim=0)

        # initialize ad part of w
        if self.n_ads == 1:
            w_actions = th.eye(self.n_actions)
            w_actions = w_actions.repeat(2 ** nf, 1)
        else:
            w_actions = th.cat((th.eye(self.n_actions).repeat_interleave(self.n_actions, dim=0), th.eye(self.n_actions).repeat(self.n_actions, 1)),dim=1)
            w_actions = w_actions.repeat(2 ** nf, 1)

        w = th.cat((w_facts, w_actions), dim=1)

        n_worlds = 2 ** nf  * self.n_actions ** self.n_ads

        valid_w = []
        # query (need problog)
        w_queries = []
        for r in range(n_worlds):
            x = self.world_tensor_to_dict(w[r].reshape(1, -1))
            q_dict = self.calculate_complete_w(x=x)
            q_tensor = self.dict_to_tensor(self.query_struct, q_dict)
            if not q_tensor[0][0].isnan():
                w_queries.append(q_tensor)
                valid_w.append(w[r])

        w_queries = th.stack(w_queries, dim=0)
        self.w_queries = w_queries
        self.w = th.stack(valid_w, dim=0)
        self.w_facts = self.w[:, 0:self.n_facts]
        self.w_actions = self.w[:, self.n_facts:]



    def world_tensor_to_dict(self, ww):
        x = {k: ww[:, v[0]:v[0]+len(v)] for k, v in self.input_struct.items()}
        return x

    def dict_to_tensor(self, struct, d):
        a = []
        for k in struct:
            a.append(d[k])
        t = th.cat(a, dim=1)
        return t

    def tensor_to_dict(self, t):
        x = dict()
        count = 0
        for k, v in self.query_struct.items():
            if type(v) is dict:
                x[k] = t[:, count:count + len(v)]
                count += len(v)
            else:
                x[k] = t[:, count:count + 1]
                count += 1
        return x

    def forward(self, x):
        xx = self.dict_to_tensor(self.input_struct, x)
        eps = 1e-9
        lp = th.log(xx + eps)
        lnp = th.log(1 - xx + eps)
        w = self.w.float()
        w_queries = self.w_queries.float()
        w_queries = w_queries[:, 0]
        logp = lp @ w.T + lnp @ (1 - w.T)
        p = th.exp(logp)
        temp = p @ w_queries / th.sum(p, axis=1, keepdim=True)
        results = self.tensor_to_dict(temp)

        return results

    def calculate_complete_w(self, x):
        """x is a dictionary <key,tensor>."""
        if self.single_output is not None:
            self.semiring.set_weights({self.single_output: x})
        else:
            self.semiring.set_weights(x)
        out = self.ddnnf.evaluate(semiring=self.semiring)
        # stacked = defaultdict(list)
        ss = defaultdict(dict)
        for k, v in out.items():
            # stacked[k.functor].append((k, v))
            if k.arity > 0:
                ss[k.functor][self.query_struct[k.functor][str(k.args[0])]] = v
            else:
                ss[k.functor][self.query_struct[k.functor]] = v
        # tensorial = {}
        # for k, v in stacked.items():
        #     v = [b for a, b in v]
        #     tensorial[k] = th.cat(v, dim=-1)

        tensorial = {}
        for k, v in ss.items():
            v = [b for a, b in sorted(ss[k].items())]
            tensorial[k] = th.cat(v, dim=-1)

        if self.single_output:
            return tensorial[self.single_output]
        else:
            return tensorial

def test1():
    queries = [
        "safe_action(one)",
        "safe_action(two)",
    ]
    evidences = ["safe_next"]
    program = """ 
f(0)::f(one).
f(1)::f(two).

action(0)::action(one);
action(1)::action(two).

safe_next :- f(one), action(one).
safe_next :- f(two), action(two).

safe_action(A):- action(A), safe_next.
"""
    input_struct = {"f": th.tensor([i for i in range(2)]),
                    "action": th.tensor([i for i in range(2, 4)])}
    query_struct = {"safe_action": th.tensor([i for i in range(2)])}
    dpl_layer = DeepProbLogLayer_Approx(program=program, queries=queries, evidences=evidences,
                                        input_struct=input_struct, query_struct=query_struct,
                                        )

    a = th.tensor([[0.2, 0.6]])
    b = th.tensor([[0.1, 0.9]])
    results = dpl_layer(
        x={
            "f": a,
            "action": b
        }
    )
    print(results)

def test2():
    queries = [
        "safe_action(no_op)",
        "safe_action(push_up)",
        "safe_action(push_down)",
        "safe_action(push_left)",
        "safe_action(push_right)"
    ]
    queries += [
        "box(0, 1)",
        "box(-1, 0)",
        "box(1, 0)",
        "box(0, -1)",
        "corner(0, 2)",
        "corner(-2, 0)",
        "corner(2, 0)",
        "corner(0, -2)",
    ]
    evidences = ["safe_next"]
    program = """ 
action(0):: action(no_op);      % 0
action(1):: action(push_up);    % 1
action(2):: action(push_down);  % 2
action(3):: action(push_left);  % 3
action(4):: action(push_right). % 4

box(0):: box( 0, 1). % 5
box(1):: box(-1, 0). % 9
box(2):: box( 1, 0). % 10
box(3):: box( 0,-1). % 14

corner(0):: corner( 0, 2). % 2
corner(1):: corner(-2, 0). % 8
corner(2):: corner( 2, 0). % 11
corner(3):: corner( 0,-2). % 17

box_transition( X,  Y, no_op,       X,  Y).
box_transition(-1,  0, push_left,  -2,  0).
box_transition( 1,  0, push_right,  2,  0).
box_transition( 0,  1, push_up,     0,  2).
box_transition( 0, -1, push_down,   0, -2).
box_transition( X,  Y, push_left,   X,  Y):- \+(X =:= -1, Y =:= 0).
box_transition( X,  Y, push_right,  X,  Y):- \+(X =:=  1, Y =:= 0).
box_transition( X,  Y, push_up,     X,  Y):- \+(X =:=  0, Y =:= 1).
box_transition( X,  Y, push_down,   X,  Y):- \+(X =:=  0, Y =:= -1).


unsafe_next :-
    box( X,  Y),
    action(A),
    box_transition(X, Y, A, NextX, NextY),
    corner(NextX, NextY).


safe_next:- \+unsafe_next.
safe_action(A):- action(A), safe_next.
"""
    input_struct = {"box": [i for i in range(4)], "corner": [i for i in range(4,8)], "action": [i for i in range(8,13)]}
    query_struct = {"box": [i for i in range(4)], "corner": [i for i in range(4,8)], "safe_action": [i for i in range(8,13)]}
    dpl_layer = DeepProbLogLayer_Approx(program=program, queries=queries, evidences=evidences,
                                        input_struct=input_struct, query_struct=query_struct,
                                        )

    a = th.tensor([[0.9, 0.2, 0.9, 0.4]])
    b = th.tensor([[0.8, 0.2, 0.5, 0.4]])
    c = th.tensor([[0.0, 0.6, 0.1, 0.2, 0.1]])

    results = dpl_layer(
        x={
            "box": a,
            "corner": b,
            "action": c
        }
    )
    print(results)
