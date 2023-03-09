from torch import nn
from collections import defaultdict
from problog.formula import LogicFormula, LogicDAG
from problog.ddnnf_formula import DDNNF
from pls.deepproblog.light.semiring import GraphSemiring
import torch as th


def compile_problog(program, queries=[], evidences=[], input_struct=None, query_struct=None,
                    single_output=None):
    query = "\n".join(["query(%s)." % q for q in queries])
    evidence = "\n".join(["evidence(%s)." % e for e in evidences])
    program = "\n\n".join([program, evidence, query])
    lf = LogicFormula.create_from(program)
    dag = LogicDAG.create_from(lf)
    ddnnf = DDNNF.create_from(dag)

    return ddnnf

def evaluate_problog(ddnnf, input_struct, single_output=None):
    x = {key: th.full((1, len(value)), 0.2) for key, value in input_struct.items()}

    semiring = GraphSemiring()
    if single_output is not None:
        semiring.set_weights({single_output: x})
    else:
        semiring.set_weights(x)
    out = ddnnf.evaluate(semiring=semiring)
    return out


class DeepProbLogLayer_Approx(nn.Module):
    """
    An optimized ProbLog implementation for a specific class of ProbLog programs. Supported programs must:
    (1) have at most one annotated disjunction
    (2) the AD (if exists) must be the predicate 'action'
    """
    def __init__(self, program, queries=[], evidences=[], input_struct=None, query_struct=None,
                 single_output=None):
        super().__init__()
        # Initialize a problog program
        query = "\n".join(["query(%s)." % q for q in queries])
        evidence = "\n".join(["evidence(%s)." % e for e in evidences])
        self.program = "\n\n".join([program, evidence, query])
        self.lf = LogicFormula.create_from(self.program)
        self.dag = LogicDAG.create_from(self.lf)
        self.ddnnf = DDNNF.create_from(self.dag)
        self.semiring = GraphSemiring()
        self.single_output = single_output

        # Prepare for optimization
        self.input_struct = input_struct
        self.n_facts = sum(len(input_struct[entry]) for entry in input_struct if "action" not in entry)
        self.n_actions = len(input_struct["action"])
        self.n_ads = len([entry for entry in input_struct if "action" in entry])
        self.n_queries = len(queries)
        self.query_struct = query_struct

        # Compile the program
        self._init()


    def _init(self):
        n_facts = self.n_facts
        # Initialize all possible worlds
        # Enumerate the facts of the worlds
        w_facts = th.zeros((2 ** n_facts, n_facts))
        for i in range(2 ** n_facts):
            bin_i = f"{bin(i)[2:]}".zfill(n_facts)
            for j in range(n_facts):
                w_facts[i][j] = float(bin_i[j])
        w_facts = w_facts.repeat_interleave(self.n_actions ** self.n_ads, dim=0)
        # Enumerate the ADs of the worlds
        if self.n_ads == 1:
            w_actions = th.eye(self.n_actions)
            w_actions = w_actions.repeat(2 ** n_facts, 1)
        else:
            # TODO: confirm this case works
            w_actions = th.cat((th.eye(self.n_actions).repeat_interleave(self.n_actions, dim=0), th.eye(self.n_actions).repeat(self.n_actions, 1)),dim=1)
            w_actions = w_actions.repeat(2 ** n_facts, 1)
        worlds = th.cat((w_facts, w_actions), dim=1)
        n_worlds = 2 ** n_facts  * self.n_actions ** self.n_ads

        # Filter valid worlds
        valid_worlds = []
        w_queries = []
        # TODO: make parallel
        for world_ix in range(n_worlds):
            x = self.worlds_to_dict(worlds[world_ix].reshape(1, -1))
            q_dict = self.calculate_complete_w(x=x)
            q_tensor = self.dict_to_worlds(self.query_struct, q_dict)
            if not q_tensor[0][0].isnan():
                w_queries.append(q_tensor)
                valid_worlds.append(worlds[world_ix])

        w_queries = th.stack(w_queries, dim=0)[:, 0]
        self.w_queries = w_queries
        self.w = th.stack(valid_worlds, dim=0)
        self.w_facts = self.w[:, 0:n_facts]
        self.w_actions = self.w[:, n_facts:]



    def worlds_to_dict(self, ww):
        x = {k: ww[:, v[0]:v[0]+len(v)] for k, v in self.input_struct.items()}
        return x

    def dict_to_worlds(self, struct, d):
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
        xx = self.dict_to_worlds(self.input_struct, x)
        p_facts = xx[:, :self.n_facts]
        p_ads = xx[:, self.n_facts:]
        eps = 1e-9
        lp_facts = th.log(p_facts + eps)
        lnp_facts = th.log(1 - p_facts + eps)
        lp_ads = th.log(p_ads + eps)
        w_facts = self.w_facts.float()
        w_queries = self.w_queries.float()
        lp_w = lp_facts @ w_facts.T + lnp_facts @ (1 - w_facts.T) + lp_ads @ self.w_actions.T
        p_w = th.exp(lp_w)
        p_queries = p_w @ w_queries
        results = self.tensor_to_dict(p_queries)

        return results

    def calculate_complete_w(self, x):
        """x is a dictionary <key,tensor>."""
        if self.single_output is not None:
            self.semiring.set_weights({self.single_output: x})
        else:
            self.semiring.set_weights(x)
        out = self.ddnnf.evaluate(semiring=self.semiring)
        ss = defaultdict(dict)
        for k, v in out.items():
            if k.arity > 0:
                ss[k.functor][self.query_struct[k.functor][str(k.args[0])]] = v
            else:
                ss[k.functor][self.query_struct[k.functor]] = v
        tensorial = {}
        for k, v in ss.items():
            v = [b for a, b in sorted(ss[k].items())]
            tensorial[k] = th.cat(v, dim=-1)

        if self.single_output:
            return tensorial[self.single_output]
        else:
            return tensorial
