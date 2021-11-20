from torch import nn
import torch
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from deepproblog.light.semiring import GraphSemiring
from collections import defaultdict
from numba import njit
import numpy as np

class DeepProbLogLayer(nn.Module):
    def __init__(self, program, queries=[], evidences=[], single_output=None):
        super().__init__()
        query = "\n".join(["query(%s)." % q for q in queries])
        evidence = "\n".join(["evidence(%s)." % e for e in evidences])
        self.program = "\n\n".join([program, evidence, query])
        self.lf = LogicFormula.create_from(self.program)
        self.dag = LogicDAG.create_from(self.lf)
        self.sdd = SDD.create_from(self.dag)
        self.semiring = GraphSemiring()
        self.single_output = single_output

    def forward(self, x):
        """x is a dictionary <key,tensor>."""
        if self.single_output is not None:
            self.semiring.set_weights({self.single_output: x})
        else:
            self.semiring.set_weights(x)
        out = self.sdd.evaluate(semiring=self.semiring)
        stacked = defaultdict(list)
        for k, v in out.items():
            stacked[k.functor].append((k, v))

        tensorial = {}
        for k, v in stacked.items():
            v = [b for a, b in v]
            tensorial[k] = torch.cat(v, dim=-1)

        if self.single_output:
            return tensorial[self.single_output]
        else:
            return tensorial
