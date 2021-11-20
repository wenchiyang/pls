from problog.evaluator import Semiring

class GraphSemiring(Semiring):
    def __init__(self):
        Semiring.__init__(self)
        self.eps = 1e-12

    def is_zero(self, value):
        return False

    def is_one(self, value):
        return False

    def negate(self, a):
        """Returns the negation of the probability a: 1-a."""
        return 1.0 - a

    def one(self):
        """Returns the identity element of the multiplication."""
        return 1.0

    def zero(self):
        """Returns the identity element of the addition."""
        return 0.0

    def plus(self, a, b):
        """Computes the addition of the given values."""
        return a + b

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        return a * b

    def set_weights(self, weights):
        self.weights = weights

    def normalize(self, a, z):
        return a / z

    def value(self, a):
        """Transform the given external value into an internal value."""
        i = int(a.args[0])
        v = self.weights[a.functor][:, i : i + 1]
        return v
