
from functools import reduce
from statistics import mean
from operator import mul

class TruthValue:
    Not = lambda x: (1-x)

    And = lambda *x: reduce(mul, x, 1)
    Or  = lambda *x: 1 - reduce(mul, (1 - xi for xi in x), 1)
    Average = lambda *x: mean(x)

    fc_to_w_plus    = lambda f, c, k: k*f*c/(1-c)
    fc_to_w         = lambda f, c, k: k*c/(1-c)
    fc_to_w_minus   = lambda f, c, k: k*(1-f)*c/(1-c)

    w_to_f          = lambda w_plus, w: w_plus/max(w, 1e-4)
    w_to_c          = lambda w, k     : w/(w+k)

    F_ded = lambda f1, c1, f2, c2: (TruthValue.And(f1, f2), TruthValue.And(f1, f2, c1, c2))  # return: f, c


    def __init__(self, f, c, k=1):
        self.f = f
        self.c = c
        self.k = k
    
    def revision(self, truthvalue: 'TruthValue'):
        w1p = TruthValue.fc_to_w_plus(self.f, self.c, self.k)
        w2p = TruthValue.fc_to_w_plus(truthvalue.f, truthvalue.c, truthvalue.k)
        w1  = TruthValue.fc_to_w(self.f, self.c, self.k)
        w2  = TruthValue.fc_to_w(truthvalue.f, truthvalue.c, truthvalue.k)
        f = TruthValue.w_to_f(w1p + w2p, w1 + w2)
        c = TruthValue.w_to_c(w1 + w2, self.k)
        self.f = f
        self.c = c
        return self
    
    @property
    def w(self):
        return TruthValue.fc_to_w(self.f, self.c, self.k)
    
    @property
    def e(self):
        return (self.c * (self.f - 0.5) + 0.5)

    @staticmethod
    def deduction(truthvalue1: 'TruthValue', truthvalue2: 'TruthValue'):
        f, c = TruthValue.F_ded(truthvalue1.f, truthvalue2.f, truthvalue1.c, truthvalue2.c)
        return TruthValue(f, c, truthvalue1.k)
        
    
    def __repr__(self):
        return f"<TruthValue: %{self.f:.2f};{self.c:.2f}% (k={self.k})>"