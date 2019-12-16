# coding: utf8

from hpopt.ge import Grammar, PGE
import numpy as np


Mercury = [0.38710,    87.9693]
Venus   = [0.72333,   224.7008]
Earth   = [1.00000,   365.2564]
Mars    = [1.52366,   686.9796]
Saturn  = [9.53707, 10775.599]
Uranus  = [19.1913, 30687.153]
Neptune = [30.0690, 60190.03]
Jupiter = [5.20336,  4332.8201]


values = [Mercury, Venus, Earth, Mars, Saturn, Uranus, Neptune, Jupiter]
radius = np.asarray([v[0] for v in values])
period = np.asarray([v[1] for v in values])


class KeplerGrammar(Grammar):
    def grammar(self):
        return {
            'Pipeline': 'Mult | Sum',
            'Mult':     'ExpR ExpT',
            'Sum':      'ExpR ExpT',
            'ExpR':     'i(-5,5)',
            'ExpT':     'i(-5,5)',
        }

    def evaluate(self, ind):
        results = ind.choose(self.mult, self.sum)(ind)
        return abs(results.mean() - 7.50)

    def mult(self, ind):
        exp_r = self.exp_r(ind)
        exp_t = self.exp_t(ind)

        return (radius ** exp_r) * (period ** exp_t)

    def sum(self, ind):
        exp_r = self.exp_r(ind)
        exp_t = self.exp_t(ind)

        return (radius ** exp_r) + (period ** exp_t)

    def exp_r(self, ind):
        return ind.nextint()

    def exp_t(self, ind):
        return ind.nextint()


def main():
    ge = PGE(KeplerGrammar(), learning=0.05, verbose=True, maximize=False)
    ge.run(10)


if __name__ == "__main__":
    main()

