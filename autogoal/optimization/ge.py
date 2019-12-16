# -*- coding: utf-8 -*-

import json
import multiprocessing
import random
import sys
import traceback
from queue import Empty
from functools import reduce

import enlighten
import time
import numpy as np
import warnings
import yaml

from .base import Metaheuristic
from .utils import InvalidPipeline


class Individual:
    """Representa un individuo de una gramática probabilística."""
    def __init__(self, values, grammar):
        self.values = values
        self.current = 0
        self.grammar = grammar
        self.state = self._walk('Pipeline')

    def reset(self):
        self.current = 0
        self.state = self._walk('Pipeline')

    def choose(self, *values):
        value = next(self.state)

        if not isinstance(value, tuple):
            raise ValueError('Cannot apply `choose` at this point (%s).' % str(value))

        options, i = value

        if options != len(values):
            raise ValueError('Need to provide exactly %i values.' % options)

        return values[i]

    def next(self):
        value = self.values[self.current]
        self.current += 1
        return value

    def nextint(self):
        value = next(self.state)

        if not isinstance(value, int):
            raise ValueError('Cannot apply `nextint` at this point (%s).' % str(value))

        return value

    def nextfloat(self):
        value = next(self.state)

        if not isinstance(value, float):
            raise ValueError('Cannot apply `nextfloat` at this point (%s).' % str(value))

        return value

    def nextbool(self):
        return self.choose('yes', 'no') == 'yes'

    def sample(self):
        return {'Pipeline': self._sample('Pipeline')}

    def _sample(self, symbol):
        production = self.grammar[symbol]
        value = production.sample(self)

        if isinstance(value, (int, float)):
            return value
        else:
            rule, _, _ = value

            rule_repr = []

            for s in rule.body:
                if s[0].isupper():
                    rule_repr.append({ s: self._sample(s) })
                else:
                    rule_repr.append(s)

            return rule_repr

    def _walk(self, symbol):
        production = self.grammar[symbol]
        value = production.sample(self)

        if isinstance(value, (int, float)):
            yield value
        else:
            rule, options, index = value

            if options > 1:
                yield options, index

            for s in rule.body:
                if s[0].isupper():
                    yield from self._walk(s)


class PGE(Metaheuristic):
    def __init__(self, grammar,
                       popsize=100,
                       selected=0.1,
                       learning=0.05,
                       timeout=None,
                       verbose=False,
                       fitness_evaluations=1,
                       errors='raise',
                       global_timeout=None,
                       maximize=True,
                       incremental=False,
                       start_complexity=0.1):
        """Representa una metaheurística de Evolución Gramatical Probabilística.

        - `popsize`: tamaño de la población
        - `selected`: cantidad de individuos seleccionados en cada paso
        - `learning`: factor de aprendizaje para ajustar las probabilidades
        """
        super().__init__()

        if isinstance(selected, float):
            selected = int(selected * popsize)

        self._grammar = grammar
        self.popsize = popsize
        self.selected = selected
        self.learning = learning
        self.timeout = timeout
        self.verbose = verbose
        self.fitness_evaluations = fitness_evaluations
        self.errors = errors
        self.global_timeout = global_timeout
        self.maximize = maximize
        self.incremental = incremental
        self.start_complexity = start_complexity
        self.indsize = self._grammar.complexity()

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _sample_population(self):
        """Construye la población inicial"""
        population = []

        for _ in range(self.popsize):
            values = []
            for _ in range(self.indsize):
                values.append(random.uniform(0,1))
            population.append(Individual(values, self._grammar))

        return population

    def _select(self, pop, fit):
        """Selecciona los mejores {self.indsize} elementos de la población."""
        sorted_pop = sorted(zip(pop,fit), key=lambda t: t[1], reverse=self.maximize)
        return [t[0] for t in sorted_pop[:self.selected]]

    def _update_model(self, best):
        model = self._grammar.get_model()

        for ind in best:
            ind.reset()
            pipe = ind.sample()

            self._update_ind_model(model, 'Pipeline', pipe['Pipeline'])

        for s, p in model.items():
            p.normalize()

        self._grammar.merge(model, self.learning)
        # self.log(yaml.dump(self._grammar._model))

    def _update_ind_model(self, model, symbol, rule):
        if isinstance(rule, list):
            body = []
            for s in rule:
                if isinstance(s, dict):
                    key = list(s.keys())[0]
                    body.append(key)

                    self._update_ind_model(model, key, s[key])

                elif isinstance(s, str):
                    body.append(s)
                else:
                    raise ValueError("Invalid type for a rule element: %s" % str(s))
            model[symbol].update(body)

        elif isinstance(rule, (int, float)):
            model[symbol].update(rule)

        else:
            raise ValueError("Invalid type for a rule: %s" % str(rule))

    def _evaluate_one(self, ind, q, cmplx):
        try:
            ind.reset()
            pipeline = self._grammar.generate(ind)

            if self.incremental:
                f = self._grammar.evaluate(pipeline, cmplx)
            else:
                f = self._grammar.evaluate(pipeline)

            q.put(f)
        except InvalidPipeline as e:
            self.log("!", str(e))
            q.put(0)
        except Exception as e:
            if self.errors == 'raise':
                _, _, tb = sys.exc_info()
                tb = traceback.format_tb(tb)
                q.put((e, tb))
            elif self.errors == 'warn':
                q.put(0)
                warnings.warn(str(e))

    def _evaluate(self, ind:Individual, manager, evalc, evalc_error, genc, genc_error, cmplx):
        """Computa el fitness de un individuo."""

        self.log(yaml.dump(ind.sample()))
        ind.reset()

        score = 0
        counter = manager.counter(desc='    Current Individual', color='green', unit='run', total=self.fitness_evaluations, leave=False)
        counter_error = counter.add_subcounter('red')

        for i in range(self.fitness_evaluations):

            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=self._evaluate_one, args=(ind, q, cmplx))
            p.start()

            try:
                s = q.get(block=True, timeout=self.timeout)

                if isinstance(s, tuple):
                    e, tb = s
                    if self.errors == 'raise':
                        print("(!) Exception caught in evaluation.\n(!) This is the original traceback.")
                        print("\n".join(tb))
                        print("(!) This is the re-raised exception.")
                        raise e
                    elif self.errors == 'warn':
                        warnings.warn(str(e))
                        counter_error.update()
                        if evalc_error:
                            evalc_error.update()
                        if genc_error:
                            genc_error.update()
                        return 0
                    elif self.errors == 'ignore':
                        counter_error.update()
                        if evalc_error:
                            evalc_error.update()
                        if genc_error:
                            genc_error.update()
                        return 0

                score += s
                if s > 0:
                    counter.update()
                    if evalc:
                        evalc.update()
                    if genc:
                        genc.update()
                else:
                    counter_error.update()
                    if evalc_error:
                        evalc_error.update()
                    if genc_error:
                        genc_error.update()
            except Empty:
                self.log("! Timeout")
                counter_error.update()
                if evalc_error:
                    evalc_error.update()
                if genc_error:
                    genc_error.update()
                return 0
            finally:
                counter.close()

        f = score / self.fitness_evaluations
        self.log("Fitness: %.3f\n----------" % f)
        return f

    def run(self, evals:int):
        """Ejecuta la metaheurística hasta el número de evaluaciones indicado"""

        start_time = time.time()

        it = 0
        self.current_best, self.current_fn = None, 0
        self.pop_avg = []
        self.pop_std = []

        manager = enlighten.get_manager(enabled=self.verbose)
        generation_counter = manager.counter(total=evals * self.popsize * self.fitness_evaluations, color='green', unit='run', desc='Overall [Best = .....]')
        generation_counter.update(0, force=True)
        generation_counter_error = generation_counter.add_subcounter('red')

        while it < evals:
            self.population = self._sample_population()
            self.fitness = []

            cmplx = it * (1 - self.start_complexity) / evals + self.start_complexity
            self.log("Complexity: %.3f" % cmplx)

            if self.current_best is not None and self.incremental:
                self.current_fn = self._evaluate(self.current_best, manager, None, None, None, None, cmplx)
                self.log("Reevaluating best pipeline")

            evaluation_counter = manager.counter(total=self.popsize * self.fitness_evaluations, color='green', unit='run', desc='  Current Population  ', leave=False)
            evaluation_counter.update(0, force=True)
            evaluation_counter_error = evaluation_counter.add_subcounter('red')

            for i in self.population:
                fn = self._evaluate(i, manager, evaluation_counter, evaluation_counter_error, generation_counter, generation_counter_error, cmplx)

                if fn == 0.0:
                    continue

                if any([self.current_best      is  None,
                        self.maximize == True  and fn > self.current_fn,
                        self.maximize == False and fn < self.current_fn]):

                    i.reset()
                    self.current_best = self._grammar.generate(i)
                    self.current_fn = fn

                    generation_counter.desc = 'Overall [Best = %.3f]' % self.current_fn
                    generation_counter.update(0, force=True)

                    self.log("Updated best: ", self.current_fn)

                self.fitness.append(fn)

            evaluation_counter.close()

            self.pop_avg.append(np.mean(self.fitness))
            self.pop_std.append(np.std(self.fitness))

            self.save(GEEncoder)

            if self.global_timeout and time.time() - start_time > self.global_timeout:
                break

            best = self._select(self.population, self.fitness)
            self._update_model(best)
            it += 1

        manager.stop()
        return self.current_best


class GEEncoder(json.encoder.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Individual):
            obj.reset()
            enc = obj.sample()
            obj.reset()
            return { 'values': obj.values, 'repr': enc }


class Production:
    def __init__(self, symbol, rules):
        self.symbol = symbol
        self.rules = rules

    def __repr__(self):
        return "Production(%s,%s)" % (self.symbol, repr(self.rules))

    def merge(self, prod, learning):
        for r1,r2 in zip(prod.rules, self.rules):
            r2.merge(r1, learning)

        self.normalize()

    def size(self, grammar):
        return reduce(lambda x,y: x+y, [r.size(grammar) for r in self.rules])

    def update(self, body):
        for r in self.rules:
            if r.body == body:
                r.prob += 1
                return

        raise ValueError("Invalid body: %s" % body)

    def normalize(self):
        total = sum(r.prob for r in self.rules)

        for r in self.rules:
            if total == 0:
                r.prob = 1.0 / len(self.rules)
            else:
                r.prob /= total

    def complexity(self, grammar):
        if len(self.rules) == 1:
            return self.rules[0].complexity(grammar)

        return 1 + sum(r.complexity(grammar) for r in self.rules)

    def clone(self):
        return Production(self.symbol, [r.clone() for r in self.rules])

    def sample(self, ind:Individual):
        if len(self.rules) == 1:
            return self.rules[0], 1, 0

        value = ind.next()
        p = 0

        for i,r in enumerate(self.rules):
            p += r.prob
            if value <= p:
                return r, len(self.rules), i

        return self.rules[-1], len(self.rules), len(self.rules) - 1


class Rule:
    def __init__(self, body, prob:float):
        self.body = body
        self.prob = prob

    def __repr__(self):
        return "Rule(%s,%s)" % (repr(self.body), self.prob)

    def size(self, grammar):
        return reduce(lambda x,y: x*y, [grammar.size(s) for s in self.body if s[0].isupper()], 1)

    def merge(self, other, learning):
        self.prob = learning * other.prob + (1-learning) * self.prob

    def clone(self):
        return Rule(self.body, 0)

    def complexity(self, grammar):
        c = 0

        for symbol in self.body:
            if symbol[0].isupper():
                c += grammar.complexity(symbol)

        return c


class IntProduction(Production):
    def __init__(self, symbol, min:int, max:int):
        self.symbol = symbol
        self.min = min
        self.max = max
        self.mean = (max + min) / 2
        self.dev = (max - min) / 2
        self.values = []

    def merge(self, other, learning):
        self.mean = learning * other.mean + (1-learning) * self.mean
        self.dev = learning * other.dev + (1-learning) * self.dev

    def __repr__(self):
        return "IntProduction(%s,%i,%i,%f,%f)"  % (self.symbol, self.min, self.max, self.mean, self.dev)

    def size(self, grammar):
        return 1

    def update(self, value):
        self.values.append(value)

    def clone(self):
        return IntProduction(self.symbol, self.min, self.max)

    def normalize(self):
        if self.values:
            self.mean = sum(self.values) / len(self.values)
            self.dev = (max(self.values) - min(self.values)) / 2
            self.values.clear()

    def complexity(self, grammar):
        return 1

    def sample(self, ind:Individual):
        value = int(self.mean + self.dev * (ind.next() - 0.5) * 2)
        return max(self.min, min(self.max, value))


class FloatProduction(IntProduction):
    def clone(self):
        return FloatProduction(self.symbol, self.min, self.max)

    def __repr__(self):
        return "FloatProduction(%s,%f,%f,%f,%f)"  % (self.symbol, self.min, self.max, self.mean, self.dev)

    def sample(self, ind:Individual):
        value = self.mean + self.dev * (ind.next() - 0.5) * 2
        return max(self.min, min(self.max, value))


class Grammar:
    def __init__(self):
        self._model = {}

        for symbol, productions in self.grammar().items():
            self._model[symbol] = []
            productions = productions.split('|')

            if len(productions) == 1:
                p = productions[0]

                if p.startswith('f('):
                    min, max = tuple(float(i) for i in p[2:-1].split(','))
                    self._model[symbol] = FloatProduction(symbol, min, max)
                    continue
                if p.startswith('i('):
                    min, max = tuple(int(i) for i in p[2:-1].split(','))
                    self._model[symbol] = IntProduction(symbol, min, max)
                    continue

            rules = []

            for p in productions:
                if p.startswith('f(') or p.startswith('i('):
                    raise ValueError('Numeric rules must be the only ones.')

                rules.append(Rule(p.split(), 1))

            production = Production(symbol, rules)
            production.normalize()

            self._model[symbol] = production

    def get_model(self):
        model = {}

        for symbol, prod in self._model.items():
            model[symbol] = prod.clone()

        return model

    def merge(self, model, learning):
        for symb, prod in model.items():
            self._model[symb].merge(prod, learning)

    def __getitem__(self, key):
        return self._model[key]

    def evaluate(self, ind:Individual) -> float:
        """Recibe un elemento de la gramática y devuelve un valor de fitness creciente."""
        raise NotImplementedError()

    def generate(self, ind:Individual):
        """Optionally performs a mapping from individual to a custom pipeline representation
        that will be later pased on to the evaluate method."""
        return ind

    def grammar(self):
        raise NotImplementedError()

    def complexity(self, symbol='Pipeline'):
        """Calcula la máxima complejidad de una solución en la gramática."""
        return self._model[symbol].complexity(self)

    def size(self, symbol='Pipeline'):
        return self._model[symbol].size(self)
