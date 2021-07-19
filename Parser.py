from Grammar import *
import itertools
from time import perf_counter

# similar to Pred, but with id-equality
class Chunk:
    instances = {}
    def __new__(cls, symbol, inputs):
        key = symbol, tuple(inputs)
        i = cls.instances.get(key)
        if i:
           return i
        i = super(Chunk, cls).__new__(cls)
        i.symbol = symbol
        i.inputs = inputs
        cls.instances[key] = i
        return i


def parse(grammar, string):
    chart = {n: {} for n in grammar.non_terminals}   # sym -> {Chunk -> prob}
    agenda = {}  # Chunk -> prob
    goal = Chunk(grammar.start, [(0, len(string))])
    scan(grammar, string, agenda)
    start = perf_counter()
    while agenda and goal not in chart[grammar.start] and perf_counter() < start+5:
        (current_best, weight) = max(agenda.items(), key=lambda x: x[1])
        chart[current_best.symbol][current_best] = weight
        del agenda[current_best]
        update_agenda(grammar, agenda, chart, current_best)
    if goal not in chart[grammar.start]:
        if perf_counter() > start+5:
            print("Timeout! Grammar: ", grammar, "\nString: ", string)
        return None
    else:
        return -chart[grammar.start][goal]


def update_agenda(grammar, agenda, chart, new_item):
    for rule in grammar.prules:
        if not rule.terminating and any(new_item.symbol == prod.symbol for prod in rule.right):
            right = list(rule.right)
            for perm in itertools.product(*(chart[prod.symbol].keys() for prod in right)):
                if new_item in perm:
                  sat = satisfies(perm, rule.left, right, chart)
                  if sat is not None:
                      sat[1] += rule.prob
                      if isNew(sat[0], chart, agenda, sat[1]):
                          agenda[sat[0]] = sat[1]


def satisfies(perm, left, right, chart):
    inp_conv = {}
    new_prob = 0
    for chunk, pred in zip(perm, right):
        for c_i, p_i in zip(chunk.inputs, pred.inputs):
            inp_conv[p_i] = c_i
        new_prob += chart[chunk.symbol][chunk]
    new_res = []
    for r in left.inputs:
        if len(r) == 1:
            new_res.append(inp_conv[r])
        else:
            lst = [inp_conv[c] for c in r]
            for k in range(len(lst)-1):
                if lst[k][1] != lst[k+1][0]:
                    return None
            new_res.append((lst[0][0], lst[-1][1]))
    if len(new_res) > 1:
        for i in range(len(new_res) - 1):
            if new_res[i][1] > new_res[i+1][0]:
                return None
    return [Chunk(left.symbol, new_res), new_prob]


def isNew(v, chart, agenda, new_prob):
    if v in chart[v.symbol]:
        return False
    else:
        return (v not in agenda) or (agenda[v] != new_prob)


def scan(grammar, string, agenda):
    for i in range(len(string)):
        for rule in grammar.prules:
            if rule.terminating:
                if rule.left.inputs[0] == string[i]:
                    agenda[Chunk(rule.left.symbol, [(i, i+1)])] = rule.prob
