import attr
import math
import random
import string
 
@attr.s(frozen=True, auto_detect=True)
class Pred:
    symbol = attr.ib()
    inputs = attr.ib()

    def __attrs_post_init__(self):
        if not isinstance(self.inputs, tuple):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "dim", len(self.inputs))
        object.__setattr__(self, "_str", self.symbol + str(list(self.inputs)))

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        return self._str == other._str

    def __lt__(self, other):
        return self._str < other._str

    def __hash__(self):
        return hash(self._str)


@attr.s(frozen=True, auto_detect=True)
class PRule:
    left = attr.ib()
    right = attr.ib()
    prob = attr.ib(default=None)
    terminating = attr.ib(default=True, init=False)

    def __attrs_post_init__(self):
        if self.right is not None:
            if not isinstance(self.right, frozenset):
                object.__setattr__(self, "right", frozenset(self.right))
            object.__setattr__(self, "terminating", False)
            object.__setattr__(self, "_str", str(self.left) + " --> " + str(sorted(self.right)))
        else:
            object.__setattr__(self, "_str", str(self.left) + " --> None")

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        return self._str == other._str

    def __lt__(self, other):
        return self._str < other._str

    def __hash__(self):
        return hash(self._str)


@attr.s(frozen=True, auto_detect=True)
class Grammar:
    terminals = attr.ib()
    non_terminals = attr.ib()
    variables = attr.ib()
    prules = attr.ib()
    start = attr.ib(default="S")

    def __attrs_post_init__(self):
        if not isinstance(self.terminals, frozenset):
            object.__setattr__(self, "terminals", frozenset(self.terminals))
        if not isinstance(self.non_terminals, frozenset):
            object.__setattr__(self, "non_terminals", frozenset(self.non_terminals))
        if not isinstance(self.variables, frozenset):
            object.__setattr__(self, "variables", frozenset(self.variables))
        if not isinstance(self.prules, frozenset):
            object.__setattr__(self, "prules", frozenset(self.prules))
        object.__setattr__(self, "_str", "\n".join(str(rule) for rule in sorted(self.prules)))

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        return self._str == other._str

    def __hash__(self):
        return hash(self._str)

    def getEncodingLength(self):
        lhs_bits = math.log(2 + len(self.variables) + len(self.terminals), 2)
        rhs_bits = math.log(1 + len(self.variables), 2)
        symbol_bits = math.log(1 + len(self.non_terminals), 2)
        total = 0
        for rule in self.prules:
            symbols = 1  # left non-terminal
            rule_count = rule.left.dim  # count $ between variables + "#"
            for i in rule.left.inputs:  # count variables on the left side
                rule_count += len(i)
            total += rule_count * lhs_bits
            rule_count = 0
            if rule.right is not None:
                for pred in rule.right:
                    symbols += 1  # one symbol
                    rule_count += pred.dim + 1  # x inputs, then #
            total += rule_count * rhs_bits + symbols * symbol_bits
        total += (len(self.prules) - 1) * rhs_bits  # add extra # between rules
        return total

    def createNeighbor(self):
        neighbor_type = [
            self.replace_non_terminal_with_its_expansion,
            self.new_terminating_rule,
            self.connect_two_non_terminals,
            self.concatenate_a_vector,
            self.add_ignored_non_terminal,
            self.delete_ignored_non_terminal,
            self.ignore_variable,
            self.use_ignored_variable,
            self.split_long_input,
            self.swap_inputs,
            self.swap_tokens,
            self.paste_inputs,
            self.mutate_terminal,
            self.mutate_non_terminal,
            self.mutate_lhs_symbol,
            self.delete_a_rule]

        res = None
        while res is None or res == self:
            func = random.choice(neighbor_type)
            res = func()
        error = res.validate()
        if error is not None:
            print("Invalid grammar: ", error, res, "\nProduced by: ", func)
        return res.fix_probabilities()

    def update_rules(self, rules, nt=None):
        if nt is None:
            nt = self.non_terminals
        return Grammar(self.terminals, nt, self.variables, rules, self.start)

    def replace_rule(self, old_rule, new_rule, vs=None):
        if vs is None:
            vs = self.variables
        return Grammar(self.terminals, self.non_terminals, vs,
                       (self.prules - {old_rule}) | {new_rule},
                       self.start)

    def delete_a_rule(self):
        if len(self.prules) < 2:
            return None
        deleted = random.choice(list(self.prules))
        if deleted.left.symbol == self.start and len([rule for rule in self.prules if rule.left.symbol == self.start]) == 1:
            return None
        return self.update_rules(self.prules - {deleted})

    def replace_non_terminal_with_its_expansion(self):
        possible_expansions = [rule for rule in self.prules if rule.left.dim == 1 and not rule.terminating]
        if not possible_expansions:
            return None
        expansion = random.choice(possible_expansions)
        expansible_rules = [rule for rule in self.prules if not rule.terminating
                            and len(rule.right) + len(expansion.right) < 5
                            and any(pred.symbol == expansion.left.symbol for pred in rule.right)]
        if not expansible_rules:
            return None
        old_rule = random.choice(expansible_rules)
        target = random.choice([nt for nt in old_rule.right if nt.symbol == expansion.left.symbol])
        replacement = expansion.left.inputs[0]
        vars_used = [v for v in self.variables if any(v in pred.inputs for pred in expansion.right)]
        new_vars = []
        vs = list(self.variables)
        for v in vars_used:
             new_var = self.find_unused_var(vs, old_rule, set(new_vars))
             new_vars.append(new_var)
        vars_dict = dict(zip(vars_used, new_vars))
        replacement = "".join(vars_dict.get(sym,sym) for sym in replacement)
        rhs = old_rule.right - {target}
        try:
            rhs |= {Pred(pred.symbol, [vars_dict[v] for v in pred.inputs]) for pred in expansion.right}
        except Exception as e:
            print(vars_used, new_vars, expansion, old_rule)
            raise e
        new_rule = PRule(Pred(old_rule.left.symbol,
                             [v.replace(target.inputs[0], replacement) for v in old_rule.left.inputs]),
                         rhs)
        return self.replace_rule(old_rule, new_rule, vs)

    def random_non_terminal_rule(self):
        options = [rule for rule in self.prules if not rule.terminating]
        return random.choice(options) if options else None

    def add_ignored_non_terminal(self):
        rules = [rule for rule in self.prules if not rule.terminating and len(rule.right) < 4]
        if not rules:
            return None
        old_rule = random.choice(rules)
        preds = [rule.left for rule in self.prules]
        if old_rule.left.symbol != self.start:
            preds = [pred for pred in preds if pred.symbol != self.start]
        pred = random.choice(preds)
        v = []
        vs = list(self.variables)
        while len(v) < pred.dim:
            v.append(self.find_unused_var(vs, old_rule, set(v)))
        new_rule = PRule(old_rule.left, old_rule.right | {Pred(pred.symbol, v)})
        return self.replace_rule(old_rule, new_rule, vs)

    def delete_ignored_non_terminal(self):
        old_rule = self.random_non_terminal_rule()
        if not old_rule:
            return None
        ignored_nts = [pred for pred in old_rule.right
                       if not any(v in i for v in pred.inputs for i in old_rule.left.inputs)]
        if not ignored_nts:
            return None
        new_rule = PRule(old_rule.left, old_rule.right - {random.choice(ignored_nts)})
        return self.replace_rule(old_rule, new_rule)

    def replace_lhs_inputs(self, rule, deleted, replacement):
        inputs = [replacement if i == deleted else i for i in rule.left.inputs]
        return PRule(Pred(rule.left.symbol, inputs), rule.right)

    def ignore_variable(self):
        old_rule = self.random_non_terminal_rule()
        if not old_rule:
            return None
        long_inputs = [i for i in old_rule.left.inputs if len(i)>1]
        if not long_inputs:
            return None
        deleted = random.choice(long_inputs)
        replacement = deleted.replace(random.choice(deleted), "")
        new_rule = self.replace_lhs_inputs(old_rule, deleted, replacement)
        return self.replace_rule(old_rule, new_rule)

    def ignored_vs(self, rule):
        return [v for pred in rule.right for v in pred.inputs
                if not any(v in i for i in rule.left.inputs)]

    def use_ignored_variable(self):
        options = [rule for rule in self.prules if not rule.terminating and self.ignored_vs(rule)]
        if not options:
            return None
        old_rule = random.choice(options)
        ignored_vs = self.ignored_vs(old_rule)
        deleted = random.choice(old_rule.left.inputs)
        replacement = list(deleted)
        replacement.insert(random.randint(0, len(deleted)), random.choice(ignored_vs))
        new_rule = self.replace_lhs_inputs(old_rule, deleted, "".join(replacement))
        return self.replace_rule(old_rule, new_rule)

    def delete_unreachable_rules(self):
        rules = [PRule(rule.left, None if rule.terminating else
                       {pred for pred in rule.right if any(v in i for v in pred.inputs for i in rule.left.inputs)})
                 for rule in self.prules]
        reachables = [self.start]
        for sym in reachables:
            for rule in rules:
                if not rule.terminating:
                    if rule.left.symbol == sym:
                        for var in rule.right:
                            if var.symbol not in reachables:
                                reachables.append(var.symbol)
        rules = [rule for rule in rules if rule.left.symbol in reachables]
        used_variables = {sign for rule in rules for inp in rule.left.inputs for sign in inp if sign not in self.terminals}
        return Grammar(self.terminals, reachables, used_variables, rules, self.start)

    def split_input(self, vs, rule, nt, i):
        if rule.terminating:
            return rule
        vars_dict = {}
        claimed = set()
        right = []
        for pred in rule.right:
            if pred.symbol == nt:
                v = self.find_unused_var(vs, rule, claimed)
                vars_dict[pred.inputs[i]] = pred.inputs[i] + v
                claimed.add(v)
                inputs = list(pred.inputs)
                inputs.insert(i + 1, v)
                right.append(Pred(nt, inputs))
            else:
                right.append(pred)
        left_inputs = ["".join(vars_dict.get(sym, sym) for sym in ii) for ii in rule.left.inputs]
        if rule.left.symbol == nt:
            old = left_inputs[i]
            split = random.randint(1, len(old) - 1)
            left_inputs[i:i+1] = old[:split], old[split:]
        return PRule(Pred(rule.left.symbol, left_inputs), right)

    def split_long_input(self):
        preds = {(rule.left.symbol, rule.left.dim) for rule in self.prules}
        options = [(nt, i) for nt, dim in preds for i in range(dim) if nt != self.start
                   and all(len(rule.left.inputs[i]) > 1 for rule in self.prules if rule.left.symbol == nt)]
        if not options:
            return None
        nt, i = random.choice(options)
        vs = list(self.variables)
        rules = [self.split_input(vs, rule, nt, i) for rule in self.prules]
        return Grammar(self.terminals, self.non_terminals, vs, rules, self.start)

    def paste_input(self, rule, nt, i):
        if rule.terminating:
            return rule
        drop = set()
        right = []
        for pred in rule.right:
            if pred.symbol == nt:
                drop.add(pred.inputs[i])
                inputs = list(pred.inputs)
                inputs.pop(i)
                right.append(Pred(nt, inputs))
            else:
                right.append(pred)
        left_inputs = ["".join(sym for sym in ii if sym not in drop) for ii in rule.left.inputs]
        if rule.left.symbol == nt:
            left_inputs[i - 1] += left_inputs[i]
            left_inputs.pop(i)
        if "" in left_inputs:
            return None
        return PRule(Pred(rule.left.symbol, left_inputs), right)

    def paste_inputs(self):
        options = [rule.left for rule in self.prules if rule.left.dim > 1]
        if not options:
            return None
        lhs = random.choice(options)
        i = random.randint(1, lhs.dim - 1)
        rules = [self.paste_input(rule, lhs.symbol, i) for rule in self.prules]
        if any(rule is None for rule in rules):
        # `None in rules` wouldn't work, as we'd like to keep PRule.__eq__ as simple as possible
            return None
        return self.update_rules(rules)

    def swap_inputs(self):
        options = [rule for rule in self.prules if rule.left.dim > 1]
        if not options:
            return None
        old_rule = random.choice(options)
        i, j = random.sample(range(old_rule.left.dim), 2)
        inputs = list(old_rule.left.inputs)
        inputs[i], inputs[j] = inputs[j], inputs[i]
        new_rule = PRule(Pred(old_rule.left.symbol, inputs), old_rule.right)
        return self.replace_rule(old_rule, new_rule)

    def swap_tokens(self):
        options = [(rule, i) for rule in self.prules for i in rule.left.inputs if len(i) > 1]
        if not options:
            return None
        old_rule, deleted = random.choice(options)
        i, j = random.sample(range(len(deleted)), 2)
        replacement = list(deleted)
        replacement[i], replacement[j] = replacement[j], replacement[i]
        new_rule = self.replace_lhs_inputs(old_rule, deleted, "".join(replacement))
        return self.replace_rule(old_rule, new_rule)

    def mutate_terminal(self):
        options = [rule for rule in self.prules if rule.terminating]
        if not options:
            return None
        old_rule = random.choice(options)
        others = list(set(self.terminals) - {old_rule.left.symbol})
        new_rule = PRule(Pred(old_rule.left.symbol, [random.choice(others)]), None)
        return self.replace_rule(old_rule, new_rule)

    def mutate_non_terminal(self):
        old_rule = self.random_non_terminal_rule()
        if not old_rule:
            return None
        pred = random.choice(list(old_rule.right))
        others = {rule.left.symbol for rule in self.prules if rule.left.dim == pred.dim}
        if old_rule.left.symbol != self.start:
            others -= {self.start}
        if pred.inputs == old_rule.left.inputs:
            others -= {old_rule.left.symbol}
        if not others:
            return None
        nt = random.choice(list(others))
        replacement = Pred(nt, pred.inputs)
        new_rule = PRule(old_rule.left, (old_rule.right - {pred}) | {replacement})
        return self.replace_rule(old_rule, new_rule)

    def mutate_lhs_symbol(self):
        old_rule = random.choice(list(self.prules))
        others = {rule.left.symbol for rule in self.prules if rule.left.dim == old_rule.left.dim}
        if not old_rule.terminating:
            others -= {pred.symbol for pred in old_rule.right if pred.inputs == old_rule.left.inputs}
        if not others:
            return None
        nt = random.choice(list(others))
        replacement = Pred(nt, old_rule.left.inputs)
        new_rule = PRule(replacement, old_rule.right)
        return self.replace_rule(old_rule, new_rule)

    def connect_two_non_terminals(self):
        options2 = [rule.left for rule in self.prules if rule.left.symbol != self.start]
        if not options2:
            return None
        nt2 = random.choice(options2)
        options1 = [rule.left.symbol for rule in self.prules if rule.left.symbol != nt2.symbol and rule.left.dim == nt2.dim]
        if not options1 or random.getrandbits(1):
            nt1 = self.find_unused_sign()
            nts = self.non_terminals | {nt1}
        else:
            nt1 = random.choice(options1)
            nts = self.non_terminals
        if any(not rule.terminating and len(rule.right) == 1 and
               rule.left.symbol == nt2.symbol and next(iter(rule.right)).symbol == nt1
               for rule in self.prules):
            return None
        v = []
        vs = list(self.variables)
        bogus = PRule(Pred(nt1, []), [])
        while len(v) < nt2.dim:
            v.append(self.find_unused_var(vs, bogus, set(v)))
        return Grammar(self.terminals, nts, vs,
                       self.prules | {PRule(Pred(nt1, v), [Pred(nt2.symbol, v)])},
                       self.start)

    def concatenate_a_vector(self):
        options2 = [rule.left.symbol for rule in self.prules if rule.left.dim == 1]
        if not options2:
            print(self)
        nt2 = random.choice(options2)
        nt3 = random.choice(options2)
        if nt2 == self.start or nt3 == self.start:
            nt1 = self.start
            nts = self.non_terminals
        else:
            options1 = list(set(options2) - {self.start})
            if not options1 or random.getrandbits(1):
                nt1 = self.find_unused_sign()
                nts = self.non_terminals | {nt1}
            else:
                nt1 = random.choice(options1)
                nts = self.non_terminals
        vs = list(self.variables)
        while len(vs) < 2:
            vs.append(self.find_unused_sign(vs))
        new_rule = PRule(Pred(nt1, [vs[0] + vs[1]]),
                        [Pred(nt2, [vs[0]]), Pred(nt3, [vs[1]])])
        return Grammar(self.terminals, nts, vs,
                       self.prules | {new_rule},
                       self.start)

    def new_terminating_rule(self):
        possible_terminals = {t: 0 for t in self.terminals}
        for rule in self.prules:
            if rule.right is None:
                possible_terminals[rule.left.inputs[0]] += 1
        terminals_list = [t for t, num in possible_terminals.items() if num < 2]
        if len(terminals_list) == 0:
            return None
        t = random.choice(terminals_list)
        if random.getrandbits(1):
            n = self.find_unused_sign()
            nts = self.non_terminals | {n}
        else:
            n = random.choice([rule.left.symbol for rule in self.prules if rule.left.dim == 1])
            nts = self.non_terminals
        return self.update_rules(self.prules | {PRule(Pred(n, [t]), None)}, nts)

    def find_unused_sign(self, vs=None):
        if vs is None:
            vs = self.variables
        return next(l for l in string.ascii_uppercase if l not in self.non_terminals and l not in vs)

    def find_unused_var(self, vs, rule, claimed=set()):
        used_variables = {v for pred in rule.right for i in pred.inputs for v in i if v in vs} | claimed
        if len(used_variables) == len(vs):
            v = self.find_unused_sign(vs)
            vs.append(v)
        else:
            v = next(var for var in vs if var not in used_variables)
        return v

    def validate(self):
        dims = {}
        for rule in self.prules:
            for pred in {rule.left} | (set() if rule.terminating else rule.right):
                if pred.symbol in dims and dims[pred.symbol] != pred.dim:
                    return "Dim mismatch for " + pred.symbol
                dims[pred.symbol] = pred.dim
            if rule.terminating:
                if rule.left.dim != 1:
                    return "Terminating rule has dim != 1"
                if len(rule.left.inputs[0]) != 1:
                    return "Terminating rule includes concatenation"
                if rule.left.inputs[0] not in self.terminals:
                    return "Terminating rule has non-terminal token"
            else:
                tokens = "".join(rule.left.inputs)
                if len(tokens) != len(set(tokens)):
                    return "LHS has duplicate tokens"
                if any(len(i) != 1 for pred in rule.right for i in pred.inputs):
                    return "RHS predicate argument is not a variable"
                if "" in rule.left.inputs:
                    return "LHS has empty input"
                vs = "".join("".join(pred.inputs) for pred in rule.right)
                if len(vs) != len(set(vs)):
                    return "RHS has duplicate tokens"
                if any(v not in vs for v in tokens):
                    return "LHS variable not predicated"
                if any(sym not in self.variables for sym in vs):
                    return "Non-terminating rule includes non-variable token"

    def fix_probabilities(self):
        productions = {n: 0 for n in self.non_terminals}
        for rule in self.prules:
            productions[rule.left.symbol] += 1
        return self.update_rules(
            [PRule(rule.left, rule.right, math.log(1 / productions[rule.left.symbol], 2))
             for rule in self.prules])
