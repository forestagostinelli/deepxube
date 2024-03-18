from typing import List, Tuple, Optional, Set, Dict, FrozenSet


class Literal:
    def __init__(self, predicate: str, arguments: Tuple[str, ...], directions: Tuple[str, ...], positive: bool = True):
        self.predicate: str = predicate
        self.arguments: Tuple[str, ...] = arguments
        self.directions: Tuple[str, ...] = directions
        for direction in self.directions:
            assert direction in ["in", "out"], f"Direction must be 'in' or 'out' but is {direction}"

        self.arity: int = len(self.arguments)
        self.positive: bool = positive

        self.inputs: Set[str] = set(arg for direction, arg in zip(self.directions, self.arguments) if direction == 'in')
        self.outputs: Set[str] = set(arg for direction, arg in zip(self.directions, self.arguments)
                                     if direction == 'out')

        self.in_out: Set[Tuple[str, str]] = set()
        for arg, direction in zip(self.arguments, self.directions):
            self.in_out.add((arg, direction))

    def to_code(self) -> str:
        prefix: str = ""
        if not self.positive:
            prefix = "not "

        if len(self.arguments) > 0:
            return f'{prefix}{self.predicate}({",".join(self.arguments)})'
        else:
            return f'{prefix}{self.predicate}'

    def get_pred_arity_pos_id(self) -> Tuple[str, int, bool]:
        tup: Tuple[str, int, bool] = (self.predicate, len(self.arguments), self.positive)
        return tup

    def __str__(self):
        return self.to_code()

    def __repr__(self):
        return self.__str__()


class VarNode:
    def __init__(self):
        self.rep: int = 0
        self.neighbors: List[VarNode] = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)


class LitNode:
    def __init__(self, predicate: str, in_body: bool, arguments: Tuple[str, ...]):
        self.rep: int = 0
        self.predicate: str = predicate
        self.in_body: bool = in_body
        self.var_nodes: List[VarNode] = []
        self.var_names: Tuple[str, ...] = arguments
        for _ in range(len(arguments)):
            self.var_nodes.append(VarNode())

    def prop_up(self):
        self.rep = hash((self.predicate, self.in_body) + tuple(x.rep for x in self.var_nodes))

    def prop_down(self):
        for var_idx, var_node in enumerate(self.var_nodes):
            var_node.rep = hash((self.rep, var_idx))


def prop_across(var_nodes: List[VarNode]):
    reps_new: List[int] = []
    for var_node in var_nodes:
        rep_new: int = sum([x.rep for x in var_node.neighbors])
        reps_new.append(rep_new)

    for var_node, rep_new in zip(var_nodes, reps_new):
        var_node.rep = rep_new


class Clause:
    def __init__(self, head: Literal, body: Tuple[Literal, ...]):
        self.head: Literal = head
        self.body: Tuple[Literal, ...] = body

        # compute all the 'vars' in the program
        self.all_vars: Set[str] = set()
        self.all_vars.update(head.arguments)
        for literal in (self.head,) + self.body:
            self.all_vars.update(literal.arguments)

        self.hash: Optional[int] = None

    def is_in_out_consistent_body(self) -> bool:
        var_has_out: Set[str] = set()

        for body_lit in self.body:
            var_has_out.update(body_lit.outputs)

        for body_lit in self.body:
            for in_var in body_lit.inputs:
                if in_var not in var_has_out:
                    return False

        return True

    def can_ground(self) -> bool:
        grounded_variables = self.head.inputs
        body_literals = set(self.body)

        while len(body_literals) > 0:
            selected_literal = None
            for literal in body_literals:
                if literal.inputs.issubset(grounded_variables):
                    selected_literal = literal
                    break

            if selected_literal is None:
                return False

            grounded_variables = grounded_variables.union(selected_literal.outputs)
            body_literals = body_literals.difference({selected_literal})

        return True

    def to_code(self) -> str:
        if len(self.body) > 0:
            return (
                f'{self.head.to_code()}:- '
                f'{",".join([blit.to_code() for blit in self.body])}'
            )
        else:
            return self.head.to_code()

    def get_lit_id_count_dict(self) -> Dict[Tuple[str, int, bool], int]:
        lit_pred_dict: Dict[Tuple[str, int, bool], int] = dict()
        for lit in [self.head] + list(self.body):
            if lit is not None:
                lit_tup: Tuple[str, int, bool] = lit.get_pred_arity_pos_id()
                if lit_tup not in lit_pred_dict:
                    lit_pred_dict[lit_tup] = 0

                lit_pred_dict[lit_tup] += 1

        return lit_pred_dict

    def theta_sub(self, other, subs_prev: Optional[Dict[str, str]] = None, negate_l: Optional[List[bool]] = None,
                  subs_forbid: Optional[Dict[str, List[str]]] = None,
                  ignore_head: bool = False) -> Optional[Dict[str, str]]:
        # Initialize
        if ignore_head:
            assert negate_l is None, "Negate not yet integrated with ignore_head"

        if subs_prev is None:
            subs_prev = dict()
        if subs_forbid is None:
            subs_forbid = dict()

        lits_self: List[Literal] = list(self.body)
        lits_other: List[Literal] = list(other.body)
        if not ignore_head:
            lits_self = [self.head] + lits_self
            lits_other = [other.head] + lits_other

        name_to_lit_other: Dict[str, List[Literal]] = dict()
        for lit in lits_other:
            if lit.predicate not in name_to_lit_other.keys():
                name_to_lit_other[lit.predicate] = []
            name_to_lit_other[lit.predicate].append(lit)

        negate_prev = negate_l
        if negate_l is None:
            negate_l = [False] * len(lits_self)

        # check all predicate names in self appears in other
        assert len(lits_self) == len(negate_l), f"{self}, {other}, {len(lits_self)}, {len(negate_l)}, {negate_prev}"
        for lit_self, negate in zip(lits_self, negate_l):
            if (not negate) and (lit_self.predicate not in name_to_lit_other.keys()):
                return None

        return theta_sub_lits(lits_self, name_to_lit_other, negate_l, subs_prev, subs_forbid)

    def __str__(self):
        if len(self.body) > 0:
            # return (
            #    f'{self.head.to_code()}:- '
            #    f'{",".join([blit.to_code() for blit in self.body if blit.predicate[:4] != "dif_"])}'
            # )
            return (
                f'{self.head.to_code()}:- '
                f'{",".join([blit.to_code() for blit in self.body])}'
            )
        else:
            return self.head.to_code()

        # return self.to_code()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        if self.hash is not None:
            return self.hash

        # make lit nodes
        lit_nodes: List[LitNode] = []
        for lit in [self.head]:
            lit_nodes.append(LitNode(lit.predicate, False, lit.arguments))

        for lit in self.body:
            lit_nodes.append(LitNode(lit.predicate, True, lit.arguments))

        # make connections between variable nodes
        for lit1_idx, lit_node1 in enumerate(lit_nodes):
            for lit2_idx, lit_node2 in enumerate(lit_nodes):
                if lit1_idx == lit2_idx:
                    continue
                for var_idx1, var_name1 in enumerate(lit_node1.var_names):
                    for var_idx2, var_name2 in enumerate(lit_node2.var_names):
                        if var_name1 == var_name2:
                            lit_node1.var_nodes[var_idx1].add_neighbor(lit_node2.var_nodes[var_idx2])

        # get all variable nodes
        var_nodes: List[VarNode] = []
        for lit_node in lit_nodes:
            var_nodes.extend(lit_node.var_nodes)

        # init representation
        for lit_node in lit_nodes:
            lit_node.prop_up()

        # propagate
        self.hash = 0
        for _ in range(10):
            for lit_node in lit_nodes:
                lit_node.prop_down()

            prop_across(var_nodes)

            for lit_node in lit_nodes:
                lit_node.prop_up()

        self.hash = sum([lit_node.rep for lit_node in lit_nodes])

        return self.hash

    def __eq__(self, other):
        # assuming no repeats
        if (self.head is None) != (other.head is None):
            return False
        if self.head.predicate != other.head.predicate:
            return False

        if self.theta_sub(other) is None:
            return False

        if other.theta_sub(self) is None:
            return False

        return True


def prune_lit(lit: Literal, lit_to_prune: Literal, idxs_vars_req: List[Tuple[int, str]]) -> bool:
    if lit.positive != lit_to_prune.positive:
        return True
    else:
        for idx, var_req in idxs_vars_req:
            if lit_to_prune.arguments[idx] != var_req:
                return True
    return False


def theta_sub_lits(lits1: List[Literal], lits2_dict: Dict[str, List[Literal]], negate_l: List[bool],
                   subs_prev: Dict[str, str], subs_forbid: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
    # TODO handle negative literals
    if len(lits1) == 0:
        return subs_prev

    # TODO, negation as failure not really theta subsumption
    lit1: Literal = lits1[0]
    negate: bool = negate_l[0]

    lits2: List[Literal] = lits2_dict.get(lit1.predicate, [])
    if len(lits2) == 0:
        if negate:
            return theta_sub_lits(lits1[1:], lits2_dict, negate_l[1:], subs_prev, subs_forbid)
        else:
            return None

    idxs_not_subbed: List[int] = []
    idxs_vars_req: List[Tuple[int, str]] = []
    for i, x in enumerate(lit1.arguments):
        var_sub: Optional[str] = subs_prev.get(x)
        if var_sub is not None:
            idxs_vars_req.append((i, var_sub))
        else:
            idxs_not_subbed.append(i)

    lits2 = [x for x in lits2 if not prune_lit(lit1, x, idxs_vars_req)]
    if len(lits2) == 0:
        if negate:
            return theta_sub_lits(lits1[1:], lits2_dict, negate_l[1:], subs_prev, subs_forbid)
        else:
            return None

    subs_rec: Optional[Dict[str, str]]
    if len(idxs_not_subbed) == 0:
        # succeeded somewhere and no substitutions needed
        if negate:
            return None
        else:
            subs_rec = theta_sub_lits(lits1[1:], lits2_dict, negate_l[1:], subs_prev, subs_forbid)
            if subs_rec is not None:
                return subs_rec
    else:
        for lit2 in lits2:
            subs: Optional[Dict[str, str]] = theta_sub_args(lit1.arguments, lit2.arguments, idxs_not_subbed,
                                                            subs_prev, subs_forbid)
            if subs is not None:
                if negate:
                    return None
                else:
                    subs_rec = theta_sub_lits(lits1[1:], lits2_dict, negate_l[1:], subs, subs_forbid)
                    if subs_rec is not None:
                        return subs_rec

    if negate:
        return theta_sub_lits(lits1[1:], lits2_dict, negate_l[1:], subs_prev, subs_forbid)
    else:
        return None


def theta_sub_args(args1: Tuple[str, ...], args2: Tuple[str, ...], idxs_not_subbed: List[int],
                   subs_prev: Dict[str, str], subs_forbid: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
    # assuming previous subs checked already if not negate
    subs: Dict[str, str] = subs_prev.copy()

    for idx in idxs_not_subbed:
        arg1: str = args1[idx]
        arg2: str = args2[idx]

        arg1_sub: Optional[str] = subs.get(arg1)
        if (arg1_sub is not None) and (arg1_sub != arg2):
            return None
        else:
            # check if neq constraint not violated
            args_other: Optional[List[str]] = subs_forbid.get(arg1, [])
            if args_other is not None:
                for arg_other in args_other:
                    arg_other_sub: Optional[str] = subs.get(arg_other)
                    if (arg_other_sub is not None) and (arg_other_sub == arg2):
                        return None

            is_var: bool = arg1[0].isupper() and arg1[0].isalpha()
            if is_var:
                # is variable, make substitution
                subs[arg1] = arg2
            else:
                # not variable, cannot substitute
                if arg1 != arg2:
                    return None

    return subs


def make_subs_lit(lit: Literal, subs: Dict[str, str]) -> Literal:
    args_sub: Tuple[str, ...] = tuple(subs[x] for x in lit.arguments)
    lit_sub: Literal = Literal(lit.predicate, args_sub, lit.directions, positive=lit.positive)
    return lit_sub


def make_subs(clause: Clause, subs: Dict[str, str]) -> Clause:
    head_sub: Literal = make_subs_lit(clause.head, subs)
    body_sub: Tuple[Literal, ...] = tuple(make_subs_lit(x, subs) for x in clause.body)
    clause_sub: Clause = Clause(head_sub, body_sub)
    return clause_sub


def theta_sub_replace(clause1: Clause, clause2: Clause, ignore_head: bool = False) -> Clause:
    subs: Optional[Dict[str, str]] = clause1.theta_sub(clause2, ignore_head=ignore_head)
    if subs is None:
        return Clause(clause2.head, clause2.body)

    clause1 = make_subs(clause1, subs)
    body_new: List[Literal] = [clause1.head]

    # TODO changes in to_code() may break this, what about direction?
    clause1_body_code_set: Set[str] = set(lit.to_code() for lit in clause1.body)
    body_new = body_new + [lit for lit in clause2.body if lit.to_code() not in clause1_body_code_set]

    clause2_new: Clause = Clause(clause2.head, tuple(body_new))
    return clause2_new


# first index is the predicate name and the remaining are the arguments
Atom = Tuple[str, ...]
Model = FrozenSet[Atom]
