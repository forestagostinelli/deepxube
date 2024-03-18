from typing import List, Set, Tuple, Dict
from deepxube.logic.logic_objects import Clause, Literal, Atom
from deepxube.utils import misc_utils
import re


def parse_literal(lit_str: str) -> Literal:
    # check for negation
    not_match = re.match(r"\s*not\s+(.*)", lit_str)
    positive: bool
    if not_match is None:
        positive = True
    else:
        positive = False
        lit_str = not_match.group(1).strip()

    # parse literal
    lit_str = misc_utils.remove_all_whitespace(lit_str)
    match = re.match(r"([^(]+)(\((.*)\))?", lit_str)
    assert match is not None
    pred_name: str = match.group(1)
    pred_args: Tuple[str, ...]
    if match.group(3) is not None:
        pred_args = tuple([x.strip() for x in match.group(3).split(",")])
    else:
        pred_args = tuple()

    # TODO, better way to handle directions?
    directions = ["in"] * len(pred_args)

    literal: Literal = Literal(pred_name, pred_args, tuple(directions), positive=positive)
    return literal


def replace_anon_vars(lit: Literal, all_lits: List[Literal]):
    all_args: Set[str] = set()
    for lit2 in [lit] + all_lits:
        all_args.update(lit2.arguments)

    args_new: List[str] = []
    for arg in lit.arguments:
        if arg == "_":
            arg_new: str = "V0"
            arg_new_idx: int = 0
            while arg_new in all_args:
                arg_new_idx += 1
                arg_new = f"V{arg_new_idx}"
            all_args.add(arg_new)

            args_new.append(arg_new)
        else:
            args_new.append(arg)

    lit_ret: Literal = Literal(lit.predicate, tuple(args_new), lit.directions, positive=lit.positive)
    return lit_ret


def parse_clause(constr_str: str) -> Tuple[Clause, Dict[str, List[str]]]:
    head_str, body_str = constr_str.split(":-")
    head_lit = parse_literal(head_str)

    # lits
    body_lit_strs: Tuple[str, ...] = tuple(re.findall(r"[^,]+\([^)]+\)|^\s*[^,]+|[^,)]+\s*$", body_str))
    body_lits: List[Literal] = []
    for body_lit_str in body_lit_strs:
        body_lit = parse_literal(body_lit_str)
        body_lits.append(body_lit)

    # subs forbid
    subs_forbid: Dict[str, List[str]] = dict()
    sub_forbid_strs = re.findall(r"\s*([^,]+)\s*!\s*=\s*([^,]+)\s*", body_str)
    for v1, v2 in sub_forbid_strs:
        v1 = v1.strip()
        v2 = v2.strip()
        if v1 not in subs_forbid:
            subs_forbid[v1] = []
        if v2 not in subs_forbid:
            subs_forbid[v2] = []

        subs_forbid[v1].append(v2)
        subs_forbid[v2].append(v1)

    head_lit = replace_anon_vars(head_lit, [head_lit] + body_lits)
    for idx, body_lit in enumerate(body_lits):
        body_lits[idx] = replace_anon_vars(body_lit, [head_lit] + body_lits)

    clause: Clause = Clause(head_lit, tuple(body_lits))
    return clause, subs_forbid


def copy_clause_with_new_head(clause: Clause, head_name_new: str) -> Clause:
    assert clause.head.positive, "Head should be positive"
    clause_new = Clause(Literal(head_name_new, clause.head.arguments, clause.head.directions,
                                positive=clause.head.positive), clause.body)

    return clause_new


def atom_to_str(atom: Atom) -> str:
    return f"{atom[0]}({','.join(atom[1:])})"
