from typing import List, Set, Optional, Tuple, Dict, Callable
from deepxube.logic.program import Clause, Literal, Atom, Model, atom_to_str

import random
import os
import clingo
from clingo import Control, parse_term, Symbol
import re

from deepxube.utils.program_utils import copy_clause_with_new_head


def model_to_body(model: Model) -> str:
    return ','.join([atom_to_str(atom) for atom in model])


def on_model_var_vals(m) -> frozenset:
    return frozenset(str(x) for x in m.symbols(shown=True))


def parse_clingo_line(line: str) -> str:
    if re.search(r"\S", line) is None:
        return ""

    line = line.strip()
    if re.search(r"\.$", line) is None:
        line = f"{line}."

    return line


class ASPSpec:
    def __init__(self, ground_atoms: List[Atom], bk: List[str]):
        self.bk: List[str] = bk
        self.goals_added: Dict[frozenset[Clause], str] = dict()
        self.models_banned: List[Model] = []

        self.ground_atoms: List[Atom] = ground_atoms
        grnd_atoms_str_l: List[str] = [atom_to_str(atom) for atom in self.ground_atoms]
        model_grnd_atoms_str: str = f"0 {{ {'; '.join(grnd_atoms_str_l)} }} {len(self.ground_atoms)}"
        grnd_atom_counts: List[str] = []
        for grnd_atom_idx, grnd_atom_str in enumerate(grnd_atoms_str_l):
            grnd_atom_counts.append(f"grnd_atom_count_num({grnd_atom_idx})")
            grnd_atom_counts.append(f"grnd_atom_present({grnd_atom_idx}) :- {grnd_atom_str}")

        count_model_grnd_atoms_str: str = f"count_model_grnd_atoms(N) :- N = #count{{ V: grnd_atom_present(V) }}"
        count_model_grnd_atoms_gt_str: str = (f"count_model_grnd_atoms_gt(N) :- grnd_atom_count_num(N), "
                                              f"count_model_grnd_atoms(M), M > N")

        # clingo control
        seed = int.from_bytes(os.urandom(4), 'big')
        arguments = [f"--models=1", f"--opt-mode=ignore", "--heuristic=Domain", "--dom-mod=5,16", "--rand-prob=1",
                     f"--seed={seed}"]
        self.ctl: Control = clingo.Control(arguments=arguments)

        for add_line in bk + grnd_atom_counts + [model_grnd_atoms_str, count_model_grnd_atoms_str,
                                                 count_model_grnd_atoms_gt_str]:
            add_line = parse_clingo_line(add_line)
            if len(add_line) == 0:
                continue
            self.ctl.add('base', [], f"{add_line}\n")
        self.ctl.ground([("base", [])])

    def get_models(self, goal: List[Clause], on_model: Callable, minimal: bool = True,
                   num_models: int = 1, models_banned: Optional[List[Model]] = None) -> List[Model]:
        """

        :param goal: Must have goal in the head.
        :param on_model: Callable that processes models
        :param minimal:
        :param num_models:
        :param models_banned:
        :return:
        """
        if models_banned is None:
            models_banned = []
        atoms_true: List[Atom] = [(self._add_goal(goal),)]

        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true, [], models_banned)
        models: List[Model] = []
        for _ in range(num_models):
            models_i: List[Model] = []
            self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_i.append(on_model(x)))
            if len(models_i) == 0:
                break
            model_i: Model = random.choice(models_i)

            if minimal:
                model_i = self.sample_minimal_model(goal, model_i)

            models.append(model_i)
            assumptions_i: List[Tuple[Symbol, bool]] = self._make_assumptions([], [], [model_i])
            assumptions.extend(assumptions_i)

        return models

    def check_model(self, goal: List[Clause], model: Model) -> bool:
        """

        :param goal: Logical or over clauses. Must have goal in the head.
        :param model:
        :return:
        """
        atoms_true: List[Atom] = [(self._add_goal(goal),)]

        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in model]
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true + list(model), atoms_false, [])
        models_ret: List = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(None))

        return len(models_ret) > 0

    def intersects(self, goal1: List[Clause], goal2: List[Clause]) -> bool:
        """

        :param goal1
        :param goal2
        :return:
        """
        atoms_true: List[Atom] = [(self._add_goal(goal1),), (self._add_goal(goal2),)]
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true, [], [])
        models_ret: List = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(None))

        return len(models_ret) > 0

    def is_equal(self, goal1: List[Clause], goal2: List[Clause]) -> bool:
        """
        :param goal1
        :param goal2
        :return: True if both are supersets of one another.
        Otherwise, returns False.
        """

        if not self.is_superset(goal1, goal2):
            return False

        if not self.is_superset(goal2, goal1):
            return False

        return True

    def is_superset(self, goal1: List[Clause], goal2: List[Clause]) -> bool:
        """
        :param goal1
        :param goal2
        :return: True if the set of stable models of goal1 is a superset of the stable models of goal2.
        Otherwise, returns False.
        """

        atom1: Atom = (self._add_goal(goal1),)
        atom2: Atom = (self._add_goal(goal2),)
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions([atom2], [atom1], [])
        models_ret: List = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(None))

        return len(models_ret) == 0

    def check_get_vars(self, goal: Clause, model: Model) -> List[Model]:
        atoms_true: List[Atom] = [(self._add_goal([goal]),)]
        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in model]
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true + list(model), atoms_false, [])

        models_ret: List[Model] = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(on_model_var_vals(x)))

        return models_ret

    def sample_minimal_model(self, goal: List[Clause], model: Model) -> Model:
        atoms_l = list(model)
        random.shuffle(atoms_l)
        atoms_true: Set[Atom] = set(atoms_l)
        for atom in atoms_l:
            atoms_true.remove(atom)
            model_new: Model = frozenset(atoms_true)
            if self.check_model(goal, model_new):
                return self.sample_minimal_model(goal, model_new)

            atoms_true.add(atom)

        return frozenset(atoms_true)

    def samp_next_state(self, state_m: Model, on_model: Callable) -> List[Model]:
        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in state_m]
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(list(state_m), atoms_false, [])
        models_ret: List[Model] = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(on_model(x)))

        return models_ret

    def _add_goal(self, goal: List[Clause]) -> str:
        """ Adds all goal clauses with same head

        :param goal:
        :return:
        """
        assert all([x.head.predicate == "goal" for x in goal]), "head should be goal"
        assert all([len(x.head.arguments) == 0 for x in goal]), "head should not have any arguments"

        goal_clauses_set: frozenset[Clause] = frozenset(goal)
        goal_new_head_pred: Optional[str] = self.goals_added.get(goal_clauses_set)
        if goal_new_head_pred is None:
            goal_num: int = len(self.goals_added)
            goal_new_head_pred: str = f"goal{goal_num}"

            prg_blk: str = goal_new_head_pred
            for goal_clause in goal:
                goal_clause_new_head: Clause = copy_clause_with_new_head(goal_clause, goal_new_head_pred)
                self.ctl.add(prg_blk, [], f"{goal_clause_new_head.to_code()}.\n")
            self.ctl.ground([(prg_blk, [])])

            self.goals_added[goal_clauses_set] = goal_new_head_pred

        return goal_new_head_pred

    def _make_assumptions(self, atoms_true: List[Atom], atoms_false: List[Atom],
                          models_banned: List[Model]) -> List[Tuple[Symbol, bool]]:
        assumed_true: List[str] = []
        assumed_false: List[str] = []

        for atom in atoms_true:
            assumed_true.append(atom_to_str(atom))
        for atom in atoms_false:
            assumed_false.append(atom_to_str(atom))

        for model_banned in models_banned:
            blits: List[Literal] = [Literal(atom[0], atom[1:], tuple(["in"] * len(atom[1:]))) for atom in model_banned]
            clause_banned: Clause = Clause(Literal("goal", tuple(), tuple()), tuple(blits))
            assumed_false.append(self._add_goal([clause_banned]))

        assumptions: List[Tuple[Symbol, bool]] = []
        for lit in assumed_true:
            assumptions.append((parse_term(lit), True))
        for lit in assumed_false:
            assumptions.append((parse_term(lit), False))

        return assumptions
