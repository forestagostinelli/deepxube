from typing import List, Set, Optional, Tuple, Dict, Callable, Any
from deepxube.logic.logic_objects import Clause, Literal, Atom, Model
from deepxube.logic.logic_utils import copy_clause_with_new_head, atom_to_str

import random
import os
import clingo
from clingo import Control, parse_term, Symbol
import re


def model_to_body(model: Model) -> str:
    return ','.join([atom_to_str(atom) for atom in model])


def on_model_var_vals(m) -> frozenset[str]:
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

        count_model_grnd_atoms_str: str = "count_model_grnd_atoms(N) :- N = #count{{ V: grnd_atom_present(V) }}"
        count_model_grnd_atoms_gt_str: str = ("count_model_grnd_atoms_gt(N) :- grnd_atom_count_num(N), "
                                              "count_model_grnd_atoms(M), M > N")

        # clingo control
        seed = int.from_bytes(os.urandom(4), 'big')
        arguments = ["--models=1", "--opt-mode=ignore", "--heuristic=Domain", "--dom-mod=5,16", "--rand-prob=1",
                     f"--seed={seed}"]
        self.ctl: Control = clingo.Control(arguments=arguments)

        for add_line in bk + grnd_atom_counts + [model_grnd_atoms_str, count_model_grnd_atoms_str,
                                                 count_model_grnd_atoms_gt_str]:
            add_line = parse_clingo_line(add_line)
            if len(add_line) == 0:
                continue
            self.ctl.add('base', [], f"{add_line}\n")
        self.ctl.ground([("base", [])])

    def get_models(self, goal: List[Clause], on_model: Callable[[Any], Model], minimal: bool = True,
                   num_models: int = 1, assumed_true: Optional[Model] = None,
                   assumed_false: Optional[List[Model]] = None, num_atoms_gt: Optional[int] = None) -> List[Model]:
        """

        :param goal: Must have goal in the head.
        :param on_model: Callable that processes models
        :param minimal: if true, only samples minimal models
        :param num_models: number of models to sample
        :param assumed_true: will only return stable models that are a superset (including equality) of given model
        :param assumed_false: will not return a stable model that is a superset (including equality) of given models
        :param num_atoms_gt: Number of atoms in model found must be greater than given number
        :return:
        """

        # add constraint if there is min number of atoms
        if num_atoms_gt is not None:
            goal_new = []
            lit_count_gt: Literal = Literal("count_model_grnd_atoms_gt", (str(num_atoms_gt),), ("in",))
            for clause in goal:
                clause_new = Clause(clause.head, clause.body + (lit_count_gt,))
                goal_new.append(clause_new)
            goal = goal_new

        # get assumptions
        if assumed_true is None:
            assumed_true = frozenset()
        if assumed_false is None:
            assumed_false = []
        atoms_true: List[Atom] = [(self._add_goal(goal),)]

        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true + list(assumed_true), [],
                                                                        assumed_false)

        # get models
        models: List[Model] = []
        for _ in range(num_models):
            models_i: List[Model] = []
            self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_i.append(on_model(x)))
            if len(models_i) == 0:
                break
            model_i: Model = random.choice(models_i)

            if minimal:
                model_i = self.sample_minimal_model(goal, model_i, assumed_true=assumed_true,
                                                    assumed_false=assumed_false)

            models.append(model_i)
            assumptions_i: List[Tuple[Symbol, bool]] = self._make_assumptions([], [], [model_i])
            assumptions.extend(assumptions_i)

        return models

    def check_model(self, goal: List[Clause], model: Model, assumed_true: Optional[Model] = None,
                    assumed_false: Optional[List[Model]] = None) -> bool:
        """

        :param goal: Logical or over clauses. Must have goal in the head.
        :param model: Model to check
        :param assumed_true: will only return stable models that are a superset (including equality) of given model
        :param assumed_false: will not return a stable model that is a superset (including equality) of given models
        :return:
        """
        if assumed_true is None:
            assumed_true = frozenset()
        if assumed_false is None:
            assumed_false = []

        atoms_true: List[Atom] = [(self._add_goal(goal),)]

        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in model]
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(atoms_true + list(assumed_true) + list(model),
                                                                        atoms_false, assumed_false)
        models_ret: List[None] = []
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
        models_ret: List[None] = []
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
        models_ret: List[None] = []
        self.ctl.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(None))

        return len(models_ret) == 0

    def sample_minimal_model(self, goal: List[Clause], model: Model, assumed_true: Optional[Model] = None,
                             assumed_false: Optional[List[Model]] = None) -> Model:
        atoms_l = list(model)
        random.shuffle(atoms_l)
        atoms_true: Set[Atom] = set(atoms_l)
        for atom in atoms_l:
            atoms_true.remove(atom)
            model_new: Model = frozenset(atoms_true)
            if self.check_model(goal, model_new, assumed_true=assumed_true, assumed_false=assumed_false):
                return self.sample_minimal_model(goal, model_new, assumed_true=assumed_true,
                                                 assumed_false=assumed_false)

            atoms_true.add(atom)

        return frozenset(atoms_true)

    def _add_goal(self, goal: List[Clause]) -> str:
        """ Adds all goal clauses with same head

        :param goal:
        :return:
        """
        assert all([x.head.predicate == "goal" for x in goal]), "head should be goal"
        assert all([len(x.head.arguments) == 0 for x in goal]), "head should not have any arguments"

        goal_clauses_set: frozenset[Clause] = frozenset(goal)
        goal_new_head_pred_get: Optional[str] = self.goals_added.get(goal_clauses_set)
        if goal_new_head_pred_get is None:
            goal_num: int = len(self.goals_added)
            goal_new_head_pred: str = f"goal{goal_num}"

            prg_blk: str = goal_new_head_pred
            for goal_clause in goal:
                goal_clause_new_head: Clause = copy_clause_with_new_head(goal_clause, goal_new_head_pred)
                self.ctl.add(prg_blk, [], f"{goal_clause_new_head.to_code()}.\n")
            self.ctl.ground([(prg_blk, [])])

            self.goals_added[goal_clauses_set] = goal_new_head_pred

            return goal_new_head_pred
        else:
            return goal_new_head_pred_get

    def _make_assumptions(self, atoms_true: List[Atom], atoms_false: List[Atom],
                          models_assumed_false: List[Model]) -> List[Tuple[Symbol, bool]]:
        assumed_true: List[str] = []
        assumed_false: List[str] = []

        for atom in atoms_true:
            assumed_true.append(atom_to_str(atom))
        for atom in atoms_false:
            assumed_false.append(atom_to_str(atom))

        for model_banned in models_assumed_false:
            blits: List[Literal] = [Literal(atom[0], atom[1:], tuple(["in"] * len(atom[1:]))) for atom in model_banned]
            clause_banned: Clause = Clause(Literal("goal", tuple(), tuple()), tuple(blits))
            assumed_false.append(self._add_goal([clause_banned]))

        assumptions: List[Tuple[Symbol, bool]] = []
        for lit in assumed_true:
            assumptions.append((parse_term(lit), True))
        for lit in assumed_false:
            assumptions.append((parse_term(lit), False))

        return assumptions
