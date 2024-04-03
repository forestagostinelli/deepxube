from typing import List, Optional, Set, Tuple, Dict, Callable, Any
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


class Spec:
    def __init__(self, goal_true: Optional[List[Clause]] = None, goal_false: Optional[List[Clause]] = None,
                 atoms_true: Optional[List[Atom]] = None, atoms_false: Optional[List[Atom]] = None,
                 models_banned: Optional[List[Model]] = None, num_atoms_gt: Optional[int] = None):
        """
        :param goal_true: Must be true. Clauses must have goal in the head.
        :param goal_false: Must be false. Clauses must have goal in the head.
        :param atoms_true: will only return stable models that are a superset (including equality) of given atoms
        :param atoms_false: will not return stable models that contain given atoms
        :param models_banned: will not return a stable model that is a superset (including equality) of given models
        :param num_atoms_gt: Number of atoms in model found must be greater than given number
        """

        self.goal_true: List[Clause] = []
        self.goal_false: List[Clause] = []
        self.atoms_true: List[Atom] = []
        self.atoms_false: List[Atom] = []
        self.models_banned: List[Model] = []
        if goal_true is not None:
            self.goal_true = goal_true.copy()
        if goal_false is not None:
            self.goal_false = goal_false.copy()
        if atoms_true is not None:
            self.atoms_true = atoms_true.copy()
        if atoms_false is not None:
            self.atoms_false = atoms_false.copy()
        if models_banned is not None:
            self.models_banned = models_banned.copy()

        if num_atoms_gt is not None:
            # add num_atoms_gt to goal
            lit_count_gt: Literal = Literal("count_model_grnd_atoms_gt", (str(num_atoms_gt),), ("in",))
            if len(self.goal_true) > 0:
                goal_true_new = []
                for clause in self.goal_true:
                    clause_new = Clause(clause.head, clause.body + (lit_count_gt,))
                    goal_true_new.append(clause_new)
                self.goal_true = goal_true_new
            else:
                clause_new = Clause(Literal("goal", tuple(), tuple()), (lit_count_gt,))
                self.goal_true = [clause_new]

    def add(self, spec_add: 'Spec') -> 'Spec':
        return Spec(goal_true=self.goal_true + spec_add.goal_true,
                    goal_false=self.goal_false + spec_add.goal_false,
                    atoms_true=self.atoms_true + spec_add.atoms_true,
                    atoms_false=self.atoms_false + spec_add.atoms_false,
                    models_banned=self.models_banned + spec_add.models_banned)


class Solver:
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

        count_model_grnd_atoms_str: str = "count_model_grnd_atoms(N) :- N = #count{ V: grnd_atom_present(V) }"
        count_model_grnd_atoms_gt_str: str = ("count_model_grnd_atoms_gt(N) :- grnd_atom_count_num(N), "
                                              "count_model_grnd_atoms(M), M > N")
        minimize_grnd_atoms_str = "#minimize {N: count_model_grnd_atoms(N)}"

        # clingo control
        seed = int.from_bytes(os.urandom(4), 'big')
        arguments_rand = ["--models=1", "--opt-mode=ignore", "--heuristic=Domain", "--dom-mod=5,16", "--rand-prob=1",
                          f"--seed={seed}"]
        arguments_min = ["--models=1", "--opt-mode=optN", "--heuristic=Domain", "--dom-mod=5,16"]
        self.ctl_rand: Control = clingo.Control(arguments=arguments_rand)
        self.ctl_min: Control = clingo.Control(arguments=arguments_min)

        for add_line in bk + grnd_atom_counts + [model_grnd_atoms_str, count_model_grnd_atoms_str,
                                                 count_model_grnd_atoms_gt_str]:
            add_line = parse_clingo_line(add_line)
            if len(add_line) == 0:
                continue
            self.ctl_rand.add('base', [], f"{add_line}\n")
            self.ctl_min.add('base', [], f"{add_line}\n")
        add_line = parse_clingo_line(minimize_grnd_atoms_str)
        self.ctl_min.add('base', [], f"{add_line}\n")

        self.ctl_rand.ground([("base", [])])
        self.ctl_min.ground([("base", [])])

    def get_models(self, spec: Spec, on_model: Callable[[Any], Model], num_models: int, minimal: bool) -> List[Model]:
        """

        :param spec: Specification
        :param on_model: Callable that processes models
        :param num_models: number of models to sample
        :param minimal: if true, only samples minimal models
        :return:
        """
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(spec)

        # get models
        models: List[Model] = []
        for model_itr in range(num_models):
            models_i: List[Model] = []
            self.ctl_rand.solve(assumptions=assumptions, on_model=lambda x: models_i.append(on_model(x)))
            if len(models_i) == 0:
                break
            model_i: Model = random.choice(models_i)

            if minimal:
                model_i = self.sample_minimal_model(spec, model_i, on_model)
                # assert model_i == self.sample_minimal_model_old(spec, model_i)

            models.append(model_i)

            if model_itr < (num_models - 1):
                assumptions_i: List[Tuple[Symbol, bool]] = self._make_assumptions(Spec(models_banned=[model_i]))
                assumptions.extend(assumptions_i)

        return models

    def check_model(self, spec: Spec, model: Model) -> bool:
        """

        :param spec: Specification
        :param model: Model to check
        :return:
        """
        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in model]
        spec_check: Spec = Spec(atoms_true=list(model), atoms_false=atoms_false)

        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(spec)
        assumptions += self._make_assumptions(spec_check)
        models_ret: List[None] = []

        self.ctl_rand.solve(assumptions=assumptions, on_model=lambda x: models_ret.append(None))

        return len(models_ret) > 0

    def sample_minimal_model(self, spec: Spec, model: Model, on_model) -> Model:
        models_min: List[Model] = []
        atoms_false: List[Atom] = [atom for atom in self.ground_atoms if atom not in model]
        spec_min: Spec = Spec(atoms_false=atoms_false)
        assumptions: List[Tuple[Symbol, bool]] = self._make_assumptions(spec.add(spec_min))
        self.ctl_min.solve(assumptions=assumptions, on_model=lambda x: models_min.append(on_model(x)))

        return random.choice(models_min)

    def sample_minimal_model_old(self, spec: Spec, model: Model):
        atoms_l = list(model)
        random.shuffle(atoms_l)
        atoms_true: Set[Atom] = set(atoms_l)
        for atom in atoms_l:
            atoms_true.remove(atom)
            model_new: Model = frozenset(atoms_true)
            if self.check_model(spec, model_new):
                return self.sample_minimal_model_old(spec, model_new)

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
                self.ctl_rand.add(prg_blk, [], f"{goal_clause_new_head.to_code()}.\n")
                self.ctl_min.add(prg_blk, [], f"{goal_clause_new_head.to_code()}.\n")
            self.ctl_rand.ground([(prg_blk, [])])
            self.ctl_min.ground([(prg_blk, [])])

            self.goals_added[goal_clauses_set] = goal_new_head_pred

            return goal_new_head_pred
        else:
            return goal_new_head_pred_get

    def _make_assumptions(self, spec: Spec) -> List[Tuple[Symbol, bool]]:
        atoms_true: List[Atom] = []
        atoms_false: List[Atom] = []
        if len(spec.goal_true) > 0:
            atoms_true.append((self._add_goal(spec.goal_true),))
        if len(spec.goal_false) > 0:
            atoms_false.append((self._add_goal(spec.goal_false),))

        atoms_true += spec.atoms_true
        atoms_false += spec.atoms_false
        for model_banned in spec.models_banned:
            blits: List[Literal] = [Literal(atom[0], atom[1:], tuple(["in"] * len(atom[1:]))) for atom in
                                    model_banned]
            clause_banned: Clause = Clause(Literal("goal", tuple(), tuple()), tuple(blits))
            atoms_false.append((self._add_goal([clause_banned]),))

        assumed_true: List[str] = []
        assumed_false: List[str] = []
        for atom in atoms_true:
            assumed_true.append(atom_to_str(atom))
        for atom in atoms_false:
            assumed_false.append(atom_to_str(atom))

        assumptions: List[Tuple[Symbol, bool]] = []
        for lit in assumed_true:
            assumptions.append((parse_term(lit), True))
        for lit in assumed_false:
            assumptions.append((parse_term(lit), False))

        return assumptions
