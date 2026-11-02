# logic.py
# --------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Propositional logic support for Ghost Sweeper.  You do not need to modify
this file.

Representation (DIMACS style, the same one real SAT solvers use):

  - A *variable* is a positive int.  In Ghost Sweeper, variable v stands for
    the proposition G[x,y]: "a ghost is at square (x,y)".  Use a VarMap to
    translate between squares and variables.
  - A *literal* is a nonzero int: v means "v is true", -v means "v is false".
  - A *clause* is a list of literals, read as their disjunction (OR).
  - A *CNF* (knowledge base) is a list of clauses, read as their conjunction
    (AND).

For example, with g1 = varMap.ghostVar((1, 2)) and g2 = varMap.ghostVar((2, 1)),
the CNF [[g1, g2], [-g1, -g2]] says "exactly one of (1,2) and (2,1) holds a
ghost".

The one function you will call is findModel(clauses), which returns a
satisfying assignment or False if the clauses are unsatisfiable.
"""

try:
    import pycosat
except ImportError:
    pycosat = None


class VarMap:
    """
    Maps the proposition G[x,y] ("a ghost is at (x,y)") to a positive int
    variable and back.

    >>> varMap = VarMap()
    >>> v = varMap.ghostVar((3, 4))
    >>> varMap.position(v)
    (3, 4)
    >>> varMap.position(-v)
    (3, 4)
    >>> varMap.literalToString(-v)
    '~G[3,4]'
    """

    def __init__(self, walls=None):
        """
        If walls (a Grid) is given, every non-wall square gets a variable up
        front, numbered in (x, y) order.  Otherwise variables are created on
        demand by ghostVar.
        """
        self._posToVar = {}
        self._varToPos = {}
        if walls is not None:
            for pos in sorted(walls.asList(False)):
                self.ghostVar(pos)

    def ghostVar(self, pos):
        """
        Returns the variable for G[pos], creating it if necessary.
        """
        pos = tuple(pos)
        if pos not in self._posToVar:
            var = len(self._posToVar) + 1
            self._posToVar[pos] = var
            self._varToPos[var] = pos
        return self._posToVar[pos]

    def position(self, literal):
        """
        Returns the square (x, y) that a variable or literal refers to.
        """
        return self._varToPos[abs(literal)]

    def numVars(self):
        return len(self._posToVar)

    def allPositions(self):
        return sorted(self._posToVar.keys())

    def literalToString(self, literal):
        x, y = self.position(literal)
        return '%sG[%d,%d]' % ('~' if literal < 0 else '', x, y)

    def clauseToString(self, clause):
        return '(' + ' | '.join(self.literalToString(l) for l in clause) + ')'

    def cnfToString(self, clauses):
        """
        Pretty-prints a CNF; handy for debugging.
        """
        return ' &\n'.join(self.clauseToString(c) for c in clauses)


def findModel(clauses):
    """
    Returns a satisfying assignment for the CNF `clauses` as a dict
    {var: bool} covering every variable that appears in `clauses`, or False if
    no satisfying assignment exists.

    >>> findModel([[1, 2], [-1]])
    {1: False, 2: True}
    >>> findModel([[1], [-1]])
    False
    >>> findModel([])
    {}
    """
    clauses = [list(c) for c in clauses]
    if any(len(c) == 0 for c in clauses):
        return False
    variables = set(abs(l) for c in clauses for l in c)
    if pycosat is not None:
        solution = pycosat.solve(clauses)
        if solution == 'UNSAT':
            return False
        assignment = dict((abs(l), l > 0) for l in solution)
    else:
        assignment = _dpll(clauses)
        if assignment is False:
            return False
    return dict((v, assignment.get(v, False)) for v in sorted(variables))


def satisfies(assignment, clauses):
    """
    Returns True iff the assignment {var: bool} makes every clause true.
    Variables missing from the assignment count as False.
    """
    for clause in clauses:
        if not any(assignment.get(abs(l), False) == (l > 0) for l in clause):
            return False
    return True


########################################
# Staff DPLL solver (used if pycosat is #
# not installed).  No need to read on.  #
########################################

def _dpll(clauses):
    clauses = [frozenset(c) for c in set(frozenset(c) for c in clauses)]
    clauses = [c for c in clauses if not any(-l in c for l in c)]  # drop tautologies
    return _dpllSearch(clauses, {})


def _assign(clauses, literal):
    """
    Simplifies clauses under literal = True.  Returns None on a conflict.
    """
    result = []
    for c in clauses:
        if literal in c:
            continue
        if -literal in c:
            c = c - {-literal}
            if not c:
                return None
        result.append(c)
    return result


def _dpllSearch(clauses, assignment):
    # Unit propagation
    while True:
        unit = None
        for c in clauses:
            if len(c) == 1:
                unit = next(iter(c))
                break
        if unit is None:
            break
        assignment[abs(unit)] = unit > 0
        clauses = _assign(clauses, unit)
        if clauses is None:
            return False
    if not clauses:
        return assignment

    # Pure literals can be set without branching
    literals = set(l for c in clauses for l in c)
    pure = [l for l in literals if -l not in literals]
    if pure:
        for l in pure:
            assignment[abs(l)] = l > 0
        pureSet = set(pure)
        clauses = [c for c in clauses if not (c & pureSet)]
        return _dpllSearch(clauses, assignment)

    # Branch on the most frequent literal among the shortest clauses
    shortest = min(len(c) for c in clauses)
    counts = {}
    for c in clauses:
        if len(c) == shortest:
            for l in c:
                counts[l] = counts.get(l, 0) + 1
    literal = max(sorted(counts), key=lambda l: counts[l])
    for choice in (literal, -literal):
        simplified = _assign(clauses, choice)
        if simplified is None:
            continue
        newAssignment = dict(assignment)
        newAssignment[abs(choice)] = choice > 0
        result = _dpllSearch(simplified, newAssignment)
        if result is not False:
            return result
    return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
