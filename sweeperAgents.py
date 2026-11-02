# sweeperAgents.py
# ----------------
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
Logic agents for Ghost Sweeper.  See logic.py for how literals, clauses and
knowledge bases are represented, and sweeper.py for the game itself.
"""

import itertools
from typing import Dict, List, Tuple
from util import raiseNotDefined
from game import Agent
from game import Directions
from game import Actions
from logic import VarMap, findModel
from sweeper import getNeighbors
import sweeper

########### ########### ###########
########### QUESTION 1a ###########
########### ########### ###########

def atLeastOne(literals: List[int]) -> List[List[int]]:
    """
    Returns a CNF (a list of clauses) that is true iff at least one of the
    literals is true.

    >>> atLeastOne([1, -2, 3])
    [[1, -2, 3]]
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def atLeastOneIsFalse(literals: List[int]) -> List[List[int]]:
    """
    Returns a CNF that is true iff at least one of the literals is false.

    >>> atLeastOneIsFalse([1, -2, 3])
    [[-1, 2, -3]]
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 1b ###########
########### ########### ###########

def atMostN(literals: List[int], n: int) -> List[List[int]]:
    """
    Returns a CNF that is true iff at most n of the literals are true.  You
    may assume 0 <= n <= len(literals).

    "At most n are true" is equivalent to: for every combination of n+1 of
    the literals, at least one of them is false.  So use atLeastOneIsFalse on
    every combination of n+1 literals, and collect all the clauses.
    itertools.combinations is useful here.

    >>> atMostN([1, 2, 3], 2)
    [[-1, -2, -3]]
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def atLeastN(literals: List[int], n: int) -> List[List[int]]:
    """
    Returns a CNF that is true iff at least n of the literals are true.  You
    may assume 0 <= n <= len(literals).

    "At least n are true" is equivalent to: for every combination of
    len(literals)-n+1 of the literals, at least one of them is true (if more
    than len(literals)-n were false, fewer than n would be true).  So use
    atLeastOne on every combination of len(literals)-n+1 literals, and
    collect all the clauses.

    >>> atLeastN([1, 2, 3], 3)
    [[1], [2], [3]]
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def exactlyN(literals: List[int], n: int) -> List[List[int]]:
    """
    Returns a CNF that is true iff exactly n of the literals are true.  You
    may assume 0 <= n <= len(literals) <= 4.

    Hint: exactly n means at most n and at least n.
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 1c ###########
########### ########### ###########

def perceptClauses(varMap: VarMap, pos: Tuple[int, int], count: int, walls) -> List[List[int]]:
    """
    Returns the clauses Pacman learns from reading `count` on square `pos`:

      - there is no ghost at pos (Pacman is standing there and still alive);
      - exactly `count` of pos's non-wall north/south/east/west neighbors hold
        a ghost.

    Use varMap.ghostVar(square) for the variable G[square], and
    sweeper.getNeighbors(pos, walls) for the non-wall neighbors of pos.
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def entails(kb: List[List[int]], literal: int) -> bool:
    """
    Returns True iff the knowledge base kb (a CNF) entails the literal, i.e.
    iff kb AND (NOT literal) is unsatisfiable.  Use findModel.
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def classify(varMap: VarMap, kb: List[List[int]], squares: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    """
    Returns a dict mapping each square in `squares` to

      'safe'    if kb entails that there is no ghost there,
      'ghost'   if kb entails that there is a ghost there,
      'unknown' otherwise.
    """
    "*** YOUR CODE HERE ***"
    raiseNotDefined()
    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

# Provided helper for Question 3b.

def bfs(start: Tuple[int, int], goals, passable, walls) -> List[str]:
    """
    Returns the list of actions along a shortest path from start to the
    nearest square in `goals`, stepping only on squares in `passable` (start
    itself need not be passable).  Neighbors are expanded in the order
    sweeper.getNeighbors returns them, so ties are broken deterministically.
    Returns None if no goal can be reached.

    You do not need to modify this function.
    """
    goals, passable = set(goals), set(passable)
    parents = {start: None}
    queue = [start]
    for square in queue:
        if square in goals and square != start:
            actions = []
            while parents[square] is not None:
                previous = parents[square]
                actions.append(Actions.vectorToDirection((square[0] - previous[0], square[1] - previous[1])))
                square = previous
            return actions[::-1]
        for neighbor in getNeighbors(square, walls):
            if neighbor not in parents and neighbor in passable:
                parents[neighbor] = square
                queue.append(neighbor)
    return None

class SweeperAgent(Agent):
    """
    A Ghost Sweeper agent that only ever steps onto squares it has proven
    safe.
    """

    def registerInitialState(self, gameState: sweeper.GameState):
        self.walls = gameState.getWalls()
        self.varMap = VarMap(self.walls)
        self.kb = []            # knowledge base: a list of clauses
        self.visited = set()    # squares Pacman has stood on
        self.known = {}         # square -> 'safe' or 'ghost', once proven

    def addPercept(self, gameState: sweeper.GameState):
        """
        If Pacman has not been on his current square before, adds it to
        self.visited, records it in self.known as 'safe', and adds the percept
        clauses for it (using gameState.getAdjacentGhostCount() and your
        perceptClauses) to self.kb.  Otherwise does nothing.

        This method is provided; you do not need to modify it.
        """
        pos = gameState.getPacmanPosition()
        if pos not in self.visited:
            self.visited.add(pos)
            self.known[pos] = 'safe'
            self.kb += perceptClauses(self.varMap, pos, gameState.getAdjacentGhostCount(), self.walls)

    ########### ########### ###########
    ########### QUESTION 3a ###########
    ########### ########### ###########

    def findSafeTargets(self) -> List[Tuple[int, int]]:
        """
        Uses classify to record in self.known every square that is now
        provably 'safe' or provably 'ghost', then returns a sorted list of the
        unvisited squares that are provably safe.

        Only squares next to a visited square can ever be anything but
        'unknown', and a square stays proven once it has been proven, so only
        ask classify about squares that are next to a visited square and not
        yet in self.known.  Asking about every square on every turn will make
        your agent noticeably slow.

        Do not store 'unknown' squares in self.known: a square that is unknown
        now may become provable after later readings, and it must be asked
        about again.
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()
        "*** END YOUR CODE HERE ***"

    ########### ########### ###########
    ########### QUESTION 3b ###########
    ########### ########### ###########

    def getAction(self, gameState: sweeper.GameState):
        """
        Calls addPercept and findSafeTargets, then moves toward the nearest
        target, stepping only on provably safe squares: use the provided bfs
        function and return the first action of the path it finds.  If there
        are no targets, or none can be reached, return Directions.STOP.
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()
        "*** END YOUR CODE HERE ***"
