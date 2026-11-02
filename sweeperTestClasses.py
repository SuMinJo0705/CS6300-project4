# sweeperTestClasses.py
# ---------------------
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


# Test classes for Ghost Sweeper (Q1-Q3).  The tracking questions' test
# classes are re-exported below, so this module serves the whole project.

import itertools
import testClasses
import layout
import logic
import sweeper
import textDisplay
from bayesHMMTestClasses import *

def parseSquares(text):
    """
    Parses "x,y x,y ..." into a list of (x, y) tuples.
    """
    return [tuple(int(c) for c in s.split(',')) for s in text.split()]

def squaresToString(squares):
    return ' '.join('%d,%d' % s for s in squares)

def checkCNF(clauses, allowedVars):
    """
    Returns an error message if clauses is not a list of lists of nonzero
    ints over allowedVars, otherwise None.
    """
    if not isinstance(clauses, list):
        return 'expected a list of clauses, got %r' % (clauses,)
    for clause in clauses:
        if not isinstance(clause, list):
            return 'expected each clause to be a list of literals, got %r' % (clause,)
        for literal in clause:
            if not isinstance(literal, int) or isinstance(literal, bool) or literal == 0:
                return 'literals must be nonzero ints, got %r in clause %r' % (literal, clause)
            if abs(literal) not in allowedVars:
                return 'clause %r mentions variable %d, which is not one of %s' % (clause, abs(literal), sorted(allowedVars))
    return None

def models(clauses, variables):
    """
    Returns the set of models of clauses over variables (by brute force), each
    model a frozenset of the true variables.
    """
    variables = sorted(variables)
    result = set()
    for bits in itertools.product([False, True], repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        if logic.satisfies(assignment, clauses):
            result.add(frozenset(v for v in variables if assignment[v]))
    return result

def layoutFromTest(testDict):
    return layout.Layout([line.strip() for line in testDict['layout_str'].strip().split('\n')])

def exactlyNByEnumeration(literals, n):
    """
    An exactly-n encoding that forbids each bad assignment with one clause.
    Exponential, but fine for at most four literals.
    """
    clauses = []
    for bits in itertools.product([False, True], repeat=len(literals)):
        if bits.count(True) != n:
            clauses.append([-l if b else l for l, b in zip(literals, bits)])
    return clauses


class CardinalityTest(testClasses.TestCase):
    """
    Each case is "function literal literal ..." or, for atMostN, atLeastN
    and exactlyN, "function n literal literal ...".  The test counts the models of the
    returned CNF over the literals' variables by brute force, and also checks
    that every model has the right number of true literals.
    """

    def __init__(self, question, testDict):
        super(CardinalityTest, self).__init__(question, testDict)
        self.cases = []
        for line in testDict['cases'].strip().split('\n'):
            pieces = line.split()
            if pieces[0] in ('atMostN', 'atLeastN', 'exactlyN'):
                self.cases.append((pieces[0], int(pieces[1]), [int(l) for l in pieces[2:]]))
            else:
                self.cases.append((pieces[0], None, [int(l) for l in pieces[1:]]))

    def countModels(self, module, function, n, literals):
        """
        Returns (count, errorMessage).
        """
        if function == 'exactlyN':
            clauses = module.exactlyN(list(literals), n)
            check = lambda k: k == n
        elif function == 'atMostN':
            clauses = module.atMostN(list(literals), n)
            check = lambda k: k <= n
        elif function == 'atLeastN':
            clauses = module.atLeastN(list(literals), n)
            check = lambda k: k >= n
        elif function == 'atLeastOneIsFalse':
            clauses = module.atLeastOneIsFalse(list(literals))
            check = lambda k: k <= len(literals) - 1
        else:
            clauses = module.atLeastOne(list(literals))
            check = lambda k: k >= 1
        variables = set(abs(l) for l in literals)
        error = checkCNF(clauses, variables)
        if error:
            return None, error
        found = models(clauses, variables)
        for model in found:
            numTrue = len([l for l in literals if (abs(l) in model) == (l > 0)])
            if not check(numTrue):
                return len(found), 'a model of your clauses makes %d of the literals true' % numTrue
        return len(found), None

    def execute(self, grades, moduleDict, solutionDict):
        module = moduleDict['sweeperAgents']
        expected = [int(c) for c in solutionDict['counts'].split()]
        passed = True
        for (function, n, literals), count in zip(self.cases, expected):
            call = '%s(%s%s)' % (function, literals, ', %d' % n if n is not None else '')
            got, error = self.countModels(module, function, n, literals)
            if error:
                self.addMessage('%s: %s' % (call, error))
                passed = False
            elif got != count:
                self.addMessage('%s: your clauses have %d models; expected %d' % (call, got, count))
                passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        module = moduleDict['sweeperAgents']
        counts = [self.countModels(module, f, n, lits)[0] for f, n, lits in self.cases]
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# Number of models of each case\'s clauses, in order.\n')
            handle.write('counts: "%s"\n' % ' '.join(str(c) for c in counts))
        return True

    def createPublicVersion(self):
        pass


class PerceptTest(testClasses.TestCase):
    """
    Each case is "x y count".  The student's clauses must have exactly the same
    models as the reference clauses stored in the solution file.
    """

    def __init__(self, question, testDict):
        super(PerceptTest, self).__init__(question, testDict)
        self.layout = layoutFromTest(testDict)
        self.cases = []
        for line in testDict['cases'].strip().split('\n'):
            x, y, count = [int(p) for p in line.split()]
            self.cases.append(((x, y), count))

    def perceptClauses(self, module, pos, count):
        varMap = logic.VarMap(self.layout.walls)
        return varMap, module.perceptClauses(varMap, pos, count, self.layout.walls)

    def execute(self, grades, moduleDict, solutionDict):
        module = moduleDict['sweeperAgents']
        passed = True
        for i, (pos, count) in enumerate(self.cases):
            varMap, clauses = self.perceptClauses(module, pos, count)
            reference = eval(solutionDict['case_%d' % i])
            error = checkCNF(clauses, set(range(1, varMap.numVars() + 1)))
            if error:
                self.addMessage('perceptClauses at %s, count %d: %s' % (pos, count, error))
                passed = False
                continue
            variables = set(abs(l) for c in clauses + reference for l in c)
            if models(clauses, variables) != models(reference, variables):
                self.addMessage('perceptClauses at %s, count %d: your clauses are not equivalent to the reference.' % (pos, count))
                self.addMessage('    yours:     %s' % varMap.cnfToString(clauses).replace('\n', ' '))
                self.addMessage('    reference: %s' % varMap.cnfToString(reference).replace('\n', ' '))
                passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        module = moduleDict['sweeperAgents']
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# Reference clauses for each case (variables numbered by logic.VarMap(walls)).\n')
            for i, (pos, count) in enumerate(self.cases):
                varMap, clauses = self.perceptClauses(module, pos, count)
                handle.write('case_%d: "%s"\n' % (i, clauses))
        return True

    def createPublicVersion(self):
        pass


class EntailsTest(testClasses.TestCase):
    """
    Each case is "clause ; clause ; ... | literal", e.g. "1 2 ; -1 | 2".
    """

    def __init__(self, question, testDict):
        super(EntailsTest, self).__init__(question, testDict)
        self.cases = []
        for line in testDict['cases'].strip().split('\n'):
            kbText, literal = line.split('|')
            kb = [[int(l) for l in c.split()] for c in kbText.split(';') if c.strip()]
            self.cases.append((kb, int(literal)))

    def execute(self, grades, moduleDict, solutionDict):
        module = moduleDict['sweeperAgents']
        expected = solutionDict['results'].split()
        passed = True
        for (kb, literal), answer in zip(self.cases, expected):
            got = module.entails(kb, literal)
            if str(got) != answer:
                self.addMessage('entails(%s, %d) returned %s; expected %s' % (kb, literal, got, answer))
                passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        module = moduleDict['sweeperAgents']
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('results: "%s"\n' % ' '.join(str(module.entails(kb, l)) for kb, l in self.cases))
        return True

    def createPublicVersion(self):
        pass


class ClassifyTest(testClasses.TestCase):
    """
    The board's G markers are the true ghosts.  The test builds a knowledge
    base from the sensor readings on the visited squares (using its own
    encoding, so Q2 does not depend on your perceptClauses), then compares your classify
    output on every open square with the reference.
    """

    def __init__(self, question, testDict):
        super(ClassifyTest, self).__init__(question, testDict)
        self.layout = layoutFromTest(testDict)
        self.visited = parseSquares(testDict['visited'])

    def classify(self, module):
        walls = self.layout.walls
        ghosts = set(pos for isPacman, pos in self.layout.agentPositions if not isPacman)
        varMap = logic.VarMap(walls)
        kb = []
        for pos in self.visited:
            neighbors = sweeper.getNeighbors(pos, walls)
            count = len([n for n in neighbors if n in ghosts])
            kb.append([-varMap.ghostVar(pos)])
            kb += exactlyNByEnumeration([varMap.ghostVar(n) for n in neighbors], count)
        squares = varMap.allPositions()
        return squares, module.classify(varMap, kb, squares)

    def execute(self, grades, moduleDict, solutionDict):
        squares, result = self.classify(moduleDict['sweeperAgents'])
        passed = True
        if not isinstance(result, dict):
            self.addMessage('classify should return a dict, got %r' % (result,))
            return self.testFail(grades)
        for label in ['safe', 'ghost', 'unknown']:
            expected = parseSquares(solutionDict[label])
            got = sorted(s for s in squares if result.get(s) == label)
            if got != expected:
                self.addMessage('squares classified %r:' % label)
                self.addMessage('    yours:    %s' % squaresToString(got))
                self.addMessage('    expected: %s' % squaresToString(expected))
                passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        squares, result = self.classify(moduleDict['sweeperAgents'])
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            for label in ['safe', 'ghost', 'unknown']:
                handle.write('%s: "%s"\n' % (label, squaresToString(sorted(s for s in squares if result[s] == label))))
        return True

    def createPublicVersion(self):
        pass


class SafetyCheckingAgent:
    """
    Wraps a SweeperAgent and records every move onto a square that was not
    provably safe given the readings so far, whether or not a ghost was there.
    """

    def __init__(self, agent):
        self.agent = agent
        self.unsafeMoves = []

    def registerInitialState(self, gameState):
        self.walls = gameState.getWalls()
        self.agent.registerInitialState(gameState)

    def getAction(self, gameState):
        action = self.agent.getAction(gameState)
        if action != sweeper.Directions.STOP and action in gameState.getLegalPacmanActions():
            x, y = gameState.getPacmanPosition()
            dx, dy = sweeper.Actions.directionToVector(action)
            target = (int(x + dx), int(y + dy))
            if target not in gameState.getVisitedReadings():
                varMap = logic.VarMap(self.walls)
                kb = []
                for pos, count in gameState.getVisitedReadings().items():
                    kb.append([-varMap.ghostVar(pos)])
                    kb += exactlyNByEnumeration([varMap.ghostVar(n) for n in sweeper.getNeighbors(pos, self.walls)], count)
                if logic.findModel(kb + [[varMap.ghostVar(target)]]) is not False:
                    self.unsafeMoves.append(target)
        return action


class SweeperGameTest(testClasses.TestCase):
    """
    Plays SweeperAgent on a board for each seed ("none" means the board's G
    markers).  Passes if every game is won, i.e. all the food is eaten, and
    Pacman never steps onto a square that was not provably safe at the time.
    """

    def __init__(self, question, testDict):
        super(SweeperGameTest, self).__init__(question, testDict)
        self.layout = layoutFromTest(testDict)
        self.layoutName = testDict['layout_name']
        self.seeds = [None if s == 'none' else int(s) for s in testDict['seeds'].split()]
        self.maxMoves = int(testDict['maxMoves'])

    def getDisplay(self):
        if self.question.getDisplay().checkNullDisplay():
            return textDisplay.NullGraphics()
        return sweeper.makeSweeperGraphics(frameTime=0.02)

    def execute(self, grades, moduleDict, solutionDict):
        module = moduleDict['sweeperAgents']
        passed = True
        for seed in self.seeds:
            agent = SafetyCheckingAgent(module.SweeperAgent())
            games = sweeper.runGames(self.layout, agent, self.getDisplay(),
                                     seed=seed, maxMoves=self.maxMoves, quiet=True)
            state = games[0].state
            if state.isWin():
                outcome = 'cleared the board'
            elif state.isLose():
                outcome = 'stepped on a ghost'
                passed = False
            elif state.isStuck():
                outcome = 'got stuck with %d food left' % state.goalFoodLeft()
                passed = False
            else:
                outcome = 'ran out of moves with %d food left' % state.goalFoodLeft()
                passed = False
            self.addMessage('%s (seed %s): %s, score %d' % (self.layoutName, seed, outcome, state.getScore()))
            if agent.unsafeMoves:
                self.addMessage('    stepped onto squares that were not provably safe: %s' % squaresToString(agent.unsafeMoves[:5]))
                passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# SweeperAgent must clear the board in every game, only ever stepping onto provably safe squares.\n')
        return True

    def createPublicVersion(self):
        pass


class SafeTargetsTest(testClasses.TestCase):
    """
    Sets up a SweeperAgent whose knowledge base holds the percepts of the
    first half of the visited squares (built by the test, so Q3a does not
    depend on your perceptClauses) and calls findSafeTargets.  It then adds the percepts of
    the remaining visited squares and calls findSafeTargets twice more, as
    getAction would on later turns.  Checks the targets returned after each
    stage, the ghosts recorded in self.known, and that self.known only ever
    holds proven ('safe' or 'ghost') squares.
    """

    def __init__(self, question, testDict):
        super(SafeTargetsTest, self).__init__(question, testDict)
        self.layout = layoutFromTest(testDict)
        self.visited = parseSquares(testDict['visited'])

    def findSafeTargets(self, module):
        """
        Returns (targets after the first half, targets after all squares
        (twice), ghost squares in self.known, bad entries in self.known).
        """
        walls = self.layout.walls
        ghosts = set(pos for isPacman, pos in self.layout.agentPositions if not isPacman)
        agent = module.SweeperAgent()
        agent.walls = walls
        agent.varMap = logic.VarMap(walls)
        agent.kb = []
        agent.visited = set()
        agent.known = {}
        badEntries = set()

        def visit(squares):
            for pos in squares:
                neighbors = sweeper.getNeighbors(pos, walls)
                count = len([n for n in neighbors if n in ghosts])
                agent.kb.append([-agent.varMap.ghostVar(pos)])
                agent.kb += exactlyNByEnumeration([agent.varMap.ghostVar(n) for n in neighbors], count)
                agent.visited.add(pos)
                agent.known[pos] = 'safe'

        def call():
            result = agent.findSafeTargets()
            badEntries.update((s, v) for s, v in agent.known.items() if v not in ('safe', 'ghost'))
            return result

        half = (len(self.visited) + 1) // 2
        visit(self.visited[:half])
        partial = call()
        visit(self.visited[half:])
        full = [call(), call()]
        ghostSquares = sorted(s for s, status in agent.known.items() if status == 'ghost')
        return partial, full, ghostSquares, sorted(badEntries)

    def execute(self, grades, moduleDict, solutionDict):
        partial, full, ghostSquares, badEntries = self.findSafeTargets(moduleDict['sweeperAgents'])
        expectedPartial = parseSquares(solutionDict['targets_partial'])
        expectedTargets = parseSquares(solutionDict['targets'])
        expectedGhosts = parseSquares(solutionDict['ghosts'])
        passed = True
        checks = [('after the first %d visited squares' % ((len(self.visited) + 1) // 2), partial, expectedPartial),
                  ('after all visited squares (first call)', full[0], expectedTargets),
                  ('after all visited squares (second call)', full[1], expectedTargets)]
        for label, got, expected in checks:
            if not isinstance(got, list) or got != expected:
                self.addMessage('%s, findSafeTargets returned %s' % (label, got if not isinstance(got, list) else squaresToString(got)))
                self.addMessage('    expected the sorted list: %s' % squaresToString(expected))
                passed = False
        if badEntries:
            self.addMessage('self.known should only hold proven squares (\'safe\' or \'ghost\'); found %s'
                            % ', '.join('%r at %d,%d' % (v, s[0], s[1]) for s, v in badEntries[:5]))
            self.addMessage('    A square that is unknown now may become provable after later readings.')
            passed = False
        if ghostSquares != expectedGhosts:
            self.addMessage('squares recorded as \'ghost\' in self.known: %s' % squaresToString(ghostSquares))
            self.addMessage('    expected: %s' % squaresToString(expectedGhosts))
            passed = False
        return self.testPass(grades) if passed else self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        partial, full, ghostSquares, badEntries = self.findSafeTargets(moduleDict['sweeperAgents'])
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('targets_partial: "%s"\n' % squaresToString(partial))
            handle.write('targets: "%s"\n' % squaresToString(full[0]))
            handle.write('ghosts: "%s"\n' % squaresToString(ghostSquares))
        return True

    def createPublicVersion(self):
        pass
