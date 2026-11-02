# sweeper.py
# ----------
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
Sweeper.py is Ghost Sweeper: a variant of Pacman where the ghosts are
stationary, invisible and deadly.  Stepping onto a ghost's square ends the
game.  On every square Pacman visits, a sensor reports how many ghosts are on
the four neighboring squares (north, south, east and west).  Food starts on
every open square, so it gives nothing away; Pacman wins once he has eaten the
food on every ghost-free square he can reach.

To play your first game, type 'python sweeper.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys); 'q' gives up.
"""
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from game import Grid
from busters import default, parseAgentArgs, loadAgent
import sys, util, time, random, layout

TIME_PENALTY = 1  # Number of points lost each move
FOOD_REWARD = 10
WIN_REWARD = 500
DEATH_PENALTY = 500

NEIGHBOR_DIRECTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

def getNeighbors(pos, walls):
    """
    Returns the non-wall squares directly north, south, east and west of pos
    (in that order).
    """
    x, y = pos
    neighbors = []
    for direction in NEIGHBOR_DIRECTIONS:
        dx, dy = Actions.directionToVector(direction)
        nx, ny = int(x + dx), int(y + dy)
        if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
            neighbors.append((nx, ny))
    return neighbors

def hideGhosts(lay):
    """
    Returns a copy of a layout with its ghost markers removed, so that agents
    cannot read the ghost positions (or how many ghosts there are) off it.
    """
    return layout.Layout([row.translate(str.maketrans('G1234', '     ')) for row in lay.layoutText])

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the state of a Ghost Sweeper game: Pacman's
    position, the remaining food, the score and the sensor reading on
    Pacman's current square.

    The ghost positions are hidden: the states your agent receives do not
    contain them, and neither does the number of ghosts on the board.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def getLegalActions(self, agentIndex=0):
        """
        Returns the legal actions for Pacman.  Directions.STOP is always
        legal; returning it means Pacman is stuck, which ends the game.
        """
        if self.isOver(): return []
        return Actions.getPossibleActions(self.getPacmanState().configuration, self.getWalls())

    def getLegalPacmanActions(self):
        return self.getLegalActions(0)

    def getPacmanState(self):
        return self.data.agentStates[0].copy()

    def getPacmanPosition(self):
        return self.data.agentStates[0].getPosition()

    def getAdjacentGhostCount(self):
        """
        Returns the sensor reading on Pacman's current square: the number of
        ghosts on the (up to four) non-wall squares directly north, south,
        east and west of Pacman.  The reading says nothing about which of
        those squares hold the ghosts.
        """
        return self.data.readings.get(self.getPacmanPosition())

    def getVisitedReadings(self):
        """
        Returns a dict {(x, y): count} of the sensor readings on every square
        Pacman has visited so far.
        """
        return dict(self.data.readings)

    def getNumAgents(self):
        return 1

    def getScore(self):
        return self.data.score

    def getNumFood(self):
        """
        Returns how much food is left on the board, including food on ghost
        squares that can never be eaten.
        """
        return self.data.food.count()

    def getFood(self):
        """
        Returns a Grid of boolean food indicator variables.  Food starts on
        every open square except Pacman's start, so it says nothing about
        where the ghosts are.
        """
        return self.data.food

    def getWalls(self):
        """
        Returns a Grid of boolean wall indicator variables.
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

    def isWin(self):
        """
        True once Pacman has eaten the food on every ghost-free square he can
        reach from his start.  Food on ghost squares (and on any squares the
        ghosts seal off) can never be eaten and does not count.
        """
        return self.data._win

    def isLose(self):
        """
        True if Pacman stepped onto a ghost.
        """
        return self.data._lose

    def isStuck(self):
        """
        True if Pacman gave up by returning Directions.STOP.
        """
        return self.data.stuck

    def isOver(self):
        outOfMoves = self.maxMoves > 0 and self.numMoves >= self.maxMoves
        return self.isWin() or self.isLose() or self.isStuck() or outOfMoves

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prevState=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None:
            self.data = GameStateData(prevState.data)
            self.data.readings = dict(prevState.data.readings)
            self.data.stuck = prevState.data.stuck
            self.data._win = prevState.data._win
            self.data._lose = prevState.data._lose
            self.data.revealedGhosts = prevState.data.revealedGhosts
            self._ghosts = prevState._ghosts
            self._goal = prevState._goal
            self._agentLayout = prevState._agentLayout
            self.numMoves = prevState.numMoves
            self.maxMoves = prevState.maxMoves
        else:
            self.data = GameStateData()
            self.data.readings = {}
            self.data.stuck = False
            self.data.revealedGhosts = None
            self._ghosts = None
            self._goal = None
            self._agentLayout = None
            self.numMoves = 0
            self.maxMoves = -1

    def getResult(self, agentIndex, action):
        """
        Returns the state after Pacman takes the action.
        """
        if self.isOver(): raise Exception('Can\'t generate a result of a terminal state.')
        if action not in self.getLegalActions(agentIndex):
            raise Exception("Illegal action {}".format(action))

        state = GameState(self)
        state.data._agentMoved = 0
        state.numMoves += 1
        if action == Directions.STOP:
            state.data.stuck = True
        else:
            pacmanState = state.data.agentStates[0]
            vector = Actions.directionToVector(action, 1)
            pacmanState.configuration = pacmanState.configuration.generateSuccessor(vector)
            state.data.scoreChange -= TIME_PENALTY
            state.visit(state.getPacmanPosition())
        if state.isOver() and state._ghosts is not None:
            state.data.revealedGhosts = sorted(state._ghosts)
        state.data.score += state.data.scoreChange
        return state

    def getPacmanResult(self, action):
        return self.getResult(0, action)

    def visit(self, pos):
        """
        Resolves Pacman arriving at pos: death, food and the sensor reading.
        States handed to agents do not know where the ghosts are, so on those
        no reading (and no death) is produced.
        """
        if self._ghosts is None:
            return
        if pos in self._ghosts:
            self.data._lose = True
            self.data.scoreChange -= DEATH_PENALTY
            return
        x, y = pos
        if self.data.food[x][y]:
            self.data.food[x][y] = False
            self.data._foodEaten = pos
            self.data.scoreChange += FOOD_REWARD
            if self.goalFoodLeft() == 0:
                self.data._win = True
                self.data.scoreChange += WIN_REWARD
        neighbors = getNeighbors(pos, self.getWalls())
        self.data.readings[pos] = len([n for n in neighbors if n in self._ghosts])

    def goalFoodLeft(self):
        """
        How much of the food Pacman must eat to win is still left.  Only the
        game's own state knows this; on states handed to agents it is None.
        """
        if self._goal is None:
            return None
        return len([1 for x, y in self._goal if self.data.food[x][y]])

    def deepCopy(self):
        """
        Returns a copy of the state with the ghost positions removed, both the
        hidden list and the layout's ghost markers.  This is what agents
        receive.
        """
        state = GameState(self)
        state.data = self.data.deepCopy()
        if self._agentLayout is not None:
            state.data.layout = self._agentLayout
        state.data.readings = dict(self.data.readings)
        state.data.stuck = self.data.stuck
        state.data._win = self.data._win
        state.data._lose = self.data._lose
        state.data.revealedGhosts = None
        state._ghosts = None
        state._goal = None
        return state

    def __eq__(self, other):
        if other is None:
            return False
        return self.data == other.data and self.data.readings == other.data.readings

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return str(self.data)

    def initialize(self, layout, ghostPositions):
        """
        Creates an initial game state from a layout and the (hidden) ghost
        positions.
        """
        self.data.initialize(layout, 0)
        self.data.capsules = []
        self._ghosts = frozenset(ghostPositions)
        self._agentLayout = hideGhosts(layout)
        walls = layout.walls
        start = self.getPacmanPosition()
        if start in self._ghosts:
            raise Exception('Pacman cannot start on a ghost')
        if any(n in self._ghosts for n in getNeighbors(start, walls)):
            raise Exception('Pacman must start on a square with no adjacent ghosts')

        # Food covers every open square, so it reveals nothing about the ghosts.
        # Pacman wins once he has eaten the food on every ghost-free square
        # reachable without crossing a ghost (the hidden goal).
        food = Grid(walls.width, walls.height, False)
        for x, y in walls.asList(False):
            if (x, y) != start:
                food[x][y] = True
        reachable = set([start])
        frontier = [start]
        while frontier:
            pos = frontier.pop()
            for n in getNeighbors(pos, walls):
                if n not in reachable and n not in self._ghosts:
                    reachable.add(n)
                    frontier.append(n)
        self._goal = frozenset(reachable - set([start]))
        self.data.food = food
        self.visit(start)

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

def getGhostPositions(layout, seed=None):
    """
    Returns the ghost positions for a game.  Without a seed, these are the
    layout's G markers.  With a seed, the same number of ghosts are placed at
    random, keeping Pacman's start square and its neighbors ghost-free.
    """
    fromLayout = [pos for isPacman, pos in layout.agentPositions if not isPacman]
    if seed is None:
        return fromLayout
    start = [pos for isPacman, pos in layout.agentPositions if isPacman][0]
    forbidden = set([start] + getNeighbors(start, layout.walls))
    candidates = [pos for pos in sorted(layout.walls.asList(False)) if pos not in forbidden]
    return sorted(random.Random(seed).sample(candidates, len(fromLayout)))

class SweeperGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, quiet=False):
        self.quiet = quiet

    def newGame(self, layout, pacmanAgent, display, ghostPositions, maxMoves=-1):
        initState = GameState()
        initState.initialize(layout, ghostPositions)
        initState.maxMoves = maxMoves
        game = Game([pacmanAgent], display, self)
        game.state = initState
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if not state.isOver(): return
        game.gameOver = True
        if self.quiet: return
        if state.isWin():
            print("Pacman cleared the board! Score: %d" % state.getScore())
        elif state.isLose():
            print("Pacman stepped on a ghost! Score: %d" % state.getScore())
        elif state.isStuck():
            print("Pacman is stuck with %d food left. Score: %d" % (state.goalFoodLeft(), state.getScore()))
        else:
            print("Pacman ran out of moves. Score: %d" % state.getScore())

    def agentCrash(self, game, agentIndex):
        print("Pacman crashed")

#############
# DISPLAYS  #
#############

READING_COLORS = {0: (.5, .5, .5), 1: (.3, .5, 1.), 2: (.2, .8, .2), 3: (1., .3, .3), 4: (.8, .3, 1.)}

class SweeperTextGraphics:
    """
    Prints the board after every move.  Visited squares show their sensor
    reading; once the game is over the ghosts are shown as G.
    """
    def __init__(self, frameTime=0.0):
        self.frameTime = frameTime

    def initialize(self, state, isBlue=False):
        self.draw(state)

    def update(self, state):
        self.draw(state)
        if self.frameTime > 0: time.sleep(self.frameTime)

    def draw(self, state):
        walls = state.layout.walls
        pacman = state.agentStates[0].getPosition()
        ghosts = set(state.revealedGhosts or [])
        rows = []
        for y in range(walls.height - 1, -1, -1):
            row = ''
            for x in range(walls.width):
                if (x, y) == pacman: row += 'P'
                elif (x, y) in ghosts: row += 'G'
                elif walls[x][y]: row += '%'
                elif (x, y) in state.readings: row += str(state.readings[(x, y)])
                elif state.food[x][y]: row += '.'
                else: row += ' '
            rows.append(row)
        print('\n'.join(rows))
        print('Score: %d   Ghosts adjacent to Pacman: %s\n' % (state.score, state.readings.get(pacman)))

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass

def makeSweeperGraphics(zoom=1.0, frameTime=0.1):
    """
    Builds the graphical display.  Ghosts stay hidden until the game ends;
    visited squares are shaded and labeled with their sensor reading.
    """
    import graphicsDisplay
    import graphicsUtils
    from game import AgentState, Configuration

    class SweeperGraphics(graphicsDisplay.PacmanGraphics):
        def initialize(self, state, isBlue=False):
            self.isBlue = isBlue
            self.startGraphics(state)
            self.distributionImages = None
            self.drawWalls(state.layout.walls)
            self.food = self.drawFood(state.food)
            self.capsules = {}
            self.readingImages = {}
            self.ghostsRevealed = False
            self.drawReadings(state)
            self.drawAgentObjects(state)
            self.previousState = state
            graphicsUtils.refresh()

        def drawReadings(self, state):
            for pos, count in sorted(state.readings.items()):
                if pos in self.readingImages: continue
                screen_x, screen_y = self.to_screen(pos)
                shade = graphicsUtils.square((screen_x, screen_y), 0.5 * self.gridSize,
                                             color=graphicsUtils.formatColor(.15, .15, .15),
                                             filled=1, behind=2)
                label = graphicsUtils.text((screen_x, screen_y),
                                           graphicsUtils.formatColor(*READING_COLORS.get(count, (1, 1, 1))),
                                           str(count), "Times", max(8, int(0.4 * self.gridSize)), "bold",
                                           anchor="center")
                self.readingImages[pos] = (shade, label)
            # Keep Pacman drawn on top of the reading on his own square
            if getattr(self, 'agentImages', None):
                for item in self.agentImages[0][1]:
                    graphicsUtils._canvas.tag_raise(item)

        def update(self, newState):
            graphicsDisplay.PacmanGraphics.update(self, newState)
            self.drawReadings(newState)
            if newState.revealedGhosts and not self.ghostsRevealed:
                self.ghostsRevealed = True
                for pos in newState.revealedGhosts:
                    self.drawGhost(AgentState(Configuration(pos, Directions.STOP), False), 1)
            graphicsUtils.refresh()

    return SweeperGraphics(zoom, frameTime=frameTime)

class SweeperKeyboardAgent:
    """
    Lets you play Ghost Sweeper yourself: one move per key press.  The keys
    are 'a', 's', 'd', and 'w' (or arrow keys); 'q' gives up.
    """
    KEYS = {'w': Directions.NORTH, 'Up': Directions.NORTH, 's': Directions.SOUTH, 'Down': Directions.SOUTH,
            'd': Directions.EAST, 'Right': Directions.EAST, 'a': Directions.WEST, 'Left': Directions.WEST,
            'q': Directions.STOP}

    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        from graphicsUtils import keys_waiting, sleep
        legal = state.getLegalActions(self.index)
        while True:
            for key in keys_waiting():
                if key in self.KEYS and self.KEYS[key] in legal:
                    return self.KEYS[key]
            sleep(0.05)

#############################
# FRAMEWORK TO START A GAME #
#############################

def readCommand(argv):
    """
    Processes the command used to run Ghost Sweeper from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python sweeper.py <options>
    EXAMPLES:   (1) python sweeper.py
                    - starts an interactive game
                (2) python sweeper.py -l sweeperOpen -p SweeperAgent
                    - runs your logic agent on the sweeperOpen board
                (3) python sweeper.py -l sweeperOpen -p SweeperAgent --seed 3
                    - same, with randomly placed ghosts
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='sweeperSmall')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the sweeperAgents module to use'),
                      metavar='TYPE', default='SweeperKeyboardAgent')
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('--seed', dest='seed', type='int',
                      help='Place the ghosts at random using this seed (game i uses seed + i) '
                           'instead of at the layout\'s G markers', default=None)
    parser.add_option('-m', '--maxMoves', dest='maxMoves', type='int',
                      help=default('End the game after this many moves; -1 means no limit'), default=-1)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")

    noGraphics = options.quietGraphics or options.textGraphics
    if options.pacman == 'SweeperKeyboardAgent':
        if noGraphics: raise Exception('Using the keyboard requires graphics (not text display)')
        pacmanType = SweeperKeyboardAgent
    else:
        pacmanType = loadAgent(options.pacman, noGraphics)
    args['pacman'] = pacmanType(**parseAgentArgs(options.agentArgs))

    if options.quietGraphics:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        args['display'] = SweeperTextGraphics(frameTime=options.frameTime)
    else:
        args['display'] = makeSweeperGraphics(options.zoom, frameTime=options.frameTime)
    args['numGames'] = options.numGames
    args['seed'] = options.seed
    args['maxMoves'] = options.maxMoves
    args['quiet'] = options.quietGraphics
    return args

def runGames(layout, pacman, display, numGames=1, seed=None, maxMoves=-1, quiet=False):
    """
    Plays numGames games and returns the list of finished Game objects.
    """
    rules = SweeperGameRules(quiet)
    games = []

    for i in range(numGames):
        ghostPositions = getGhostPositions(layout, None if seed is None else seed + i)
        game = rules.newGame(layout, pacman, display, ghostPositions, maxMoves)
        game.run()
        games.append(game)

    if numGames > 1 and not quiet:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate))
        print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

    return games

if __name__ == '__main__':
    """
    The main function called when sweeper.py is run
    from the command line:

    > python sweeper.py

    See the usage string for more details.

    > python sweeper.py --help
    """
    args = readCommand(sys.argv[1:])  # Get game components based on input
    runGames(**args)
