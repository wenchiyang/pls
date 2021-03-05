# warehouse.py
#----------
# The shielded warehouse code is build on the PAC-MAN environment
# from UC Berkeley.
#
# ---------
# Licensing Information from UC Berkeley:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import os
import random
import sys

import layout
from game import Actions
from game import Directions
from game import Game
from game import GameStateData
from util import manhattanDistance
from util import nearestPoint


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the packages,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    # static variable keeps track of which states have had getLegalActions called
    explored = set()

    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    getAndResetExplored = staticmethod(getAndResetExplored)

    def getLegalActions(self, agentIndex=0):
        """
        Returns the legal actions for the agent specified.
        """
        #        GameState.explored.add(self)
        if self.isWin() or self.isLose():
            return []

        if agentIndex == 0:  # Pacman is moving
            return PacmanRules.getLegalActions(self)
        else:
            return GhostRules.getLegalActions(self, agentIndex)

    def generateSuccessor(self, agentIndex, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose():
            raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agentIndex == 0:  # Pacman is moving
            state.data._eaten = [False for i in range(state.getNumAgents())]
            PacmanRules.applyAction(state, action)
        else:  # A ghost is moving
            GhostRules.applyAction(state, action, agentIndex)

        # Time passes
        if agentIndex == 0:
            state.data.scoreChange += -TIME_PENALTY  # Penalty for waiting around
        else:
            GhostRules.decrementTimer(state.data.agentStates[agentIndex])

        # Resolve multi-agent effects
        GhostRules.checkDeath(state, agentIndex)

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
        GameState.explored.add(self)
        GameState.explored.add(state)
        return state

    def getLegalPacmanActions(self):
        return self.getLegalActions(0)

    def generatePacmanSuccessor(self, action):
        """
        Generates the successor state after the specified pacman move
        """
        return self.generateSuccessor(0, action)

    def getPacmanState(self):
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agentStates[0].copy()

    def getPacmanPosition(self):
        return self.data.agentStates[0].getPosition()

    def getPacmanDirection(self):
        return self.data.agentStates[0].getDirection()

    def getGhostStates(self):
        return self.data.agentStates[1:]

    def getGhostState(self, agentIndex):
        if agentIndex == 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getGhostState")
        return self.data.agentStates[agentIndex]

    def getGhostPosition(self, agentIndex):
        if agentIndex == 0:
            raise Exception("Pacman's index passed to getGhostPosition")
        return self.data.agentStates[agentIndex].getPosition()

    def getGhostDirection(self, agentIndex):
        if agentIndex == 0:
            raise Exception("Pacman's index passed to getGhostPosition")
        return self.data.agentStates[agentIndex].getDirection()

    def getGhostPositions(self):
        return [s.getPosition() for s in self.getGhostStates()]

    def getGhostDirections(self):
        return [s.getDirection() for s in self.getGhostStates()]

    def getNumAgents(self):
        return len(self.data.agentStates)

    def getScore(self):
        return float(self.data.score)

    def getNumPackages(self):
        return self.data.packages.count()

    def getLoadingInfo(self):
        return self.data.agentStates[0].getLoadingInfo()

    def getExit(self):
        exit = self.data.layout.exitPos
        if exit == None:
            print("[Error] Layout does not contain an exit!")
        assert (exit != None)
        return exit

    def subFrame(self, x, y, r):
        # Copy current state
        state = GameState()
        state.data = self.data.subFrame(x, y, r)
        return state

    def getPackages(self):
        """
        Returns a Grid of boolean package indicator variables.
        """
        return self.data.packages

    # For a adversery, the packages are also walls
    def getWalls(self, adversary=False):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is a wall at (x,y), just call

        walls = state.getWalls()
        if walls[x][y] == True: ...
        """
        if adversary:
            return self.data.layout.wallsGhost

        return self.data.layout.walls


    def hasPackage(self, x, y):
        return self.data.packages[x][y]

    def hasWall(self, x, y, adversary=False):

        if adversary:
            return self.data.layout.wallsGhost[x][y]

        return self.data.layout.walls[x][y]

    def isLose(self):
        return self.data._lose

    def isWin(self):
        return self.data._win

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prevState=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState != None:  # Initial state
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        return state

    def __eq__(self, other):
        """
        Allows two states to be compared.
        """
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.data)

    def __str__(self):

        return str(self.data)

    def initialize(self, layout, numGhostAgents=1000):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numGhostAgents)


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

SCARED_TIME = 40  # Moves ghosts are scared
COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY = 1  # Number of points lost each round


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame(self, layout, pacmanAgent, ghostAgents, display, quiet=False, catchExceptions=False, symX=False,
                symY=False, distCrossings=0):
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize(layout, len(ghostAgents))
        game = Game(agents, display, self, catchExceptions=catchExceptions, symX=symX, symY=symY, distCrossings=distCrossings)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)

    def win(self, state, game):
        #if not self.quiet:
        print("Pacman emerges victorious! Score: %d" % state.data.score)
        game.gameOver = True

    def lose(self, state, game):
        #if not self.quiet:
        x_pac = state.getPacmanPosition()[0]
        y_pac = state.data.layout.height - state.getPacmanPosition()[1] - 1
        print("Pacman died! Score: %d At Position: %d %d" % (state.data.score, x_pac, y_pac))
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumPackages()) / self.initialState.getNumPackages()

    def agentCrash(self, game, agentIndex):
        if agentIndex == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    def getMaxTotalTime(self, agentIndex):
        return self.timeout

    def getMaxStartupTime(self, agentIndex):
        return self.timeout

    def getMoveWarningTime(self, agentIndex):
        return self.timeout

    def getMoveTimeout(self, agentIndex):
        return self.timeout

    def getMaxTimeWarnings(self, agentIndex):
        return 0


class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    PACMAN_SPEED = 1

    def getLegalActions(state):
        """
        Returns a list of possible actions.
        """
        if state.data.agentStates[0]._loaded:
            return Actions.getPossibleActions(state.getPacmanState().configuration, state.data.layout.wallsGhost)
        return Actions.getPossibleActions(state.getPacmanState().configuration, state.data.layout.walls)

    def applyAction(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.getLegalActions(state)
        #TODO!!!!!!!!!!!!!!
        #print("ERROR: Illegal action " + str(action))
        #if action not in legal:
        #    raise Exception("Illegal action " + str(action))

        pacmanState = state.data.agentStates[0]

        # Update Configuration
        vector = Actions.directionToVector(action, PacmanRules.PACMAN_SPEED)
        pacmanState.configuration = pacmanState.configuration.generateSuccessor(vector)

        # Eat /load
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint(next)
        x, y = nearest
        if manhattanDistance(nearest, next) <= 0.5 and state.data.packages[x][y] == True and pacmanState._loaded == False:
            # Remove package
            pacmanState._loaded = True
            PacmanRules.consume(nearest, state)

        # unload
        if pacmanState.configuration.getPosition() in state.data.layout.exitPos and pacmanState._loaded == True:
            pacmanState._loaded = False
            state.data.scoreChange += 25

        # check for win
        # TODO: cache num_packages?
        num_packages = state.getNumPackages()

        # TODO: Warehouse: changed winning condition
        if num_packages == 0 and not state.data._lose:
            if pacmanState.configuration.getPosition() in state.data.layout.exitPos:
                state.data.scoreChange += 500
                state.data._win = True

    applyAction = staticmethod(applyAction)

    def consume(position, state):
        x, y = position
        # Load Package
        if state.data.packages[x][y]:
            state.data.scoreChange += 25
            state.data.packages = state.data.packages.copy()
            state.data.packages[x][y] = False
            state.data._package_eaten = position

    consume = staticmethod(consume)


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED = 1.0

    def getLegalActions(state, ghostIndex):
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.getGhostState(ghostIndex).configuration
        possibleActions = Actions.getPossibleActions(conf, state.data.layout.wallsGhost)
        reverse = Actions.reverseDirection(conf.direction)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        if reverse in possibleActions and len(possibleActions) > 1:
            possibleActions.remove(reverse)
        return possibleActions

    getLegalActions = staticmethod(getLegalActions)

    def applyAction(state, action, ghostIndex):
        ghostState = state.data.agentStates[ghostIndex]
        speed = GhostRules.GHOST_SPEED
        if ghostState.scaredTimer > 0:
            speed /= 2.0
        vector = Actions.directionToVector(action, speed)
        ghostState.configuration = ghostState.configuration.generateSuccessor(vector)

    applyAction = staticmethod(applyAction)

    def decrementTimer(ghostState):
        timer = ghostState.scaredTimer
        if timer == 1:
            ghostState.configuration.pos = nearestPoint(ghostState.configuration.pos)
        ghostState.scaredTimer = max(0, timer - 1)

    decrementTimer = staticmethod(decrementTimer)

    def checkDeath(state, agentIndex):
        pacmanPosition = state.getPacmanPosition()
        if agentIndex == 0:  # Pacman just moved; Anyone can kill him
            for index in range(1, len(state.data.agentStates)):
                ghostState = state.data.agentStates[index]
                ghostPosition = ghostState.configuration.getPosition()
                if GhostRules.canKill(pacmanPosition, ghostPosition):
                    GhostRules.collide(state, ghostState, index)
        else:
            ghostState = state.data.agentStates[agentIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if GhostRules.canKill(pacmanPosition, ghostPosition):
                GhostRules.collide(state, ghostState, agentIndex)

    checkDeath = staticmethod(checkDeath)

    def collide(state, ghostState, agentIndex):
        if ghostState.scaredTimer > 0:
            state.data.scoreChange += 200
            GhostRules.placeGhost(state, ghostState)
            ghostState.scaredTimer = 0
            # Added for first-person
            state.data._eaten[agentIndex] = True
        else:
            if not state.data._win:
                state.data.scoreChange -= 500
                state.data._lose = True

    collide = staticmethod(collide)

    def canKill(pacmanPosition, ghostPosition):
        return manhattanDistance(ghostPosition, pacmanPosition) <= COLLISION_TOLERANCE

    canKill = staticmethod(canKill)

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start

    placeGhost = staticmethod(placeGhost)


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'


def parseAgentArgs(str):
    if str == None:
        return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python warehouse.py <options>
    EXAMPLES:   (1) python warehouse.py
                    - starts an interactive game
                (2) python warehouse.py --layout smallClassic --zoom 2
                OR  python warehouse.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int', help=default('the number of GAMES to play'),
                      metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'), metavar='LAYOUT_FILE',
                      default='mediumClassic')
    parser.add_option('-d', '--dump', dest='dump', help=default('the DUMP_FILE to which to dump the shield to'),
                      metavar='DUMP_FILE', default='')
    parser.add_option('-o', '--open', dest='open', help=default('the OPEN_FILE from which to load the shield'),
                      metavar='OPEN_FILE', default='')
    parser.add_option('-p', '--pacman', dest='pacman', help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'), metavar='TYPE',
                      default='RandomGhost')
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom', help=default('Zoom the size of the graphics window'),
                      default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='gameToReplay', help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3,opt4=val4"')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('-y', '--numGhostTraining', dest='numGhostTraining', type='int',
                      help=default('How many episodes are used to learn ghost models (suppresses output)'), default=0)
    parser.add_option('-w', '--withoutShield', dest='withoutShield', type='int',
                      help=default('Learning without a shield to get safe actions'), default=0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)
    parser.add_option('-i', '--symX', action='store_true', dest='symX',
                      help='enables optimizations for x-symetric labyrinths', default=False)
    parser.add_option('-j', '--symY', action='store_true', dest='symY',
                      help='Genables optimizations for y-symetric labyrinths', default=False)
    parser.add_option('-b', '--distCrossings', dest='distCrossings', type='int',
                      help=default('distance to the exit, in which crossigns will be shielded'), default=0)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed:
        random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] == None:
        raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    noKeyboard = options.gameToReplay == None and (options.textGraphics or options.quietGraphics)
    pacmanType = loadAgent(options.pacman, noKeyboard)
    agentOpts = parseAgentArgs(options.agentArgs)

    if options.distCrossings > 0:
        args['distCrossings'] = options.distCrossings
        if 'distCrossings' not in agentOpts:
            agentOpts['distCrossings'] = options.distCrossings

    if options.numTraining > 0:
        args['numTraining'] = options.numTraining
        if 'numTraining' not in agentOpts:
            agentOpts['numTraining'] = options.numTraining

    if options.numGhostTraining > 0:
        args['numGhostTraining'] = options.numGhostTraining
        if 'numGhostTraining' not in agentOpts:
            agentOpts['numGhostTraining'] = options.numGhostTraining

    if options.withoutShield > 0:
        args['withoutShield'] = options.withoutShield
        if 'withoutShield' not in agentOpts:
            agentOpts['withoutShield'] = options.withoutShield

    pacman = pacmanType(**agentOpts)  # Instantiate Pacman with agentArgs
    args['pacman'] = pacman

    if str(type(pacman)) == "<class 'qlearningAgents.ApproximateQAgent'>":
        pacman.setDumpParameters(options.dump, options.open)
        pacman.setSymmetryParameters(options.symX, options.symY)
        pacman.setDistanceParameter(options.distCrossings)

    # Don't display training games
    if 'numTrain' in agentOpts:
        options.numQuiet = int(agentOpts['numTrain'])
        options.numIgnore = int(agentOpts['numTrain'])

    # Choose a ghost agent
    ghostType = loadAgent(options.ghost, noKeyboard)
    if options.ghost == "ForkTruckPath":
        args['layout'].getPaths()
        args['ghosts'] = [ghostType(i + 1, args['layout'].paths) for i in range(args['layout'].numGhosts)]
    else:
        args['ghosts'] = [ghostType(i + 1) for i in range(options.numGhosts)]

    # Choose a display format
    if options.quietGraphics:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.textGraphics:
        import textDisplay
        textDisplay.SLEEP_TIME = options.frameTime
        args['display'] = textDisplay.PacmanGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime=options.frameTime)
    args['numGames'] = options.numGames
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    args['timeout'] = options.timeout
    args['symX'] = options.symX
    args['symY'] = options.symY

    # Special case: recorded games don't use the runGames method or args structure
    if options.gameToReplay != None:
        print('Replaying recorded game %s.' % options.gameToReplay)
        import pickle
        f = open(options.gameToReplay)
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    return args


def loadAgent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        pythonPathDirs = pythonPathStr.split(':')
    else:
        pythonPathDirs = pythonPathStr.split(';')
    pythonPathDirs.append('.')

    # TODO:
    # print("Remove this hack for Stefans Laptop....")

    # return getattr(module, pacman)

    for moduleDir in pythonPathDirs:
        if not os.path.isdir(moduleDir):
            continue
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('gents.py')]
        # print (moduleNames)
        for modulename in moduleNames:
            # print(modulename)
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            # print(module)
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')


def replayGame(layout, actions, display):
    import avatarAgents, adversaryAgents
    rules = ClassicGameRules()
    agents = [avatarAgents.GreedyAgent()] + [adversaryAgents.RandomGhost(i + 1) for i in range(layout.getNumGhosts())]
    game = rules.newGame(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
        # Execute the action
        state = state.generateSuccessor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()


def runGames(layout, pacman, ghosts, display, numGames, record, numTraining=0, numGhostTraining=0, withoutShield=0, distCrossings=0,
             catchExceptions=False, timeout=60, symX=False, symY=False):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []
    stat_games = []
    last_n_games = []
    average_scores = []
    win_rates = []

    for i in range(numGames):
        print("Game Nr %d" % (i))
        beQuiet = i < numTraining + numGhostTraining
        #beQuiet = False
        if beQuiet:
            # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display

            rules.quiet = False

        layout_copy = layout.deepCopy()
        game = rules.newGame(layout_copy, pacman, ghosts, gameDisplay, beQuiet, catchExceptions, symX, symY, distCrossings)

        game.run()
        if not beQuiet:
            games.append(game)
        stat_games.append(game)
        last_n_games.append(game)

        if (i + 1) % 10 == 0:
            print("Games %d - %d " % (i, i + 10))
            scores = [game.state.getScore() for game in last_n_games]
            wins = [game.state.isWin() for game in last_n_games]
            winRate = wins.count(True) / float(len(wins))
            print('Average Score :', sum(scores) / float(len(scores)))
            print('Scores:       ', ', '.join([str(score) for score in scores]))
            print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate))
            print("------------------------------------------")

            last_n_games = []
            average_scores.append(sum(scores) / float(len(scores)))
            win_rates.append(winRate)

        if i == numTraining + numGhostTraining - 1:
            # compute training statistic
            print("==============================================================")
            print("Statistic - Training Phase")
            if withoutShield > 0:
                print("-- NO shield used for Training")
            else:
                print("-- Shield used for Training to get safe actions")

            scores = [game.state.getScore() for game in stat_games]
            wins = [game.state.isWin() for game in stat_games]
            winRate = wins.count(True) / float(len(wins))
            print('Average Score :', sum(scores) / float(len(scores)))
            print('Average Scores per 10 episodes: ', ', '.join([str(score) for score in average_scores]))
            print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate))
            print('Win Rate per 10 episodes: ', ', '.join([str(win_rate) for win_rate in win_rates]))
            print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

            print("================================================================")

    if (numGames - numTraining - numGhostTraining) > 0:
        print("=======================================================")
        print("Statistic - Exploitation Phase")
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), winRate))
        print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))
        print("========================================================")

    return games


if __name__ == '__main__':
    """
    The main function called when warehouse.py is run
    from the command line:

    > python warehouse.py

    See the usage string for more details.

    > python warehouse.py --help
    """
    # args = readCommand(sys.argv[1:])  # Get game components based on input
    # runGames(**args)
    args = [
        '--layout', 'warehouse',
        '--pacman', 'ApproximateQAgent',
        '--withoutShield', '1',
        '--numGhostTraining', '0',
        '--numTraining', '10',  # Training episodes
        '--numGames', '11'  # Total episodes
    ]
    args = readCommand(args)
    runGames(**args)

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass
