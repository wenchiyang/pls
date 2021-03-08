# adversaryAgents.py
# --------------
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


import util
from game import Actions
from game import Agent
from game import Directions
from util import manhattanDistance


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.5, prob_scaredFlee=0.8):  # , prob_attack=0.6, before: prob_attack=0.8
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()

        return dist


class ForkTruckPath(GhostAgent):
    # here you can change the probabilities of the paths
    probability = {1: util.Counter({0: 1.0}), 2: util.Counter({0: 0.8, 1: 0.2}),
                   3: util.Counter({0: 0.7, 1: 0.2, 2: 0.1}), 4: util.Counter({0: 0.6, 1: 0.2, 2: 0.1, 3: 0.1}),
                   5: util.Counter({0: 0.6, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.05})}

    def __init__(self, index, allPaths):
        self.index = index
        self.paths = allPaths[index - 1]
        self.paths.sort(key=len)
        self.paths = self.paths[:5]
        self.currentPathIndex = 0
        if (len(self.paths) == 0):
            print("Fork Truck ", index, " has no possible paths!!!!")
            return
        self.start = self.paths[0][0]
        self.end = self.paths[0][len(self.paths[0]) - 1]
        self.goToEnd = False

    def getDistribution(self, state):
        dist = util.Counter()

        x, y = state.getGhostState(self.index).configuration.pos
        forkTruckPosition = (int(x), int(y))

        if forkTruckPosition == self.start:
            self.currentPathIndex = util.chooseFromDistribution(self.probability[len(self.paths)])
            self.goToEnd = True

        if forkTruckPosition == self.end:
            self.currentPathIndex = util.chooseFromDistribution(self.probability[len(self.paths)])
            self.goToEnd = False

        currentPath = self.paths[self.currentPathIndex]
        currentIndex = currentPath.index(forkTruckPosition)

        if self.goToEnd:
            nextPosition = currentPath[currentIndex + 1]
        else:
            nextPosition = currentPath[currentIndex - 1]

        x_cur, y_cur = forkTruckPosition
        x_next, y_next = nextPosition

        if x_next < x_cur:
            dist[Directions.WEST] = 1.0
        elif x_next > x_cur:
            dist[Directions.EAST] = 1.0
        elif y_next < y_cur:
            dist[Directions.SOUTH] = 1.0
        elif y_next > y_cur:
            dist[Directions.NORTH] = 1.0
        else:
            dist[Directions.STOP] = 1.0

        dist.normalize()
        return dist
