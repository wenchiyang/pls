# layout.py
# ---------
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


import os
import random
from functools import reduce

from game import Grid
from util import manhattanDistance


VISIBILITY_MATRIX_CACHE = {}


class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, layoutText):
        self.width = len(layoutText[0])
        self.height = len(layoutText)
        self.walls = Grid(self.width, self.height, False)
        self.wallsGhost = Grid(self.width, self.height, False)
        self.packages = Grid(self.width, self.height, False)
        self.exitPos = []
        self.agentPositions = []
        self.endPointPositions = []
        self.numGhosts = 0
        self.processLayoutText(layoutText)
        self.layoutText = layoutText
        self.totalPackages = len(self.packages.asList())
        self.paths = []

        # self.initializeVisibilityMatrix()

    def getNumGhosts(self):
        return self.numGhosts

    def initializeVisibilityMatrix(self):
        global VISIBILITY_MATRIX_CACHE
        if reduce(str.__add__, self.layoutText) not in VISIBILITY_MATRIX_CACHE:
            from game import Directions
            vecs = [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]
            dirs = [Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
            vis = Grid(self.width, self.height,
                       {Directions.NORTH: set(), Directions.SOUTH: set(), Directions.EAST: set(),
                        Directions.WEST: set(), Directions.STOP: set()})
            for x in range(self.width):
                for y in range(self.height):
                    if self.walls[x][y] == False:
                        for vec, direction in zip(vecs, dirs):
                            dx, dy = vec
                            nextx, nexty = x + dx, y + dy
                            while (nextx + nexty) != int(nextx) + int(nexty) or not self.walls[int(nextx)][int(nexty)]:
                                vis[x][y][direction].add((nextx, nexty))
                                nextx, nexty = x + dx, y + dy
            self.visibility = vis
            VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)] = vis
        else:
            self.visibility = VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)]

    def isWall(self, pos):
        x, col = pos
        return self.walls[x][col]

    def getRandomLegalPosition(self):
        x = random.choice(list(range(self.width)))
        y = random.choice(list(range(self.height)))
        while self.isWall((x, y)):
            x = random.choice(list(range(self.width)))
            y = random.choice(list(range(self.height)))
        return (x, y)

    def getRandomCorner(self):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, pacPos):
        poses = [(1, 1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
        return pos

    def isVisibleFrom(self, ghostPos, pacPos, pacDirection):
        row, col = [int(x) for x in pacPos]
        return ghostPos in self.visibility[row][col][pacDirection]

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        return Layout(self.layoutText[:])

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Package
         G - Ghost
         P - Pacman
         A - Position of exit only a single one working!
        Other characters are ignored.
        """
        maxY = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                self.processLayoutChar(x, y, layoutChar)
        self.agentPositions.sort()
        self.agentPositions = [(i == 0, pos) for i, pos in self.agentPositions]
        self.endPointPositions.sort()
        self.endPointPositions = [(i == 0, pos) for i, pos in self.endPointPositions]

        if len(self.exitPos) == 0:
            raise Exception('There must be an exit')

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '%':
            self.walls[x][y] = True
            self.wallsGhost[x][y] = True
        elif layoutChar == 'o':
            self.wallsGhost[x][y] = True
            self.packages[x][y] = True
        elif layoutChar == 'P':
            self.agentPositions.append((0, (x, y)))
        elif layoutChar == 'A':
            self.exitPos.append((x, y))
        elif layoutChar in ['G']:
            self.agentPositions.append((1, (x, y)))
            self.numGhosts += 1
        elif layoutChar in ['1', '2', '3', '4']:
            self.agentPositions.append((int(layoutChar), (x, y)))
            self.numGhosts += 1
        elif layoutChar in ['a', 'b', 'c', 'd']:
            self.endPointPositions.append((int(['a', 'b', 'c', 'd'].index(layoutChar) + 1), (x, y)))

    def getPaths(self):
        if len(self.endPointPositions) != self.numGhosts:
            raise Exception('Every Ghost needs an Endpoint')

        for i in range(self.numGhosts):
            offset = len(self.agentPositions) - len(self.endPointPositions)
            _, pos_start = self.agentPositions[i + offset]
            _, pos_end = self.endPointPositions[i]
            x_start, y_start = pos_start
            x_end, y_end = pos_end

            if x_start < x_end:
                x_left = x_start
                x_right = x_end
            else:
                x_left = x_end
                x_right = x_start

            if y_start < y_end:
                y_bottom = y_start
                y_top = y_end
            else:
                y_bottom = y_end
                y_top = y_start

            all_paths = []
            self.calculatePath(all_paths, x_start, y_start, [], pos_end, (x_left, x_right, y_top, y_bottom))
            self.paths.append(all_paths)

    def calculatePath(self, all_paths, x, y, path, end, boundaries):
        path.append((x, y))
        if end == (x, y):
            all_paths.append(path)
            return

        left, right, top, bottom = boundaries
        l, r, t, b = False, False, False, False
        count = 0

        if y < self.height and y < top + 1 and not self.wallsGhost[x][y + 1]:
            count += 1
            t = True
        if x < self.width and x < right + 1 and not self.wallsGhost[x + 1][y]:
            count += 1
            r = True
        if y > 0 and y > bottom - 1 and not self.wallsGhost[x][y - 1]:
            count += 1
            b = True
        if x > 0 and x > left - 1 and not self.wallsGhost[x - 1][y]:
            count += 1
            l = True

        if count >= 3 and ((t and path.count((x, y + 1)) == 1) or (b and path.count((x, y - 1)) == 1) or len(path) == 1):
            if t and b and l and not self.wallsGhost[x - 1][y + 1] and not self.wallsGhost[x - 1][y - 1]:
                l = False
            if t and b and r and not self.wallsGhost[x + 1][y + 1] and not self.wallsGhost[x + 1][y - 1]:
                r = False
        elif count >= 3 and ((l and path.count((x - 1, y)) == 1) or (r and path.count((x + 1, y)) == 1) or len(path) == 1):
            if l and r and t and not self.wallsGhost[x - 1][y + 1] and not self.wallsGhost[x + 1][y + 1]:
                t = False
            if l and r and b and not self.wallsGhost[x - 1][y - 1] and not self.wallsGhost[x + 1][y - 1]:
                b = False

        if t and y < top and not self.walls[x][y + 1] and path.count((x, y + 1)) == 0:
            self.calculatePath(all_paths, x, y + 1, path.copy(), end, boundaries)
        if r and x < right and not self.walls[x + 1][y] and path.count((x + 1, y)) == 0:
            self.calculatePath(all_paths, x + 1, y, path.copy(), end, boundaries)
        if b and y > bottom and not self.walls[x][y - 1] and path.count((x, y - 1)) == 0:
            self.calculatePath(all_paths, x, y - 1, path.copy(), end, boundaries)
        if l and x > left and not self.walls[x - 1][y] and path.count((x - 1, y)) == 0:
            self.calculatePath(all_paths, x - 1, y, path.copy(), end, boundaries)


def getLayout(name, back=2):
    if name.endswith('.lay'):
        layout = tryToLoad('layouts/' + name)
        if layout == None:
            layout = tryToLoad(name)
    else:
        layout = tryToLoad('layouts/' + name + '.lay')
        if layout == None:
            layout = tryToLoad(name + '.lay')
    if layout == None and back >= 0:
        curdir = os.path.abspath('.')
        os.chdir('..')
        layout = getLayout(name, back - 1)
        os.chdir(curdir)
    return layout


def tryToLoad(fullname):
    if (not os.path.exists(fullname)):
        return None
    f = open(fullname)

    try:
        return Layout([line.strip() for line in f])
    finally:
        f.close()
