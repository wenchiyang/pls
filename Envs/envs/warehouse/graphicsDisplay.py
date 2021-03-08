# graphicsDisplay.py
# ------------------
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


from game import Directions
from graphicsUtils import *


###########################
#  GRAPHICS DISPLAY CODE  #
###########################

# Most code by Dan Klein and John Denero written or rewritten for cs188, UC Berkeley.
# Some code from a Pacman implementation by LiveWires, and used / modified with permission.

DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = formatColor(222, 222, 222)
PACKAGE_COLOR = formatColor(91, 64, 48)
WALL_COLOR = formatColor(122, 122, 122)

GHOST_COLORS = [formatColor(.9, 0, 0),  # Red
                formatColor(0, .3, .9),  # Blue
                formatColor(.98, .41, .07),  # Orange
                formatColor(.1, .75, .7),  # Green
                formatColor(1.0, 0.6, 0.0),  # Yellow
                formatColor(.4, 0.13, 0.91)]  # Purple

TEAM_COLORS = GHOST_COLORS[:2]

TRUCK_FORK_SHAPE_NORTH = [(-0.6, 0.05), (-0.6, 0), (-0.4, 0), (-0.4, -0.75), (-0.2, -0.75), (-0.2, -0.05), (0.2, -0.05),
                          (0.2, -0.75), (0.4, -0.75), (0.4, 0), (0.6, 0), (0.6, 0.05)]

TRUCK_FORK_SHAPE_SOUTH = [(-0.6, -0.05), (-0.6, -0), (-0.4, -0), (-0.4, 0.75), (-0.2, 0.75), (-0.2, 0.05), (0.2, 0.05),
                          (0.2, 0.75), (0.4, 0.75), (0.4, -0), (0.6, -0), (0.6, -0.05)]

TRUCK_FORK_SHAPE_WEST = [(0.05, -0.6), (0, -0.6), (0, -0.4), (-0.75, -0.4), (-0.75, -0.2), (-0.05, -0.2), (-0.05, 0.2),
                         (-0.75, 0.2), (-0.75, 0.4), (0, 0.4), (0, 0.6), (0.05, 0.6)]

TRUCK_FORK_SHAPE_EAST = [(-0.05, -0.6), (-0, -0.6), (-0, -0.4), (0.75, -0.4), (0.75, -0.2), (0.05, -0.2), (0.05, 0.2),
                         (0.75, 0.2), (0.75, 0.4), (-0, 0.4), (-0, 0.6), (-0.05, 0.6)]

TRUCK_BODY_SHAPE_NORTH = [(-0.45, 0.75), (-0.45, 0.1), (0.45, 0.1), (0.45, 0.75)]

TRUCK_BODY_SHAPE_SOUTH = [(-0.45, -0.75), (-0.45, -0.1), (0.45, -0.1), (0.45, -0.75)]

TRUCK_BODY_SHAPE_EAST = [(-0.75, -0.45), (-0.1, -0.45), (-0.1, 0.45), (-0.75, 0.45)]

TRUCK_BODY_SHAPE_WEST = [(0.75, -0.45), (0.1, -0.45), (0.1, 0.45), (0.75, 0.45)]

TRUCK_ROOF_SHAPE_NORTH = [(-0.35, 0.45), (-0.35, 0.15), (0.35, 0.15), (0.35, 0.45)]

TRUCK_ROOF_SHAPE_SOUTH = [(-0.35, -0.45), (-0.35, -0.15), (0.35, -0.15), (0.35, -0.45)]

TRUCK_ROOF_SHAPE_WEST = [(0.45, -0.35), (0.15, -0.35), (0.15, 0.35), (0.45, 0.35)]

TRUCK_ROOF_SHAPE_EAST = [(-0.45, -0.35), (-0.15, -0.35), (-0.15, 0.35), (-0.45, 0.35)]

COLLISION_SHAPE = [(0, -0.75), (0.25, -0.25), (0.5, -0.5), (0.3, 0), (0.75, 0.5), (0.2, 0.4), (0.05, 0.75),
                   (-0.25, 0.55), (-0.75, 0.65), (-0.5, 0.25), (-0.75, 0.2), (-0.5, -0.1), (-0.65, -0.75),
                   (-0.2, -0.55)]

TRUCK_SIZE = 0.7

GHOST_VEC_COLORS = map(colorToVector, GHOST_COLORS)

PLAYER_COLOR = formatColor(255, 255, 61)


class InfoPane:
    def __init__(self, layout, gridSize):
        self.ghostDistanceText = []
        self.gridSize = gridSize
        self.width = (layout.width) * gridSize
        self.base = (layout.height + 1) * gridSize
        self.height = INFO_PANE_HEIGHT
        self.fontSize = 24
        self.textColor = formatColor(0, 0, 0)
        self.drawPane()

    def toScreen(self, pos, y=None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x, y = pos
        else:
            x = pos

        x = self.gridSize + x  # Margin
        y = self.base + y
        return x, y

    def drawPane(self):
        self.scoreText = text(self.toScreen(0, 0), self.textColor, "PRODUCTIVITY:    0", "Times", self.fontSize, "bold")

    def initialize_ghost_distances(self, distances):

        size = 20
        if self.width < 240:
            size = 12
        if self.width < 160:
            size = 10

        for i, d in enumerate(distances):
            t = text(self.toScreen(self.width / 2 + self.width / 8 * i, 0), GHOST_COLORS[i + 1], d, "Times", size,
                     "bold")
            self.ghostDistanceText.append(t)

    def update_score(self, score):
        changeText(self.scoreText, "PRODUCTIVITY: % 4d" % score)

    def updateGhostDistances(self, distances):
        if len(distances) == 0:
            return
        if 'ghostDistanceText' not in dir(self):
            self.initialize_ghost_distances(distances)
        else:
            for i, d in enumerate(distances):
                changeText(self.ghostDistanceText[i], d)


class PacmanGraphics:
    def __init__(self, zoom=1.0, frameTime=0.0, capture=False):
        self.have_window = 0
        self.currentGhostImages = {}
        self.pacmanImage = None
        self.zoom = zoom
        self.gridSize = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frameTime = frameTime

    def initialize(self, state, isBlue=False):
        self.isBlue = isBlue
        self.startGraphics(state)

        # self.drawDistributions(state)
        self.distributionImages = None  # Initialized lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state

    def startGraphics(self, state):
        self.layout = state.layout
        layout = self.layout
        self.width = layout.width
        self.height = layout.height
        self.make_window(self.width, self.height)
        self.infoPane = InfoPane(layout, self.gridSize)
        self.currentState = layout

    def drawDistributions(self, state):
        walls = state.layout.walls
        dist = []
        for x in range(walls.width):
            distx = []
            dist.append(distx)
            for y in range(walls.height):
                (screen_x, screen_y) = self.to_screen((x, y))
                block = square((screen_x, screen_y), 0.5 * self.gridSize, color=BACKGROUND_COLOR, filled=1, behind=2)
                distx.append(block)
        self.distributionImages = dist

    def drawStaticObjects(self, state):
        layout = self.layout
        self.drawWalls(layout.wallsGhost)
        for pos in layout.exitPos:
            x, y = self.to_screen(pos)
            self.drawExit(x, y)
        # set all background to black
        self.background = self.drawBackground(state.colorFields)
        self.packages = self.drawAllPackages(layout.packages)
        refresh()

    def drawAgentObjects(self, state):
        self.agentImages = []  # (agentState, image)
        for index, agent in enumerate(state.agentStates):
            if agent.isPacman:
                image = self.draw_fork_truck(agent, PLAYER_COLOR)
                self.agentImages.append((agent, image))
            else:
                image = self.draw_fork_truck(agent, GHOST_COLORS[index])
                self.agentImages.append((agent, image))
        refresh()

    def swapImages(self, agentIndex, newState):
        """
          Changes an image from a ghost to a pacman or vis versa (for capture)
        """
        prevState, prevImage = self.agentImages[agentIndex]
        for item in prevImage:
            remove_from_screen(item)
        if newState.isPacman:
            image = self.draw_fork_truck(newState, PLAYER_COLOR)
            self.agentImages[agentIndex] = (newState, image)
        else:
            image = self.draw_fork_truck(newState, GHOST_COLORS[agentIndex])
            self.agentImages[agentIndex] = (newState, image)
        refresh()

    def update(self, newState):

        agentIndex = newState._agentMoved
        agentState = newState.agentStates[agentIndex]

        # set all background to black
        for f in self.background:
            remove_from_screen(f)

        # draw colorFields
        self.background = self.drawBackground(newState.colorFields)

        if self.agentImages[agentIndex][0].isPacman != agentState.isPacman:
            self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        self.move_fork_truck(agentState, prevState, prevImage, newState._package_eaten, self.packages)
        self.agentImages[agentIndex] = (agentState, prevImage)

        if agentState.isPacman:
            frames = 4.0
            for i in range(1, int(frames) + 1):
                sleep(abs(self.frameTime) / frames)

        self.infoPane.update_score(newState.score)
        if 'ghostDistances' in dir(newState):
            self.infoPane.updateGhostDistances(newState.ghostDistances)

    def make_window(self, width, height):
        grid_width = (width - 1) * self.gridSize
        grid_height = (height - 1) * self.gridSize
        screen_width = 2 * self.gridSize + grid_width
        screen_height = 2 * self.gridSize + grid_height + INFO_PANE_HEIGHT

        begin_graphics(screen_width, screen_height, BACKGROUND_COLOR, "Warehouse")

    def draw_fork_truck(self, fork_truck, color):
        screen_x, screen_y = self.to_screen(self.getPosition(fork_truck))
        coords_fork = []
        coords_body = []
        coords_roof = []

        for (x, y) in TRUCK_ROOF_SHAPE_NORTH:
            coords_roof.append((x * self.gridSize * TRUCK_SIZE + screen_x, y * self.gridSize * TRUCK_SIZE + screen_y))
        for (x, y) in TRUCK_BODY_SHAPE_NORTH:
            coords_body.append((x * self.gridSize * TRUCK_SIZE + screen_x, y * self.gridSize * TRUCK_SIZE + screen_y))
        for (x, y) in TRUCK_FORK_SHAPE_NORTH:
            coords_fork.append((x * self.gridSize * TRUCK_SIZE + screen_x, y * self.gridSize * TRUCK_SIZE + screen_y))
        return [polygon(coords_fork, formatColor(0, 0, 0), smoothed=0), polygon(coords_body, color, smoothed=0),
                polygon(coords_roof, formatColor(0, 0, 0), filled=0, smoothed=0)]

    def move_truck_part(self, shape, screen_x, screen_y):
        coords = []
        for x, y in shape:
            coords.append(x * self.gridSize * TRUCK_SIZE + screen_x)
            coords.append(y * self.gridSize * TRUCK_SIZE + screen_y)
        return coords

    def move_package(self, screen_x, screen_y, fork_truck_image_parts, x_off, y_off):
        s = self.gridSize / 2
        top_left_x, top_left_y = ((-0.6 + x_off) * s + screen_x, (-0.6 + y_off) * s + screen_y)
        top_right_x, top_right_y = ((0.6 + x_off) * s + screen_x, (-0.6 + y_off) * s + screen_y)
        bottom_right_x, bottom_right_y = ((0.6 + x_off) * s + screen_x, (0.6 + y_off) * s + screen_y)
        bottom_left_x, bottom_left_y = ((-0.6 + x_off) * s + screen_x, (0.6 + y_off) * s + screen_y)
        new_shape(fork_truck_image_parts[3],
                  [top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x,
                   bottom_left_y])
        new_shape(fork_truck_image_parts[4], [top_left_x, top_left_y, bottom_right_x, bottom_right_y])
        new_shape(fork_truck_image_parts[5], [top_right_x, top_right_y, bottom_left_x, bottom_left_y])

    def move_fork_truck(self, fork_truck, prev_fork_truck, fork_truck_image_parts, food_eaten, packages):
        new_x, new_y = self.to_screen(self.getPosition(fork_truck))
        old_x, old_y = self.to_screen(self.getPosition(prev_fork_truck))
        delta = old_x - new_x, old_y - new_y

        direction = self.getDirection(fork_truck)
        coords_fork = []
        coords_body = []
        coords_roof = []
        x, y = 0, 0

        if direction == 'Stop':
            return
        if direction == 'North':
            coords_roof = self.move_truck_part(TRUCK_ROOF_SHAPE_NORTH, new_x, new_y)
            coords_body = self.move_truck_part(TRUCK_BODY_SHAPE_NORTH, new_x, new_y)
            coords_fork = self.move_truck_part(TRUCK_FORK_SHAPE_NORTH, new_x, new_y)
            y = -0.7
        if direction == 'South':
            coords_roof = self.move_truck_part(TRUCK_ROOF_SHAPE_SOUTH, new_x, new_y)
            coords_body = self.move_truck_part(TRUCK_BODY_SHAPE_SOUTH, new_x, new_y)
            coords_fork = self.move_truck_part(TRUCK_FORK_SHAPE_SOUTH, new_x, new_y)
            y = 0.7
        if direction == 'East':
            coords_roof = self.move_truck_part(TRUCK_ROOF_SHAPE_EAST, new_x, new_y)
            coords_body = self.move_truck_part(TRUCK_BODY_SHAPE_EAST, new_x, new_y)
            coords_fork = self.move_truck_part(TRUCK_FORK_SHAPE_EAST, new_x, new_y)
            x = 0.7
        if direction == 'West':
            coords_roof = self.move_truck_part(TRUCK_ROOF_SHAPE_WEST, new_x, new_y)
            coords_body = self.move_truck_part(TRUCK_BODY_SHAPE_WEST, new_x, new_y)
            coords_fork = self.move_truck_part(TRUCK_FORK_SHAPE_WEST, new_x, new_y)
            x = -0.7

        new_shape(fork_truck_image_parts[0], coords_fork)
        new_shape(fork_truck_image_parts[1], coords_body)
        new_shape(fork_truck_image_parts[2], coords_roof)

        if fork_truck._loaded and prev_fork_truck._loaded:
            self.move_package(new_x, new_y, fork_truck_image_parts, x, y)

        if not fork_truck._loaded and prev_fork_truck._loaded:
            remove_from_screen(fork_truck_image_parts[5])
            remove_from_screen(fork_truck_image_parts[4])
            remove_from_screen(fork_truck_image_parts[3])
            fork_truck_image_parts.pop()
            fork_truck_image_parts.pop()
            fork_truck_image_parts.pop()

        refresh()

        if food_eaten is not None:
            self.remove_package(food_eaten, packages)
            fork_truck_image_parts.extend(self.drawPackage(new_x, new_y, x, y))

        refresh()

    def getPosition(self, agentState):
        if agentState.configuration == None:
            return (-1000, -1000)
        return agentState.getPosition()

    def getDirection(self, agentState):
        if agentState.configuration == None:
            return Directions.STOP
        return agentState.configuration.getDirection()

    def finish(self):
        end_graphics()

    def to_screen(self, point):
        x, y = point
        x = (x + 1) * self.gridSize
        y = (self.height - y) * self.gridSize
        return x, y

    def drawExit(self, x, y):
        color = formatColor(245, 224, 65)
        s = self.gridSize / 2
        square((x, y), s * 0.8, formatColor(0, 0, 0))
        line((-0.8 * s + x, -0.8 * s + y), (0.8 * s + x, 0.8 * s + y), color, 3)
        line((-0.8 * s + x, -0.2 * s + y), (0.2 * s + x, 0.8 * s + y), color, 3)
        line((-0.2 * s + x, -0.8 * s + y), (0.8 * s + x, 0.2 * s + y), color, 3)
        line((-0.8 * s + x, 0.4 * s + y), (-0.4 * s + x, 0.8 * s + y), color, 3)
        line((0.4 * s + x, -0.8 * s + y), (0.8 * s + x, -0.4 * s + y), color, 3)
        square((x, y), s * 0.8, color, filled=0)

    def draw_vertical_shelf(self, x, y):
        s = self.gridSize / 2

        coord = [(-0.5 * s + x, -0.9 * s + y), (0.5 * s + x, -0.9 * s + y), (0.5 * s + x, 0.9 * s + y),
                 (-0.5 * s + x, 0.9 * s + y)]
        polygon(coord, WALL_COLOR, smoothed=0)
        circle((-0.4 * s + x, -0.8 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((0.6 * s + x, -0.8 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((0.6 * s + x, 1.0 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((-0.4 * s + x, 1.0 * s + y), 0.1, WALL_COLOR, WALL_COLOR)

    def draw_horizontal_shelf(self, x, y):
        s = self.gridSize / 2

        coord = [(-0.9 * s + x, -0.5 * s + y), (-0.9 * s + x, 0.5 * s + y), (0.9 * s + x, 0.5 * s + y),
                 (0.9 * s + x, -0.5 * s + y)]
        polygon(coord, WALL_COLOR, smoothed=0)
        circle((-0.8 * s + x, -0.4 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((-0.8 * s + x, 0.6 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((1.0 * s + x, 0.6 * s + y), 0.1, WALL_COLOR, WALL_COLOR)
        circle((1.0 * s + x, -0.4 * s + y), 0.1, WALL_COLOR, WALL_COLOR)

    def drawWalls(self, wallMatrix):
        for xNum, x in enumerate(wallMatrix):
            for yNum, cell in enumerate(x):
                if cell:  # There's a wall here
                    (pos_x, pos_y) = self.to_screen((xNum, yNum))

                    # draw each quadrant of the square based on adjacent walls
                    wIsWall = self.isWall(xNum - 1, yNum, wallMatrix)
                    eIsWall = self.isWall(xNum + 1, yNum, wallMatrix)
                    nIsWall = self.isWall(xNum, yNum + 1, wallMatrix)
                    sIsWall = self.isWall(xNum, yNum - 1, wallMatrix)

                    if eIsWall or wIsWall:
                        self.draw_horizontal_shelf(pos_x, pos_y)
                    if nIsWall or sIsWall:
                        self.draw_vertical_shelf(pos_x, pos_y)

    def isWall(self, x, y, walls):
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def drawBackground(self, colorFields):
        colorImages = []
        for field in colorFields:
            color = formatColor(0.0, 0.0, 0.0)

            if field['color'] == "RED":
                color = formatColor(0.7, 0.0, 0.0)
            if field['color'] == "ORANGERED":
                color = formatColor(1, 0.3, 0.0)
            if field['color'] == "ORANGE":
                color = formatColor(1, 0.64, 0.0)
            if field['color'] == "GREEN":
                color = formatColor(0.0, 0.7, 0.0)
            if field['color'] == "BLUE":
                color = formatColor(0.0, 0.0, 0.7)
            if field['color'] == "GOLD":
                color = formatColor(1, 0.64, 0.0)
            if field['color'] == "YELLOW":
                color = formatColor(1, 1, 0.0)

            imageRow = []
            colorImages.append(imageRow)
            screen = self.to_screen(field['coordinate'])
            block = square(screen, 0.5 * self.gridSize, color=color, filled=1, behind=4)
            imageRow.append(block)
        return colorImages

    def drawAllPackages(self, packageMatrix):
        packageImages = []
        for xNum, x in enumerate(packageMatrix):
            imageRow = []
            packageImages.append(imageRow)
            for yNum, cell in enumerate(x):
                if cell:  # There's a package here
                    (screen_x, screen_y) = self.to_screen((xNum, yNum))
                    images = self.drawPackage(screen_x, screen_y)
                    imageRow.append(images)
                else:
                    imageRow.append(None)
        return packageImages

    def drawPackage(self, screen_x, screen_y, x_off=0, y_off=0):
        s = self.gridSize / 2
        images = []
        top_left = ((-0.6 + x_off) * s + screen_x, (-0.6 + y_off) * s + screen_y)
        top_right = ((0.6 + x_off) * s + screen_x, (-0.6 + y_off) * s + screen_y)
        bottom_right = ((0.6 + x_off) * s + screen_x, (0.6 + y_off) * s + screen_y)
        bottom_left = ((-0.6 + x_off) * s + screen_x, (0.6 + y_off) * s + screen_y)
        p1 = polygon([top_left, top_right, bottom_right, bottom_left], PACKAGE_COLOR, smoothed=0)
        l1 = line(top_left, bottom_right, width=1)
        l2 = line(top_right, bottom_left, width=1)
        images.append(p1)
        images.append(l1)
        images.append(l2)
        return images

    def remove_package(self, cell, package_images):
        x, y = cell
        self.layout.walls[x][y] = True
        images = package_images[x][y]
        for img in images:
            remove_from_screen(img)

    def drawExpandedCells(self, cells):
        """
        Draws an overlay of expanded grid positions for search agents
        """
        n = float(len(cells))
        baseColor = [1.0, 0.0, 0.0]
        self.clearExpandedCells()
        self.expandedCells = []
        for k, cell in enumerate(cells):
            screenPos = self.to_screen(cell)
            cellColor = formatColor(*[(n - k) * c * .5 / n + .25 for c in baseColor])
            block = square(screenPos, 0.5 * self.gridSize, color=cellColor, filled=1, behind=2)
            self.expandedCells.append(block)
            if self.frameTime < 0:
                refresh()

    def clearExpandedCells(self):
        if 'expandedCells' in dir(self) and len(self.expandedCells) > 0:
            for cell in self.expandedCells:
                remove_from_screen(cell)

    def updateDistributions(self, distributions):
        "Draws an agent's belief distributions"
        # copy all distributions so we don't change their state
        distributions = [x.copy() for x in distributions]
        if self.distributionImages == None:
            self.drawDistributions(self.previousState)
        for x in range(len(self.distributionImages)):
            for y in range(len(self.distributionImages[0])):
                image = self.distributionImages[x][y]
                weights = [dist[(x, y)] for dist in distributions]

                if sum(weights) != 0:
                    pass
                # Fog of war
                color = [0.0, 0.0, 0.0]
                colors = GHOST_VEC_COLORS[1:]  # With Pacman
                if self.capture:
                    colors = GHOST_VEC_COLORS
                for weight, gcolor in zip(weights, colors):
                    color = [min(1.0, c + 0.95 * g * weight ** .3) for c, g in zip(color, gcolor)]
                changeColor(image, formatColor(*color))
        refresh()


class FirstPersonPacmanGraphics(PacmanGraphics):
    def __init__(self, zoom=1.0, showGhosts=True, capture=False, frameTime=0):
        PacmanGraphics.__init__(self, zoom, frameTime=frameTime)
        self.showGhosts = showGhosts
        self.capture = capture

    def initialize(self, state, isBlue=False):

        self.isBlue = isBlue
        PacmanGraphics.startGraphics(self, state)
        # Initialize distribution images
        walls = state.layout.walls
        dist = []
        self.layout = state.layout

        # Draw the rest
        self.distributionImages = None  # initialize lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state


    def getPosition(self, ghostState):
        if not self.showGhosts and not ghostState.isPacman and ghostState.getPosition()[1] > 1:
            return -1000, -1000
        else:
            return PacmanGraphics.getPosition(self, ghostState)


# Saving graphical output
# -----------------------
# Note: to make an animated gif from this postscript output, try the command:
# convert -delay 7 -loop 1 -compress lzw -layers optimize frame* out.gif
# convert is part of imagemagick (freeware)

SAVE_POSTSCRIPT = False
POSTSCRIPT_OUTPUT_DIR = 'frames'
FRAME_NUMBER = 0
import os


def saveFrame():
    "Saves the current graphical output as a postscript file"
    global SAVE_POSTSCRIPT, FRAME_NUMBER, POSTSCRIPT_OUTPUT_DIR
    if not SAVE_POSTSCRIPT:
        return
    if not os.path.exists(POSTSCRIPT_OUTPUT_DIR):
        os.mkdir(POSTSCRIPT_OUTPUT_DIR)
    name = os.path.join(POSTSCRIPT_OUTPUT_DIR, 'frame_%08d.ps' % FRAME_NUMBER)
    FRAME_NUMBER += 1
    writePostscript(name)  # writes the current canvas
