import stormpy
import stormpy.core
import os
import numpy
import time
from multiprocessing.dummy import Pool as ThreadPool
from .util import ShortestPath
import tempfile

import multiprocessing

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
TOP_RIGHT = 4
TOP_LEFT = 5
BOTTOM_RIGHT = 6
BOTTOM_LEFT = 7
STOP = 8

STR_RIGHT = "r"
STR_UP = "u"
STR_LEFT = "l"
STR_DOWN = "d"
STR_STOP = "s"
STR_TOP_RIGHT = "u-r"
STR_TOP_LEFT = "u-l"
STR_BOTTOM_RIGHT = "d-r"
STR_BOTTOM_LEFT = "d-l"

STEPS = 10  # number of steps we plan ahead
STEPS_IN_ENCODING = True

USE_CORRIDOR_ENCODING = True


class StormEncoder:

    def __init__(self, state, symX, symY):
        # get board propertiesin
        self.symX = symX
        self.symY = symY
        self.walls = state.getWalls()
        self.w = state.data.layout.width
        self.h = state.data.layout.height
        # find crossings and horizontal and vertical corridors on board
        self.crossings, self.hcorr, self.vcorr, self.deadend = self.mapBoard()
        self.corners = self.mapCorners()

        # split corridors at crossings in half
        self.hcorr = self.splitHorizontal(self.hcorr, self.crossings)
        self.vcorr = self.splitVertical(self.vcorr, self.crossings)

        self.connected_corrs = self.computeConnectingCorridors()
        self.relevant_crossings = self.getRelevantCrossings()
        self.pacman = state.getPacmanPosition()
        self.ghosts = state.getGhostPositions()
        #self.computePathCounter()


    #def computePathCounter(self):
    #    for init_pacman in self.relevant_crossings:
    #        for next_dir_pacman in range(0, 4):
    #            next_pos_pacman = self.getNextPosition(init_pacman, next_dir_pacman)
    #            if not self.isWall(next_pos_pacman):
    #                self.computePaths(init_pacman, next_dir_pacman)


    def getRelevantCrossings(self):

        if not self.symY and not self.symX:
            return self.crossings

        relevant_crossings = []

        max_x = self.w - 2
        max_y = self.h - 2

        if self.symX:
            assert (max_x % 2 == 1)
            max_x = (max_x // 2) + 1

        if self.symY:
            assert (max_y % 2 == 1)
            max_y = (max_y // 2) + 1

        for crossing in self.crossings:
            if crossing[0] <= max_x and crossing[1] <= max_y:
                relevant_crossings.append(crossing)

        return relevant_crossings

    def vertCorr(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pl) \
               and self.isWall(pr) \
               and not self.isWall(pu) \
               and not self.isWall(pd)

    def horzCorr(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and self.isWall(pd) \
               and not self.isWall(pl) \
               and not self.isWall(pr)

    def cornerLeftBottom(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def cornerRightBottom(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def cornerLeftTop(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def cornerRightTop(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def deadEndUp(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and self.isWall(pl) \
               and self.isWall(pr)

    def deadEndDown(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and self.isWall(pl) \
               and self.isWall(pr)

    def deadEndLeft(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def deadEndRight(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def leftUpperCorner(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def rightUpperCorner(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def rightBottomCorner(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def leftBottomCorner(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def centerCrossing(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and not self.isWall(pd) \
               and not self.isWall(pl) \
               and not self.isWall(pr)

    def tUpCrossing(self, n):
        [pu, pd, pl, pr] = n
        return self.isWall(pu) \
               and not self.isWall(pd) \
               and not self.isWall(pl) \
               and not self.isWall(pr)

    def tDownCrossing(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and self.isWall(pd) \
               and not self.isWall(pl) \
               and not self.isWall(pr)

    def tLeftCrossing(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and not self.isWall(pd) \
               and not self.isWall(pl) \
               and self.isWall(pr)

    def tRightCrossing(self, n):
        [pu, pd, pl, pr] = n
        return not self.isWall(pu) \
               and not self.isWall(pd) \
               and self.isWall(pl) \
               and not self.isWall(pr)

    def isCrossing(self, x, y):
        p = (x, y)
        nh = self.neighborHood(p)

        if (x < 1 or y < 1 or x > self.w - 1 or y > self.h - 1):
            return False

        return (self.centerCrossing(nh) \
                or self.tDownCrossing(nh) \
                or self.tUpCrossing(nh) \
                or self.tLeftCrossing(nh) \
                or self.tRightCrossing(nh))

    def isCorner(self, x, y):
        for c in self.corners:
            if (x, y) == c[0]:
                return True
        return False

    def isDeadend(self, x, y):
        for d in self.deadend:
            if (x, y) == d[0]:
                return True
        return False

    def neighborHood(self, p):
        x = p[0]
        y = p[1]
        pd = (x, y - 1)
        pu = (x, y + 1)
        pl = (x - 1, y)
        pr = (x + 1, y)
        return [pd, pu, pl, pr]

    def intersects(self, corridor, x):
        return corridor[0][0] <= x[0] <= corridor[1][0] and corridor[0][1] <= x[1] <= corridor[1][1]

    def isWall(self, p):

        if p[0] <= 0 or p[0] >= self.w - 1:
            return True
        if p[1] <= 0 or p[1] >= self.h - 1:
            return True

        return self.walls[p[0]][self.h - p[1] - 1]

    def isInsideWall(self, p):
        return self.walls[p[0]][self.h - p[1] - 1] and 0 <= p[0] <= self.w and p[1] >= 0 and p[1] <= self.h

    def mapCrossingsAndHorizontal(self):
        crossings = []
        hcorr = []
        startH = None

        for y in range(1, self.h):
            for x in range(1, self.w):
                p = (x, y)

                # is crossing
                nh = self.neighborHood(p)
                if not self.isInsideWall(p) and \
                        (self.centerCrossing(nh) \
                         or self.tDownCrossing(nh) \
                         or self.tUpCrossing(nh) \
                         or self.tLeftCrossing(nh) \
                         or self.tRightCrossing(nh)):
                    crossings.append(p)

                # horizontal
                if not self.isWall(p) and not startH:  # corridor starts
                    startH = p
                elif self.isWall(p) and startH:  # corridor ends
                    if p[0] - startH[0] > 1:  # check if not vertical
                        endH = tuple(numpy.subtract(p, (1, 0)))
                        hcorr.append((startH, endH))
                    startH = None
        return [crossings, hcorr]

    def mapDeadEndAndVertical(self):
        vcorr = []
        deadend = []
        startV = None

        # classify all positions to find crossings and horizontal
        for x in range(1, self.w):
            for y in range(1, self.h):
                p = (x, y)
                n = self.neighborHood(p)

                # deadend
                if not self.isInsideWall(p):
                    if self.deadEndRight(n):
                        deadend.append((p, 0))
                    if self.deadEndLeft(n):
                        deadend.append((p, 2))
                    if self.deadEndUp(n):
                        deadend.append((p, 1))
                    if self.deadEndDown(n):
                        deadend.append((p, 3))

                # vertical
                if not self.isWall(p) and not startV:  # corridor starts
                    startV = p
                elif self.isWall(p) and startV:  # corridor ends
                    if p[1] - startV[1] > 1:  # check if not horizontal
                        endV = tuple(numpy.subtract(p, (0, 1)))
                        vcorr.append((startV, endV))
                    startV = None

        return deadend, vcorr

    def mapCorners(self):

        corners = []

        for x in range(1, self.w):
            for y in range(1, self.h):
                p = (x, y)
                n = self.neighborHood(p)

                if not self.isWall(p):

                    if self.cornerRightTop(n):
                        corners.append((p, 0))
                    if self.cornerRightBottom(n):
                        corners.append((p, 1))
                    if self.cornerLeftTop(n):
                        corners.append((p, 2))
                    if self.cornerLeftBottom(n):
                        corners.append((p, 3))
        return corners

    def mapBoard(self):
        crossings, hcorr = self.mapCrossingsAndHorizontal()
        deadend, vcorr = self.mapDeadEndAndVertical()
        return [crossings, hcorr, vcorr, deadend]

    def splitHorizontal(self, hcorr, crossings):
        intersected = True
        while intersected:
            intersected = False
            for crs in crossings:
                tmph = []
                for ch in hcorr:
                    if ch[0][1] == crs[1] and not intersected:  # is on x axis, go and split
                        if self.intersects(ch, crs):  # check if this corr intersects
                            intersected = True
                            leftPoint = tuple(numpy.subtract(crs, (1, 0)))
                            rightPoint = tuple(numpy.add(crs, (1, 0)))

                            if leftPoint >= ch[0]:
                                c1 = (ch[0], leftPoint)
                                tmph.append(c1)

                            if rightPoint <= ch[1]:
                                c2 = (rightPoint, ch[1])
                                tmph.append(c2)

                        else:
                            tmph.append(ch)
                    else:  # just leave this corridor
                        tmph.append(ch)
                hcorr = tmph.copy()
        return hcorr

    def splitVertical(self, vcorr, crossings):
        intersected = True
        while intersected:
            intersected = False
            # get all corridors on y axis of crossing
            for crs in crossings:
                tmpv = []
                for cv in vcorr:
                    if cv[0][0] == crs[0] and not intersected:
                        if self.intersects(cv, crs):
                            intersected = True
                            upperPoint = tuple(numpy.subtract(crs, (0, 1)))
                            lowerPoint = tuple(numpy.add(crs, (0, 1)))

                            if upperPoint >= cv[0]:
                                c1 = (cv[0], upperPoint)
                                tmpv.append(c1)

                            if lowerPoint <= cv[1]:
                                c2 = (lowerPoint, cv[1])
                                tmpv.append(c2)
                        else:
                            tmpv.append(cv)
                    else:  # just leave this corridor
                        tmpv.append(cv)
                vcorr = tmpv.copy()
        return vcorr

    def encodePositioningTerm(self, pos):

        positioning_str = ""
        assert (pos >= 0 and pos < 8)

        if pos == RIGHT:
            positioning_str = "xG<xP & yG=yP "
        if pos == UP:
            positioning_str = "yG<yP & xG=xP "
        if pos == LEFT:
            positioning_str = "xG>xP & yG=yP "
        if pos == DOWN:
            positioning_str = "yG>yP & xG=xP "

        if pos == TOP_RIGHT:
            positioning_str = "xG<xP & yG<yP "
        if pos == TOP_LEFT:
            positioning_str = "xG>xP & yG<yP "
        if pos == BOTTOM_RIGHT:
            positioning_str = "xG<xP & yG>yP "
        if pos == BOTTOM_LEFT:
            positioning_str = "xG>xP & yG>yP "

        return positioning_str

    # union find algorithm used to compute the connection corridors
    def add(self, a, b):
        leadera = self.leaders.get(a)
        leaderb = self.leaders.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return  # nothing to do
                groupa = self.groups[leadera]
                groupb = self.groups[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.groups[leaderb]
                for k in groupb:
                    self.leaders[k] = leadera
            else:
                self.groups[leadera].add(b)
                self.leaders[b] = leadera
        else:
            if leaderb is not None:
                self.groups[leaderb].add(a)
                self.leaders[a] = leaderb
            else:
                self.leaders[a] = self.leaders[b] = a
                self.groups[a] = set([a, b])

    def computeConnectingCorridors(self):

        self.leaders = {}  # maps a member to the group's leader
        self.groups = {}  # maps a group leader to the group (which is a set)

        for vcorr in self.vcorr:
            self.add(vcorr[0], vcorr[1])
        for hcorr in self.hcorr:
            self.add(hcorr[0], hcorr[1])

        connecting_corridors = []

        for leader in self.groups:
            new_group = []
            for pos in self.groups[leader]:
                new_group = new_group + self.getCorridorIDAtPos(pos[0], pos[1])
            connecting_corridors.append(list(set(new_group)))

        self.connecting_corridors_via_pos = self.computeConnectingCorridorsViaPos()

        return connecting_corridors

    def computeConnectingCorridorsViaPos(self):

        connecting_corridors_via_pos = []

        for leader in self.groups:
            new_group = []
            for pos in self.groups[leader]:
                # get corridor that includes pos:
                for corr in self.vcorr + self.hcorr:
                    if pos in corr:
                        if not corr in new_group:
                            new_group.append(corr)
            connecting_corridors_via_pos.append(new_group)
        return connecting_corridors_via_pos

    def getConnectingCorridorsViaPos(self, pos):

        corr_id_pos = self.getCorridorIDAtPos(pos[0], pos[1])

        for connected_corrs in self.connecting_corridors_via_pos:
            for connceted_corr in connected_corrs:
                if list(set(corr_id_pos) & set(self.getCorridorIDAtPos(connceted_corr[0][0], connceted_corr[0][1]))):
                    return connected_corrs
        return []

    def isSameConnectingCorridor(self, pos1, pos2):

        corr_id1 = self.getCorridorIDAtPos(pos1[0], pos1[1])
        corr_id2 = self.getCorridorIDAtPos(pos2[0], pos2[1])

        for connecting_corr in self.connected_corrs:
            if list(set(connecting_corr) & set(corr_id1)):
                if list(set(connecting_corr) & set(corr_id2)):
                    return True
        return False

    def encodeCrossingStatementStop(self, x_ghost, y_ghost):

        prismStr = "  [g] (xG=" + str(x_ghost) + " & yG=" + str(y_ghost)
        prismStr += " &  xG=xP & yG=yP)"

        prismStr += " -> "
        prismStr += "1: (xG'=xG);\n"

        return prismStr


    def getProbabilitiesFromGhostTable(self, crossing_id_ghost, ghost_dir, positioning_pacman, allowed_next_ghost_dirs):

        probabilities = [0, 0, 0, 0]

        for entry in self.ghost_table:
            if entry[0] == crossing_id_ghost and entry[1] == ghost_dir and entry[2] == positioning_pacman:
                if entry[3] == RIGHT:
                    probabilities[0] = entry[4]
                if entry[3] == UP:
                    probabilities[1] = entry[4]
                if entry[3] == LEFT:
                    probabilities[2] = entry[4]
                if entry[3] == DOWN:
                    probabilities[3] = entry[4]

        # str1 = " crossing_id_ghost "+ str(crossing_id_ghost) + " ghost_dir " + str(ghost_dir) + " positioning_pacman "+ str(positioning_pacman)
        # print(str1)
        # print(allowed_next_ghost_dirs)
        # print(probabilities)

        # check if there are entries in the ghost table that are not allowed
        num_allowed_direction = 0
        j = 0
        for allowed_dir in allowed_next_ghost_dirs:
            if not allowed_dir:
                assert (probabilities[j] == 0)
            else:
                num_allowed_direction += 1
            j += 1

        assert (num_allowed_direction == 2 or num_allowed_direction == 3)

        # if there is no entry in the ghost table, use uniform distribution
        if sum(probabilities) == 0:

            probabilities = []

            if num_allowed_direction == 2:
                for allowed_dir in allowed_next_ghost_dirs:
                    if not allowed_dir:
                        probabilities.append(0)
                    else:
                        probabilities.append(0.5)

            if num_allowed_direction == 3:
                for allowed_dir in allowed_next_ghost_dirs:
                    if not allowed_dir:
                        probabilities.append(0)
                    else:
                        probabilities.append(1 / 3)

        probabilities = [round(elem, 2) for elem in probabilities]

        if (sum(probabilities) == 0.97):
            probabilities = [elem + 0.01 for elem in probabilities if elem >= 0]

        if (sum(probabilities) == 0.98):
            for i in range(len(probabilities)):
                if probabilities[i] > 0:
                    probabilities[i] = probabilities[i] + 0.02
                    break

        if (sum(probabilities) == 0.99):
            for i in range(len(probabilities)):
                if probabilities[i] > 0:
                    probabilities[i] = probabilities[i] + 0.01
                    break

        if (sum(probabilities) == 1.01):
            for i in range(len(probabilities)):
                if probabilities[i] > 0:
                    probabilities[i] = probabilities[i] - 0.01
                    break

        if (sum(probabilities) == 1.02):
            for i in range(len(probabilities)):
                if probabilities[i] > 0:
                    probabilities[i] = probabilities[i] - 0.02
                    break

        if (sum(probabilities) == 1.03):
            probabilities = [elem - 0.01 for elem in probabilities if elem >= 0]

        if (sum(probabilities) != 1.0):
            print("WARNING: Sum of probsbilities is not 1.0!", probabilities)
            str1 = " crossing_id_ghost " + str(crossing_id_ghost) + " ghost_dir " + str(
                ghost_dir) + " positioning_pacman " + str(positioning_pacman)
            print(str1)

        return probabilities

    def encodeCrossingStatement(self, x_ghost, y_ghost, dir_ghost, positioning_pacman, allowed_next_ghost_dirs):

        crossing_id_ghost = self.getCrossingIDAtPos(x_ghost, y_ghost)

        probabilites = self.getProbabilitiesFromGhostTable(crossing_id_ghost, dir_ghost, positioning_pacman,
                                                           allowed_next_ghost_dirs)

        # encode left handside
        prismStr = "  [g] (xG=" + str(x_ghost) + " & yG=" + str(y_ghost) + " & dG=" + str(dir_ghost) + " & "
        prismStr += self.encodePositioningTerm(positioning_pacman)
        prismStr += ")"

        prismStr += " -> "

        # encode right handside
        first = True

        if probabilites[0] > 0:
            prismStr += str(probabilites[0]) + ": (xG'=xG+1) & (dG'=" + str(RIGHT) + ")"
            first = False

        if probabilites[1] > 0:
            if not first:
                prismStr += " + "
            prismStr += str(probabilites[1]) + ": (yG'=yG+1) & (dG'=" + str(UP) + ")"
            first = False

        if probabilites[2] > 0:
            if not first:
                prismStr += " + "
            prismStr += str(probabilites[2]) + ": (xG'=xG-1) & (dG'=" + str(LEFT) + ")"
            first = False

        if probabilites[3] > 0:
            if not first:
                prismStr += " + "
            prismStr += str(probabilites[3]) + ": (yG'=yG-1) & (dG'=" + str(DOWN) + ")"

        prismStr += ";"
        return prismStr

    def encodeTLeftCrossing(self, x_ghost, y_ghost):

        prismStr = "  [t-left crossing]" + "\n"

        for positioning_pacman in self.getRelevantPositionings(x_ghost, y_ghost):
            prismStr += "\n  //- positioning pacman = " + self.mapIntDirectionToString(positioning_pacman) + "\n\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, RIGHT, positioning_pacman,
                                                     [False, True, False, True])
            prismStr += "  // ghost is moving to the right -> continues to go up or down" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, UP, positioning_pacman,
                                                     [False, True, True, False])
            prismStr += "  // ghost is moving up -> continues to go up or left" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, DOWN, positioning_pacman,
                                                     [False, False, True, True])
            prismStr += "  // ghost is moving down -> continues to go left or down" + "\n"

        prismStr += "\n  //- position pacman == position ghost\n\n"
        prismStr += self.encodeCrossingStatementStop(x_ghost, y_ghost)

        return prismStr

    def encodeTRightCrossing(self, x_ghost, y_ghost):

        prismStr = "  [t-right crossing]" + "\n"

        for positioning_pacman in self.getRelevantPositionings(x_ghost, y_ghost):
            prismStr += "\n  //- positioning pacman = " + self.mapIntDirectionToString(positioning_pacman) + "\n\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, LEFT, positioning_pacman,
                                                     [False, True, False, True])
            prismStr += "  // ghost is moving to the left -> continues to go up or down" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, UP, positioning_pacman,
                                                     [True, True, False, False])
            prismStr += "  // ghost is moving up -> continues to go right or up" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, DOWN, positioning_pacman,
                                                     [True, False, False, True])
            prismStr += "  // ghost is moving down -> continues to go right or down" + "\n"

        prismStr += "\n  //- position pacman == position ghost\n\n"
        prismStr += self.encodeCrossingStatementStop(x_ghost, y_ghost)

        return prismStr

    def encodeTDownCrossing(self, x_ghost, y_ghost):

        prismStr = "  [t-down crossing]" + "\n "

        for positioning_pacman in self.getRelevantPositionings(x_ghost, y_ghost):
            prismStr += "\n  //- positioning pacman = " + self.mapIntDirectionToString(positioning_pacman) + "\n\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, RIGHT, positioning_pacman,
                                                     [True, False, False, True])
            prismStr += "  // ghost is moving to the right -> continues to go right or down" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, UP, positioning_pacman,
                                                     [True, False, True, False])
            prismStr += "  // ghost is moving up -> continues to go right or left" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, LEFT, positioning_pacman,
                                                     [False, False, True, True])
            prismStr += "  // ghost is moving left -> continues to go left or down" + "\n"

        prismStr += "\n  //- position pacman == position ghost\n\n"
        prismStr += self.encodeCrossingStatementStop(x_ghost, y_ghost)

        return prismStr

    def encodeTUpCrossing(self, x_ghost, y_ghost):

        prismStr = "  [t-up crossing]" + "\n "

        for positioning_pacman in self.getRelevantPositionings(x_ghost, y_ghost):
            prismStr += "\n  //- positioning pacman = " + self.mapIntDirectionToString(positioning_pacman) + "\n\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, RIGHT, positioning_pacman,
                                                     [True, True, False, False])
            prismStr += "  // ghost is moving to the right -> continues to go right or up" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, DOWN, positioning_pacman,
                                                     [True, False, True, False])
            prismStr += "  // ghost is moving down -> continues to go right or left" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, LEFT, positioning_pacman,
                                                     [False, True, True, False])
            prismStr += "  // ghost is moving to the left -> continues to go up or left" + "\n"

        prismStr += "\n  //- position pacman == position ghost\n\n"
        prismStr += self.encodeCrossingStatementStop(x_ghost, y_ghost)

        return prismStr

    def encodeCenterCrossing(self, x_ghost, y_ghost):

        prismStr = "  [center crossing]" + "\n"

        for positioning_pacman in self.getRelevantPositionings(x_ghost, y_ghost):
            prismStr += "\n  //- positioning pacman = " + self.mapIntDirectionToString(positioning_pacman) + "\n\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, RIGHT, positioning_pacman,
                                                     [True, True, False, True])
            prismStr += "  // ghost is moving to the right -> continues to go right, up or down" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, UP, positioning_pacman,
                                                     [True, True, True, False])
            prismStr += "  // ghost is moving up -> continues to go right, up or left" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, LEFT, positioning_pacman,
                                                     [False, True, True, True])
            prismStr += "  // ghost is moving left -> continues to go up, left or down" + "\n"

            prismStr += self.encodeCrossingStatement(x_ghost, y_ghost, DOWN, positioning_pacman,
                                                     [True, False, True, True])
            prismStr += "  // ghost is moving down -> continues to go right, left or down" + "\n"

        prismStr += "\n  //- position pacman == position ghost\n\n"
        prismStr += self.encodeCrossingStatementStop(x_ghost, y_ghost)

        return prismStr

    # returns for the current position of the ghost the positionings of pacman that are possible
    def getRelevantPositionings(self, x_ghost, y_ghost):

        positionings_pacman = []

        if x_ghost < self.w - 2:
            positionings_pacman.append(RIGHT)
        if y_ghost < self.h - 2:
            positionings_pacman.append(UP)
        if x_ghost > 1:
            positionings_pacman.append(LEFT)
        if y_ghost > 1:
            positionings_pacman.append(DOWN)

        if x_ghost < self.w - 2 and y_ghost < self.h - 2:
            positionings_pacman.append(TOP_RIGHT)
        if x_ghost > 1 and y_ghost < self.h - 2:
            positionings_pacman.append(TOP_LEFT)
        if x_ghost < self.w - 2 and y_ghost > 1:
            positionings_pacman.append(BOTTOM_RIGHT)
        if x_ghost > 1 and y_ghost > 1:
            positionings_pacman.append(BOTTOM_LEFT)

        return positionings_pacman

    def mapIntDirectionToString(self, direction):

        if direction == RIGHT:
            return STR_RIGHT
        if direction == LEFT:
            return STR_LEFT
        if direction == UP:
            return STR_UP
        if direction == DOWN:
            return STR_DOWN
        if direction == STOP:
            return STR_STOP
        if direction == TOP_RIGHT:
            return STR_TOP_RIGHT
        if direction == TOP_LEFT:
            return STR_TOP_LEFT
        if direction == BOTTOM_RIGHT:
            return STR_BOTTOM_RIGHT
        if direction == BOTTOM_LEFT:
            return STR_BOTTOM_LEFT

    def encodeHorizontalCorridor(self, x_start, x_end, y, isGhost):

        if isGhost:
            agent = "G"
        else:
            agent = "P"

        crossing_left = 0
        if self.isCrossing(x_start - 1, y):
            crossing_left = -1

        crossing_right = 0
        if self.isCrossing(x_end + 1, y):
            crossing_right = 1

        deadend_left = 0
        if self.isDeadend(x_start, y):
            deadend_left = 1

        deadend_right = 0
        if self.isDeadend(x_end, y):
            deadend_right = -1

        corner_left = 0
        if self.isCorner(x_start, y):
            corner_left = 1

        corner_right = 0
        if self.isCorner(x_end, y):
            corner_right = -1

        assert (corner_left + deadend_left <= 1)
        assert (corner_right + deadend_right <= 1)

        prismStr = "  //horizontal corridor" + "\n"

        prismStr += "  [" + agent.lower() + "] (y" + agent + "=" + str(y) + " & x" + agent + " >=" + str(
            x_start + deadend_left + corner_left) + " & x" + agent + " <=" + str(
            x_end - 1 + crossing_right) + " & d" + agent + " =0) -> 1:(x" + agent + "'=x" + agent + "+1) & (d" + agent + "'=0); " + "\n"

        prismStr += "  [" + agent.lower() + "] (y" + agent + "=" + str(y) + " & x" + agent + " >=" + str(
            x_start + 1 + crossing_left) + " & x" + agent + " <=" + str(
            x_end + deadend_right + corner_right) + " & d" + agent + " =2) -> 1:(x" + agent + "'=x" + agent + "-1) & (d" + agent + "'=2); " + "\n"

        """old
        prismStr += "  ["+ agent.lower() + "] (y"+ agent +"=" + str(y) + " & x"+ agent +" >=" + str(x_start + deadend_left + corner_left) + " & x"+ agent +" <=" + str(
            x_end - 1 + crossing_right) + " & d"+ agent +" !=2) -> 1:(x"+ agent +"'=x"+ agent +"+1) & (d"+ agent +"'=0); " + "\n"

        prismStr += "  ["+ agent.lower() + "] (y"+ agent +"=" + str(y) + " & x"+ agent +" >=" + str(x_start + 1 + crossing_left) + " & x"+ agent +" <=" + str(
            x_end + deadend_right + corner_right) + " & d"+ agent +" !=0) + ") -> 1:(x"+ agent +"'=x"+ agent +"-1) & (d"+ agent +"'=2); " + "\n"
        """

        return prismStr

    def encodeVerticalCorridor(self, x, y_start, y_end, isGhost):

        if isGhost:
            agent = "G"
        else:
            agent = "P"

        crossing_below = 0
        if self.isCrossing(x, y_start - 1):
            crossing_below = -1

        crossing_above = 0
        if self.isCrossing(x, y_end + 1):
            crossing_above = 1

        deadend_below = 0
        if self.isDeadend(x, y_start):
            deadend_below = 1

        deadend_above = 0
        if self.isDeadend(x, y_end):
            deadend_above = -1

        corner_below = 0
        if self.isCorner(x, y_start):
            corner_below = 1

        corner_above = 0
        if self.isCorner(x, y_end):
            corner_above = -1

        assert (corner_below + deadend_below <= 1)
        assert (corner_above + deadend_above <= 1)

        prismStr = "  //vertical corridor" + "\n"
        prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + " >=" + str(
            y_start + deadend_below + corner_below) + " & y" + agent + " <=" + str(
            y_end - 1 + crossing_above) + " & d" + agent + "=1) -> 1:(y" + agent + "'=y" + agent + "+1) & (d" + agent + "'=1); " + "\n"
        prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + " >=" + str(
            y_start + 1 + crossing_below) + " & y" + agent + " <=" + str(
            y_end + deadend_above + corner_above) + " & d" + agent + " =3) -> 1:(y" + agent + "'=y" + agent + "-1) & (d" + agent + "'=3); " + "\n"

        """ old
        prismStr = "  //vertical corridor" + "\n"
        prismStr += "  ["+ agent.lower() + "] (x"+ agent +"=" + str(x) + " & y"+ agent +" >=" + str(y_start + deadend_below + corner_below) + " & y"+ agent +" <=" + str(
            y_end - 1 + crossing_above) + " & d"+ agent +" !=3) -> 1:(y"+ agent +"'=y"+ agent +"+1) & (d"+ agent +"'=1); " + "\n"
        prismStr += "  ["+ agent.lower() + "] (x"+ agent +"=" + str(x) + " & y"+ agent +" >=" + str(y_start + 1 + crossing_below) + " & y"+ agent +" <=" + str(
            y_end + deadend_above + corner_above) + " & d"+ agent +" !=1) -> 1:(y"+ agent +"'=y"+ agent +"-1) & (d"+ agent +"'=3); " + "\n"
        """

        return prismStr

    def encodeDeadend(self, x, y, dir, isGhost):

        if isGhost:
            agent = "G"
        else:
            agent = "P"

        prismStr = ""
        if dir == RIGHT:
            prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + "=" + str(
                y) + ") -> 1: (x" + agent + "'=x" + agent + "-1) & (d" + agent + "'=2);" + "\n"
        if dir == LEFT:
            prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + "=" + str(
                y) + ") -> 1: (x" + agent + "'=x" + agent + "+1) & (d" + agent + "'=0);" + "\n"
        if dir == UP:
            prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + "=" + str(
                y) + ") -> 1: (y" + agent + "'=y" + agent + "+1) & (d" + agent + "'=1);" + "\n"
        if dir == DOWN:
            prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + "=" + str(
                y) + ") -> 1: (y" + agent + "'=y" + agent + "-1) & (d" + agent + "'=3);" + "\n"

        return prismStr

    def encodeCorner(self, x, y, isGhost):

        assert (self.isCorner(x, y))
        nh = self.neighborHood((x, y))

        prismStr = ""

        if self.cornerRightBottom(nh):
            prismStr += self.encodeCornerStatement(x, y, isGhost, RIGHT, UP)
            prismStr += self.encodeCornerStatement(x, y, isGhost, DOWN, LEFT)

        if self.cornerRightTop(nh):
            prismStr += self.encodeCornerStatement(x, y, isGhost, RIGHT, DOWN)
            prismStr += self.encodeCornerStatement(x, y, isGhost, UP, LEFT)

        if self.cornerLeftBottom(nh):
            prismStr += self.encodeCornerStatement(x, y, isGhost, LEFT, UP)
            prismStr += self.encodeCornerStatement(x, y, isGhost, DOWN, RIGHT)

        if self.cornerLeftTop(nh):
            prismStr += self.encodeCornerStatement(x, y, isGhost, LEFT, DOWN)
            prismStr += self.encodeCornerStatement(x, y, isGhost, UP, RIGHT)

        return prismStr

    def encodeCornerStatement(self, x, y, isGhost, dir, next_dir):

        if isGhost:
            agent = "G"
        else:
            agent = "P"

        prismStr = ""
        prismStr += "  [" + agent.lower() + "] (x" + agent + "=" + str(x) + " & y" + agent + "=" + str(y) + " & "
        prismStr += "(d" + agent + "=" + str(dir) + " | d" + agent + "=" + str(next_dir) + "))"

        if next_dir == RIGHT:
            prismStr += " -> 1: (x" + agent + "'=x" + agent + "+1) & (d" + agent + "'=0);" + "\n"
        if next_dir == LEFT:
            prismStr += " -> 1: (x" + agent + "'=x" + agent + "-1) & (d" + agent + "'=2);" + "\n"
        if next_dir == UP:
            prismStr += "-> 1: (y" + agent + "'=y" + agent + "+1) & (d" + agent + "'=1);" + "\n"
        if next_dir == DOWN:
            prismStr += " -> 1: (y" + agent + "'=y" + agent + "-1) & (d" + agent + "'=3);" + "\n"

        return prismStr

    def encodeGhostModule(self):

        prismStr = "\n//GHOST0\n"

        prismStr += "module ghost0\n"
        prismStr += "  xG : [0..xSize] init x; // x position of Ghost" + "\n"
        prismStr += "  yG : [0..ySize] init x; // y position of Ghost" + "\n"
        prismStr += "  dG : [0..3] init x; //direction of ghost (0=right, 1=up, 2=left, 3=down)" + "\n\n"

        # encode horizontal corridors
        for c in self.hcorr:
            prismStr += self.encodeHorizontalCorridor(c[0][0], c[1][0], c[0][1], True)
        prismStr += "\n"

        # encode vertical corridors
        for c in self.vcorr:
            prismStr += self.encodeVerticalCorridor(c[0][0], c[0][1], c[1][1], True)
        prismStr += "\n"

        # encode deadends
        prismStr += "  //deadends" + "\n"
        for d in self.deadend:
            prismStr += self.encodeDeadend(d[0][0], d[0][1], d[1], True)
        prismStr += "\n"

        # encode corners
        prismStr += "  //corners" + "\n"
        for c in self.corners:
            prismStr += self.encodeCorner(c[0][0], c[0][1], True)
        prismStr += "\n"

        # encode crossings
        for c in self.crossings:
            x = c[0]
            y = c[1]
            nh = self.neighborHood(c)

            prismStr += "\n  //crossing at position (" + str(x) + ", " + str(y) + ")"

            # t-crossings
            if self.tLeftCrossing(nh):
                prismStr += self.encodeTLeftCrossing(x, y)

            if self.tRightCrossing(nh):
                prismStr += self.encodeTRightCrossing(x, y)

            if self.tDownCrossing(nh):
                prismStr += self.encodeTDownCrossing(x, y)

            if self.tUpCrossing(nh):
                prismStr += self.encodeTUpCrossing(x, y)

            # center-crossing
            if self.centerCrossing(nh):
                prismStr += self.encodeCenterCrossing(x, y)

        # patch variables to ghostnumber
        prismStr = prismStr.replace("xG", "xG0")
        prismStr = prismStr.replace("yG", "yG0")
        prismStr = prismStr.replace("dG", "dG0")
        prismStr = prismStr.replace("[g]", "[g0]")

        prismStr += "  [stop0] true -> 1:(xG0'=0) & (yG0'=0);\n"

        # define footer
        prismStr += "endmodule" + "\n"
        return prismStr

    def encodePacmanStatement(self, x_c, y_c, x_n, y_n):
        return "  [p] (xP=" + str(x_c) + " & yP=" + str(y_c) + ") -> 1: (xP'=" + str(x_n) + ") & (yP'=" + str(
            y_n) + ");"

    def encodePacmanStandsStill(self, x, y, dir):
        prismStr = "  [" + dir + "] (xP=" + str(x) + " & yP=" + str(y) + " & pMove=" + str(
            ghost_nr) + ") -> 1: (xP'=xP);\n"
        return prismStr

    def encodeCorssingsPacman(self):

        prismStr = ""
        for c in self.crossings:

            x = c[0]
            y = c[1]

            prismStr += "  // crossing at position(" + str(x) + ", " + str(y) + ")\n\n"

            # action is RIGHT
            if (self.isWall([x + 1, y])):
                prismStr += self.encodePacmanStandsStill(x, y, "right")
            else:
                prismStr += "  [right] (xP=" + str(x) + " & yP=" + str(y) + ") -> 1: (xP'=xP+1) & (dP'=0);\n"

            # action is UP
            if (self.isWall([x, y + 1])):
                prismStr += self.encodePacmanStandsStill(x, y, "up")
            else:
                prismStr += "  [up] (xP=" + str(x) + " & yP=" + str(y) + ") -> 1: (yP'=yP+1) & (dP'=1);\n"

            # action is LEFT
            if (self.isWall([x - 1, y])):
                prismStr += self.encodePacmanStandsStill(x, y, "left")
            else:
                prismStr += "  [left] (xP=" + str(x) + " & yP=" + str(y) + ") -> 1: (xP'=xP-1) & (dP'=2);\n"

            # action is DOWN
            if (self.isWall([x, y - 1])):
                prismStr += self.encodePacmanStandsStill(x, y, "down")
            else:
                prismStr += "  [down] (xP=" + str(x) + " & yP=" + str(y) + ") -> 1: (yP'=yP-1) & (dP'=3);\n"
            prismStr += "\n"

        return prismStr

    def encodePacmanModuleUsingCrossings(self):

        prismStr = ""

        # encode horizontal corridors
        for c in self.hcorr:
            prismStr += self.encodeHorizontalCorridor(c[0][0], c[1][0], c[0][1], False)
        prismStr += "\n"

        # encode vertical corridors
        for c in self.vcorr:
            prismStr += self.encodeVerticalCorridor(c[0][0], c[0][1], c[1][1], False)
        prismStr += "\n"

        # encode deadends
        prismStr += "  //deadends" + "\n"
        for d in self.deadend:
            prismStr += self.encodeDeadend(d[0][0], d[0][1], d[1], False)
        prismStr += "\n"

        # encode corners
        prismStr += "  //corners" + "\n"
        for c in self.corners:
            prismStr += self.encodeCorner(c[0][0], c[0][1], False)
        prismStr += "\n"

        prismStr += self.encodeCorssingsPacman()

        return prismStr

    def encodePacmanModule(self):

        prismStr = "\n//PACMAN" + "\n"
        prismStr += "module pacman" + "\n"

        prismStr += "  xP : [1..xSize] init x;" + "// x position of Pacman\n"
        prismStr += "  yP : [1..ySize] init x;" + "// y position of Pacman\n"
        if not USE_CORRIDOR_ENCODING:
            prismStr += "  dP : [0 .. 3] init x; //direction of pacman (0=right, 1=up, 2=left, 3=down)\n"
        prismStr += "\n"

        if USE_CORRIDOR_ENCODING:
            prismStr += "\nINSERT CURRENT PATH HERE \n"
        else:
            prismStr += self.encodePacmanModuleUsingCrossings()

        prismStr += "endmodule"

        return prismStr

    def encodeArbiter(self):

        prismStr = "module arbiter\n"
        prismStr += "  pMove : [0 .. 1] init 0; //token to determine who is allowed to move\n"
        if STEPS_IN_ENCODING:
            prismStr += "  steps : [0 .. MAXSTEPS] init 0; //number of steps we plan ahead\n"
        prismStr += "\n"

        prismStr += "  [g0]    (pMove = 0) -> 1:(pMove' = 1);\n"


        prismStr += "  [g0]    (pMove=0) & !deactive0 & ((xG0 < xP ? xP - xG0: xG0 - xP)"
        prismStr += "+ (yG0 < yP ? yP - yG0: yG0 - yP) <= 2 * (MAXSTEPS - steps)) -> 1:(pMove' = 1);\n"

        prismStr += "  [stop0] (pMove=0) & (deactive0 | ((xG0 < xP ? xP - xG0: xG0 - xP)"
        prismStr += " + (yG0 < yP ? yP - yG0: yG0 - yP) > 2 * (MAXSTEPS - steps))) ->  1:(pMove' = 1);\n"

        prismStr += "\n  [p]     (pMove = 1 & steps < MAXSTEPS ) -> 1:(pMove' = 0) & (steps' = steps + 1);\n"

        if not USE_CORRIDOR_ENCODING:
            prismStr += "  [left]  (pMove = 1 & steps < MAXSTEPS ) -> 1:(pMove' = 0) & (steps' = steps + 1);\n"
            prismStr += "  [right] (pMove = 1 & steps < MAXSTEPS ) -> 1:(pMove' = 0) & (steps' = steps + 1);\n"
            prismStr += "  [up]    (pMove = 1 & steps < MAXSTEPS ) -> 1:(pMove' = 0) & (steps' = steps + 1);\n"
            prismStr += "  [down]  (pMove = 1 & steps < MAXSTEPS ) -> 1:(pMove' = 0) & (steps' = steps + 1);\n"
        prismStr += "\n"

        prismStr += "  []    (pMove=1 & (steps = MAXSTEPS | deactive0)) -> 1: (pMove'=1);\n"

        prismStr += "endmodule\n\n"

        return prismStr

    def transformStateToComment(self, state):
        resultStr = ""

        stateStr = str(state)
        stateLines = stateStr.splitlines()
        stateLines.pop()  # remove last line "score: " info
        for l in stateLines:
            l = "// " + l + "\n"
            resultStr += l
        return resultStr


    # can only be used if the ghost is not at a crossing.
    # use old ghost direction to determine new position
    def getNextGhostPosition(self, init_ghost, dir_ghost):

        nh = self.neighborHood(init_ghost)  # nh = [pd,pu,pl,pr]
        next_ghost = None

        if self.isCrossing(init_ghost[0], init_ghost[1]):
            return None

        if not self.isDeadend(init_ghost[0], init_ghost[1]):
            dir_counter = 0
            for pos in nh:
                if not self.isWall(pos):
                    dir_counter += 1
            if not dir_counter == 2:
                print("init_ghost, dir_ghost, dir_counter:", init_ghost, dir_ghost, dir_counter)
                assert (False)

            if dir_ghost == RIGHT:
                if not self.isWall(nh[3]):
                    next_ghost = nh[3]
                if not self.isWall(nh[0]):
                    next_ghost = nh[0]
                if not self.isWall(nh[1]):
                    next_ghost = nh[1]

            if dir_ghost == LEFT:
                if not self.isWall(nh[2]):
                    next_ghost = nh[2]
                if not self.isWall(nh[0]):
                    next_ghost = nh[0]
                if not self.isWall(nh[1]):
                    next_ghost = nh[1]

            if dir_ghost == UP:
                if not self.isWall(nh[2]):
                    next_ghost = nh[2]
                if not self.isWall(nh[3]):
                    next_ghost = nh[3]
                if not self.isWall(nh[1]):
                    next_ghost = nh[1]

            if dir_ghost == DOWN:
                if not self.isWall(nh[2]):
                    next_ghost = nh[2]
                if not self.isWall(nh[3]):
                    next_ghost = nh[3]
                if not self.isWall(nh[0]):
                    next_ghost = nh[0]

        else:  # ghost at Deadend
            next_ghost = self.getPreviousPosition(init_ghost, dir_ghost)

        assert (next_ghost != None)
        assert (not self.isWall(next_ghost))

        return next_ghost


    # Prints all paths from init_pacman in start_direction with m steps
    def computePaths(self, init_pacman, start_direction):

        # all computed paths
        self.paths = []

        next_pos = self.getNextPosition(init_pacman, start_direction)

        assert (not self.isWall(next_pos))
        # Call the recursive helper function to print all paths
        self.computeAllPathsUtil(init_pacman, next_pos, STEPS - 1, [])

        # add last pos twice at each path to get a self loop in the storm encoding (avoids deadlocks)
        final_paths = []
        for path in self.paths:
            new_path = path[:]
            new_path.append(path[len(path) - 1])
            final_paths.append(new_path)

        # store number of paths in Table
        #flag = False
        #for row in range(0, len(self.path_counter)):
        #    r = self.path_counter[row]
        #    if r[0] == init_pacman[0] and r[1] == init_pacman[1] and r[2] == start_direction:
        #        flag = True
        #        break
        #if flag == False:
        #    self.path_counter.append((init_pacman[0], init_pacman[1], start_direction, len(final_paths)))

        for path in final_paths:
            assert (len(path) == STEPS+1)

        return final_paths

    '''A recursive function to print all paths from 'pos' in direction 'dir' with length of 'steps'.
        visited[] keeps track of vertices in current path.
        path[] stores actual vertices and path_index is current
        index in path[]'''

    def computeAllPathsUtil(self, prev_pos, curr_pos, steps, prev_path):

        assert (curr_pos[0] >= 1 and curr_pos[1] >= 1 and curr_pos[0] < self.w - 1 and curr_pos[1] < self.h - 1)
        assert (not self.isWall(curr_pos))

        path = prev_path[:]
        path.append(curr_pos)

        # If current path is long enough, then add current path[]
        if steps == 0:
            self.paths.append(list(path))
            return
        else:
            # Recur for all the vertices adjacent to this vertex
            if self.isDeadend(curr_pos[0], curr_pos[1]):
                next_pos = prev_pos
                self.computeAllPathsUtil(curr_pos, next_pos, steps-1, path)
                return

            if self.isCrossing(curr_pos[0], curr_pos[1]):
                for next_pos in self.neighborHood(curr_pos):
                    if not self.isWall(next_pos):
                        if self.countsOccurencesOfPosInPath(next_pos,path)<1:
                            self.computeAllPathsUtil(curr_pos, next_pos, steps-1, path)
                return

            #curr_pos must be corridor or corner
            for next_pos in self.neighborHood(curr_pos):
                if not self.isWall(next_pos):
                    if next_pos[0]!=prev_pos[0] or next_pos[1]!=prev_pos[1]:
                        self.computeAllPathsUtil(curr_pos, next_pos, steps - 1, path)
                        return

    def countsOccurencesOfPosInPath(self, pos, path):
        counter=0
        for pos_path in path:
            if pos[0]== pos_path[0] and pos[1]== pos_path[1]:
                counter=counter+1
        return counter

    def getDirection(self, position, next_position):

        if position[0]+1 == next_position[0]:
            return RIGHT
        if position[0]-1 == next_position[0]:
            return LEFT
        if position[1]+1 == next_position[1]:
            return UP
        if position[1]-1 == next_position[1]:
            return DOWN
        assert(False)

    def getNextPosition(self, position, next_direction):

        #assert (not self.isWall(position))

        next_pos = list(position)

        if next_direction == RIGHT:
            next_pos[0] = next_pos[0] + 1
        if next_direction == UP:
            next_pos[1] = next_pos[1] + 1
        if next_direction == LEFT:
            next_pos[0] = next_pos[0] - 1
        if next_direction == DOWN:
            next_pos[1] = next_pos[1] - 1

        return next_pos

    def getPreviousPosition(self, position, direction):

        assert (not self.isWall(position))

        prev_pos = list(position)

        if direction == RIGHT:
            prev_pos[0] = prev_pos[0] - 1
        if direction == UP:
            prev_pos[1] = prev_pos[1] - 1
        if direction == LEFT:
            prev_pos[0] = prev_pos[0] + 1
        if direction == DOWN:
            prev_pos[1] = prev_pos[1] + 1

        return prev_pos

    def encodeModel(self, state, ghost_table):

        self.ghost_table = ghost_table
        self.ghostsEncoded = 0

        prismStr = self.transformStateToComment(state)

        prismStr += "\n"
        prismStr += "mdp" + "\n\n"
        prismStr += "//CONSTANTS" + "\n"
        prismStr += "const xSize = " + str(self.w - 1) + ";" + "\n"
        prismStr += "const ySize = " + str(self.h - 1) + ";" + "\n\n"
        if STEPS_IN_ENCODING:
            prismStr += "const MAXSTEPS = " + str(STEPS) + ";" + "\n"
        prismStr += "\n"

        prismStr += "formula deactive0 = (xG0 = 0);\n"
        prismStr += "\n"

        prismStr += self.encodeArbiter()
        prismStr += self.encodeGhostModule()
        prismStr += self.encodePacmanModule()

        # encode crash label
        prismStr += "\n\n//CRASH\n"
        prismStr += "label \"crash\" = (xP = xG0 & yP = yG0);\n"
        prismStr += "label \"safe\" = deactive0;\n"

        return prismStr

    def getCrossingIDAtPos(self, x, y):

        id = 0
        for c in self.crossings:
            if c[0] == x and c[1] == y:
                return id
            id += 1
        return None

    # if the position is a corner, the function returns the ID of both corridors
    def getCorridorIDAtPos(self, x, y):
        id = 0
        allcorr = self.hcorr.copy()
        allcorr += self.vcorr
        ids = []
        for c in allcorr:
            start = c[0]
            end = c[1]
            if start[0] <= x <= end[0]:
                if start[1] <= y <= end[1]:
                    ids.append(id)
            id += 1
        return ids

    def getPositionIDAtPos(self, x, y):
        cid = self.getCorridorIDAtPos(x, y)[0]
        if cid == None:
            cid = self.getCrossingIDAtPos(x, y)
            cid += len(self.hcorr) + len(self.vcorr)
        return cid

    def getHeightOfLayout(self):
        return self.h

    def getWidthOfLayout(self):
        return self.w

    def getVerticalCorridors(self):
        return self.vcorr

    def getHorizintalCorridors(self):
        return self.hcorr

    def getCrossings(self):
        return self.crossings

