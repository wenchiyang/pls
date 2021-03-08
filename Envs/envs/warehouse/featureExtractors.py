# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

import util
from game import Actions


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closestPackages(pos, packages, walls, exclude_closest_packages = False):
    """
    closestPackages -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """

    if exclude_closest_packages:
        return 200

    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a package at this location then exit
        if packages[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no package found
    return None


def distExit(pos, exit, walls):
    "returns the distance to the exit"

    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a package at this location then exit
        if pos_x == exit[0] and pos_y == exit[1]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no package found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether package will be eaten
    - how far away the next package is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of packages and wall locations and get the ghost locations
        packages = state.getPackages()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        loaded = state.getLoadingInfo()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1,2 and 3-steps away
        sum_ghosts_close=0
        for g in ghosts:
            if g in Actions.getLegalClosestNeighbors((x,y), walls):
                sum_ghosts_close=sum_ghosts_close+1

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if loaded is true, the goal is to reach the exit, and no packages can be eaten. --> closest-packages is replacesd
        # by distance to exit
        if loaded:
            features["loaded"] = 1.0
            exit_x, exit_y = state.getExit()[0]
            # if there is no danger of ghosts then add the exit feature
            if not features["#-of-ghosts-1-step-away"] and next_x == exit_x and next_y == exit_y:
                features["goto-exit"] = 1.0

            dist_exit = distExit((next_x, next_y), (exit_x, exit_y), walls)
            assert (dist_exit != None)
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["dist-exit"] = float(dist_exit) / (walls.width * walls.height)

        else:
            features["loaded"] = 0.0
            # if there is no danger of ghosts then add the packages feature
            #if not features["#-of-ghosts-1-step-away"] and packages[next_x][next_y]:

            if packages[next_x][next_y]:
                if sum_ghosts_close == 0:
                    features["eats-packages"] = 1.0
                    dist_package = closestPackages((next_x, next_y), packages, walls, False)
                else:
                    dist_package = closestPackages((next_x, next_y), packages, walls, True)
            else:
                dist_package = closestPackages((next_x, next_y), packages, walls, False)

            if dist_package is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-packages"] = float(dist_package) / (walls.width * walls.height)

        features.divideAll(10.0)

        # print(features)
        # print("---------")

        return features
