import stormpy
import stormpy.core
import os
import numpy
import time
from multiprocessing.dummy import Pool as ThreadPool
from util import ShortestPath
import tempfile
from stormEncoder import StormEncoder
import pickle
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


class Shield:

    def __init__(self, state,symX,symY):
        # get board propertiesin
        self.path_counter = []
        self.shield = None
        self.symX = symX
        self.symY = symY
        self.encoder = StormEncoder(state,symX,symY)


    def getShield(self):
        return self.shield

    def setShield(self, shield):
        self.shield = shield

    def loadShield(self, filename):
        self.shield = pickle.load(open(filename, "rb"))

    def dumpShield(self, dump):
        print("dumping current shield to file: " + dump)
        pickle.dump(self.shield, open(dump, "wb"))


    def prettyPrintShield(self):

        print("\**********************************************************************")
        print("\*************START SHIELD*********************************************")
        print("\**********************************************************************")

        for entry in self.shield:
            init_pacman_x = entry[0]
            init_pacman_y = entry[1]
            dir_pacman = self.encoder.mapIntDirectionToString(entry[2])
            init_ghost_x = entry[3]
            init_ghost_y = entry[4]
            dir_ghost = self.encoder.mapIntDirectionToString(entry[5])

            if USE_CORRIDOR_ENCODING:
                path_nr = entry[6]
                prob = str(round(entry[7], 2))
                # if entry[6]>0:
                print(
                    "Init Pacman: (%d,%d), Dir Pacman %s, Init Ghost: (%d,%d), Dir Ghost %s, path_n %d, Probability %s" % (
                    init_pacman_x, init_pacman_y, dir_pacman, init_ghost_x, init_ghost_y, dir_ghost, path_nr, prob))

            else:
                prob = str(round(entry[6], 2))
                if entry[6] > 0:
                    print(
                        "Init Pacman: (%d,%d), Dir Pacman %s, Init Ghost: (%d,%d), Dir Ghost %s, Probability %s" % (
                        init_pacman_x, init_pacman_y, dir_pacman, init_ghost_x, init_ghost_y, dir_ghost, prob))

        print("\**********************************************************************")
        print("\*************END SHIELD*********************************************")
        print("\**********************************************************************")



    def agentsMoveTowardEachOther(self, next_pos_pacman, init_ghost, dir_ghost):

        next_ghost = self.encoder.getNextGhostPosition(init_ghost, dir_ghost)
        if self.encoder.isCrossing(next_ghost[0], next_ghost[1]):
            return False

        connecting_corr = self.encoder.getConnectingCorridorsViaPos(next_pos_pacman)

        graph = ShortestPath(connecting_corr)
        path = graph.findShortestPath(next_pos_pacman, init_ghost)
        n_path = graph.findShortestPath(next_pos_pacman, next_ghost)

        if len(path) > len(n_path):
            return True
        return False

    def isCollisionAssured(self, init_pacman, next_dir_pacman, init_ghost, dir_ghost):

        next_pos_pacman = self.encoder.getNextPosition(init_pacman, next_dir_pacman)

        if next_pos_pacman[0] == init_ghost[0] and next_pos_pacman[1] == init_ghost[1]:  # pacman moved in ghost with first step
            # print("Coll Assured: init_pacman, dir_pacman, init_ghosts, dir_ghosts", init_pacman, dir_pacman,init_ghosts, dir_ghosts)
            return True

        assert (not self.encoder.isWall(next_pos_pacman))

        if self.encoder.isSameConnectingCorridor(next_pos_pacman, init_ghost):
            if not self.encoder.isCrossing(init_ghost[0], init_ghost[1]):
                if self.agentsMoveTowardEachOther(next_pos_pacman, init_ghost, dir_ghost):
                    # print("Coll Assured: init_pacman, dir_pacman, init_ghosts, dir_ghosts", init_pacman, dir_pacman, init_ghosts, dir_ghosts)
                    return True

        return False

    def ghostTooFarAway(self, init_pacman, init_ghost):

        x_diff = abs(init_pacman[0] - init_ghost[0])
        y_diff = abs(init_pacman[1] - init_ghost[1])

        if x_diff + y_diff > 2 * STEPS:
            return True
        return False

    def getSymmetricArguments(self, init_pacman, dir_pacman, init_ghosts, dir_ghosts):

        if not self.symX and not self.symY:
            return init_pacman, dir_pacman, init_ghosts, dir_ghosts

        sym_init_ghosts = []
        sym_dir_ghosts = []

        max_x = self.encoder.getWidthOfLayout() - 2
        max_y = self.encoder.getHeightOfLayout() - 2
        middle_x = (max_x // 2) + 1
        middle_y = (max_y // 2) + 1

        sym_init_pac_x = -1
        sym_init_pac_y = -1
        sym_dir_pacman = -1
        if self.symX and init_pacman[0] > middle_x:  # switch positions and directions of Pacman and Ghosts
            # flip pos and dir of pacman
            x_shift = init_pacman[0] - middle_x
            sym_init_pac_x = middle_x - x_shift

            if dir_pacman == RIGHT:
                sym_dir_pacman = LEFT
            if dir_pacman == LEFT:
                sym_dir_pacman = RIGHT

        if self.symY and init_pacman[1] > middle_y:
            # flip pos and dir of pacman
            y_shift = init_pacman[1] - middle_y
            sym_init_pac_y = middle_y - y_shift

            if dir_pacman == UP:
                sym_dir_pacman = DOWN
            if dir_pacman == DOWN:
                sym_dir_pacman = UP

        if sym_dir_pacman == -1:
            sym_dir_pacman = dir_pacman

        if sym_init_pac_x == -1:
            sym_init_pac_x = init_pacman[0]
        if sym_init_pac_y == -1:
            sym_init_pac_y = init_pacman[1]

        # switch dir of ghosts
        for i in range(0, len(init_ghosts)):
            sym_dir_ghost = STOP
            if self.symX and init_pacman[0] > middle_x:
                if dir_ghosts[i] == RIGHT:
                    sym_dir_ghost = LEFT
                if dir_ghosts[i] == LEFT:
                    sym_dir_ghost = RIGHT

            if self.symY and init_pacman[1] > middle_y:
                if dir_ghosts[i] == UP:
                    sym_dir_ghost = DOWN
                if dir_ghosts[i] == DOWN:
                    sym_dir_ghost = UP

            if sym_dir_ghost == STOP:
                sym_dir_ghost = dir_ghosts[i]

            sym_dir_ghosts.append(sym_dir_ghost)

        # switch pos of ghosts
        for i in range(0, len(init_ghosts)):

            sym_x_ghost = -1
            if self.symX and init_pacman[0] > middle_x:
                if init_ghosts[i][0] > middle_x:
                    x_shift = init_ghosts[i][0] - middle_x
                    sym_x_ghost = middle_x - x_shift

                if init_ghosts[i][0] < middle_x:
                    x_shift = middle_x - init_ghosts[i][0]
                    sym_x_ghost = middle_x + x_shift
            if sym_x_ghost == -1:
                sym_x_ghost = init_ghosts[i][0]

            sym_y_ghost = -1
            if self.symY and init_pacman[1] > middle_y:
                if init_ghosts[i][1] > middle_y:
                    y_shift = init_ghosts[i][1] - middle_y
                    sym_y_ghost = middle_y - y_shift

                if init_ghosts[i][1] < middle_y:
                    y_shift = middle_y - init_ghosts[i][1]
                    sym_y_ghost = middle_y + y_shift
            if sym_y_ghost == -1:
                sym_y_ghost = init_ghosts[i][1]

            sym_init_ghosts.append([sym_x_ghost, sym_y_ghost])

        return [[sym_init_pac_x, sym_init_pac_y], sym_dir_pacman, sym_init_ghosts, sym_dir_ghosts]

    def computeJointProbability(self, probs):

        res = -1

        if len(probs) == 1:
            res = probs[0]

        if len(probs) == 2:
            res = probs[0] + probs[1] - probs[0] * probs[1]

        if len(probs) == 3:
            res = probs[0] + probs[1] + probs[2] - probs[0] * probs[1] - probs[0] * probs[2] - probs[1] * probs[2] + \
                  probs[0] * probs[1] * probs[2]

        if len(probs) == 4:
            probs1 = probs[0] + probs[1] + probs[2] + probs[3]
            probs2 = -probs[0] * probs[1] - probs[0] * probs[2] - probs[0] * probs[3] - probs[1] * probs[2] - probs[1] * \
                     probs[3] - probs[2] * probs[3]
            probs3 = probs[0] * probs[1] * probs[2] + probs[0] * probs[1] * probs[3] + probs[0] * probs[2] * probs[3] + \
                     probs[1] * probs[2] * probs[3]
            probs4 = -probs[0] * probs[1] * probs[2] * probs[3]
            res = probs1 + probs2 + probs3 + probs4

        assert (res >= 0)
        return res


    def getFromShieldProbabilityToGetEaten(self, init_pacman, next_dir_pacman, init_ghosts, dir_ghosts):

        init_pacman, next_dir_pacman, init_ghosts, dir_ghosts = self.getSymmetricArguments(init_pacman, next_dir_pacman,
                                                                                           init_ghosts, dir_ghosts)
        next_pos_pacman = self.encoder.getNextPosition(init_pacman, next_dir_pacman)

        assert (not self.encoder.isWall(init_pacman))
        assert (len(init_ghosts) == len(dir_ghosts))
        assert (not self.encoder.isWall(next_pos_pacman))

        for i in range(0, len(init_ghosts)):
            assert (not self.encoder.isWall(init_ghosts[i]))

        for init_ghost in init_ghosts:
            if init_ghost[0] == init_pacman[0] and init_ghost[1] == init_pacman[1]:
                return 1.0

        for i in range (0, len(init_ghosts)):
            if self.isCollisionAssured(init_pacman, next_dir_pacman, init_ghosts[i], dir_ghosts[i]):
                return 1.0

        if USE_CORRIDOR_ENCODING:
            probs_per_path = []  # path_nr [probs per ghost per path]

            for rowcount in range(0, len(self.shield)):  # get current entry
                r = self.shield[rowcount]
                for ghost_nr in range(0, len(init_ghosts)):
                    if r[0] == init_pacman[0] and r[1] == init_pacman[1] and r[2] == next_dir_pacman:
                        if r[3] == init_ghosts[ghost_nr][0] and r[4] == init_ghosts[ghost_nr][1] and r[5] == dir_ghosts[ghost_nr]:
                            path_nr = r[6]
                            prob_for_path_for_ghost = r[7]

                            # store entry in probs_per_path
                            flag = False
                            for rowcount_probs in range(0, len(probs_per_path)):  # get current entry
                                r_probs = probs_per_path[rowcount_probs]

                                if r_probs[0] == path_nr:
                                    new_probs = r_probs[1] + [prob_for_path_for_ghost]
                                    probs_per_path[rowcount_probs] = (path_nr, new_probs)
                                    flag = True

                            if not flag:  # create new list entry
                                probs_per_path.append((path_nr, [prob_for_path_for_ghost]))

            if len(probs_per_path) < len(self.encoder.computePaths(init_pacman, next_dir_pacman)):
                return 0

            probs = []
            for path_nr in range(0, len(probs_per_path)):
                probs.append(self.computeJointProbability(probs_per_path[path_nr][1]))
            res_prob = min(probs)

        else:
            probs = []
            for rowcount in range(0, len(self.shield)):  # get current entry
                r = self.shield[rowcount]
                for ghost_nr in range(0, len(init_ghosts)):
                    if r[0] == init_pacman[0] and r[1] == init_pacman[1] and r[2] == next_dir_pacman:
                        if r[3] == init_ghosts[ghost_nr][0] and r[4] == init_ghosts[ghost_nr][1] and r[5] == dir_ghosts[
                            ghost_nr]:
                            probs.append(r[6])

            return(max(probs))

        if res_prob < 0:
            print(
                "RESULT: init_pacman[0], init_pacman[1], dir_pacman, init_ghosts[0][0], init_ghosts[0][1], dir_ghosts[0], prob",
                init_pacman[0], init_pacman[1], next_dir_pacman, init_ghosts[0][0], init_ghosts[0][1], dir_ghosts[0],
                res_prob)

        assert (res_prob >= 0)
        return res_prob


    def computeShieldEntry(self, init_pacman, next_dir_pacman, init_ghost, dir_ghost, prismStr):

        # direction is always the direction the agent has when he arrived at trhe current position

        assert (not self.encoder.isWall(init_pacman))

        next_pos_pacman = self.encoder.getNextPosition(init_pacman, next_dir_pacman)
        if self.encoder.isWall(next_pos_pacman):
            return

        if self.encoder.isWall(init_ghost):
            return
        if init_ghost[0] == init_pacman[0] and init_ghost[1] == init_pacman[1]:
            return
        if init_ghost[0] == next_pos_pacman[0] and init_ghost[1] == next_pos_pacman[1]:
            return
        if self.encoder.isWall(self.encoder.getPreviousPosition(init_ghost, dir_ghost)):
            return
        if self.isCollisionAssured(init_pacman, next_dir_pacman, init_ghost, dir_ghost):
            return

        #counts number of storm calls
        self.counter += 1

        if USE_CORRIDOR_ENCODING:
            paths = self.encoder.computePaths(init_pacman, next_dir_pacman)

            for path_nr in range(0, len(paths)):
                prob = self.computeProbabilityToGetEaten(next_pos_pacman, next_dir_pacman, init_ghost, dir_ghost,
                                                         prismStr, paths[path_nr])
                if prob > 0:
                    self.shield.append((init_pacman[0], init_pacman[1], next_dir_pacman, init_ghost[0],
                                        init_ghost[1], dir_ghost, path_nr, prob))
        else:
            prob = self.computeProbabilityToGetEaten(next_pos_pacman, next_dir_pacman, init_ghost, dir_ghost,
                                                     prismStr)

            assert (prob >= 0)
            self.shield.append((init_pacman[0], init_pacman[1], next_dir_pacman, init_ghost[0], init_ghost[1], dir_ghost, prob))

        if self.counter%20==0:
            print("Computed Shield entry so far: ", self.counter)
            self.end = time.time()
            print("time needed for the last 1 calls:", self.end - self.start)
            self.start = time.time()

    def returnWindowAroundPacman(self, pos_pacman):

        x_left = pos_pacman[0] - 2 * STEPS
        if x_left < 1:
            x_left = 1

        x_right = pos_pacman[0] + 2 * STEPS
        if x_right >= self.encoder.getWidthOfLayout() - 1:
            x_right = self.encoder.getWidthOfLayout() - 2

        y_down = pos_pacman[1] - 2 * STEPS
        if y_down < 1:
            y_down = 1

        y_above = pos_pacman[1] + 2 * STEPS
        if y_above >= self.encoder.getHeightOfLayout() - 1:
            y_above = self.encoder.getHeightOfLayout() - 2

        return [x_left, x_right, y_down, y_above]


    def computeShield(self, state, ghost_table):

        self.prismStr = self.encoder.encodeModel(state, ghost_table)

        self.shield = []

        start_total_time = time.time()
        self.counter = 1
        self.start = time.time()

        for init_pacman in self.encoder.getRelevantCrossings():
            for next_dir_pacman in range(0, 4):
                if not self.encoder.isWall(self.encoder.getNextPosition(init_pacman, next_dir_pacman)):
                    window = self.returnWindowAroundPacman(init_pacman)
                    for init_x_ghost in range(window[0], window[1] + 1):
                        for init_y_ghost in range(window[2], window[3] + 1):
                            for ghost_dir in range(0, 4):
                                local_copy_of_prismStr = (self.prismStr + '.')[:-1]
                                self.computeShieldEntry(init_pacman, next_dir_pacman, [init_x_ghost, init_y_ghost], ghost_dir,
                                                        local_copy_of_prismStr)

        end_total_time = time.time()
        print("Total time needed to create the Shield:", end_total_time - start_total_time)

        return self.shield

    def computeProbabilityToGetEaten(self, init_pacman, dir_pacman, init_ghost, dir_ghost, prismStr, path=None):

        prismStr = prismStr.replace("xP : [1..xSize] init x", "xP : [1..xSize] init " + str(init_pacman[0]))
        prismStr = prismStr.replace("yP : [1..ySize] init x", "yP : [1..ySize] init " + str(init_pacman[1]))

        if not USE_CORRIDOR_ENCODING:
            prismStr = prismStr.replace("dP : [0 .. 3] init x", "dP : [0 .. 3] init " + str(dir_pacman))

        prismStr = prismStr.replace("xG0 : [0..xSize] init x",
                                    "xG0 : [0..xSize] init " + str(init_ghost[0]))
        prismStr = prismStr.replace("yG0 : [0..ySize] init x",
                                    "yG0 : [0..ySize] init " + str(init_ghost[1]))
        prismStr = prismStr.replace("dG0 : [0..3] init x",
                                    "dG0 : [0..3] init " + str(dir_ghost))

        if USE_CORRIDOR_ENCODING:
            assert (path != None)
            path_str = ""
            for i in range(0, len(path) - 1):
                assert (not self.encoder.isWall(path[i]))
                path_str += self.encoder.encodePacmanStatement(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1]) + "\n"
            prismStr = prismStr.replace("INSERT CURRENT PATH HERE", path_str)

        res = self.invokeStorm(prismStr)

        # if self.invokeStormToCheckDeadlock(prismStr):
        #    print("Deadlock at ", init_pacman, dir_pacman, init_ghosts, dir_ghosts)
        #    assert(False)

        return res


    def invokeStorm(self, mdpprog):

        # write program to RAM
        # print("writing prism program to RAM")
        temp_name = next(tempfile._get_candidate_names())
        file_name = "/dev/shm/prism-" + temp_name + ".nm"
        text_file = open(file_name, "w")
        text_file.write(mdpprog)
        text_file.close()

        # read program from RAM
        # print("parse prism program from RAM")
        program = stormpy.parse_prism_program(file_name)

        # print("parse properties")
        # prop = "Pmin=? [ F<=" + str((self.num_ghosts+1)*STEPS-1) +" \"crash\" ]"
        prop = "Pmin=? [ F \"crash\" ]"
        properties = stormpy.parse_properties_for_prism_program(prop, program, None)

        # print("Build Model")
        start = time.time()
        model = stormpy.build_model(program, properties)
        initial_state = model.initial_states[0]
        end = time.time()
        # print(end-start)

        result = stormpy.model_checking(model, properties[0])
        # print(result.at(initial_state))

        #os.remove(file_name)

        return result.at(initial_state)

    def invokeStormToCheckDeadlock(self, mdpprog):

        # write program to RAM
        # print("writing prism program to RAM")
        temp_name = next(tempfile._get_candidate_names())
        file_name = "/dev/shm/prism-" + temp_name + ".nm"
        text_file = open(file_name, "w")
        text_file.write(mdpprog)
        text_file.close()

        # read program from RAM
        # print("parse prism program from RAM")
        program = stormpy.parse_prism_program(file_name)

        # print("parse properties")
        # prop = "Pmin=? [ F<=" + str((self.num_ghosts+1)*STEPS-1) +" \"crash\" ]"
        prop = "Pmin=? [ F \"deadlock\" ]"
        properties = stormpy.parse_properties_for_prism_program(prop, program, None)

        # print("Build Model")
        start = time.time()
        model = stormpy.build_model(program, properties)
        initial_state = model.initial_states[0]
        end = time.time()
        # print(end-start)

        # print("model checking")
        result = stormpy.model_checking(model, properties[0])
        # print(result.at(initial_state))

        if result.at(initial_state) != 0:
            assert (False)

        os.remove(file_name)

        if result.at(initial_state) != 0:
            return True
        else:
            return False

