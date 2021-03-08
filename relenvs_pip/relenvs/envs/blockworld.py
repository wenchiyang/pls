import numpy as np
import gym
import random




class BlockWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, N=3, episode_length=30, reward_end=100, reward_allowed_move=0, reward_impossible_move=-30, random=False):


        self.random = random
        self.reward_end = reward_end
        self.reward_allowed_move = reward_allowed_move
        self.reward_impossible_move = reward_impossible_move

        self.episode_length = episode_length
        self.N = N+1
        self.reset()
        _, self.target, _ = self.random_state()




    def step(self, action:str):
        f, t = action.split("move(")[1].split(",")
        t = t.replace(")", "")

        if "on(%s,%s)" % (f,t) not in self.state and "clear(%s)"%f in self.state and (t == "0" or "clear(%s)" % t in self.state):
            # You can move it

            # Delete from and Clear the one below
            oldr, oldc = self.state_dict[int(f)]
            self.matrix[oldr, oldc] = 0
            if oldr > 0:
                # not on ground
                freed = self.matrix[oldr-1, oldc]
                self.state.add("clear(%d)" % freed)
                self.state.remove("on(%s,%d)" % (f, freed))
            else:
                self.state.remove("on(%s,0)" % (f))



            # Put on top of t if is not the ground
            if t !='0' and "clear(%s)" % t in self.state:
                self.state.remove("clear(%s)" % t)
                tr, tc = self.state_dict[int(t)]
            else:
                tr = -1
                for k in range(self.N):
                    if self.matrix[0,k] == 0:
                        tc = k
                        break
            self.state.add("on(%s,%s)" % (f, t))

            self.matrix[tr+1, tc] = int(f)
            self.state_dict[int(f)] = tr+1, tc
            if self.check():
                return self.state, self.reward_end, True, {}
            return self.state, self.reward_allowed_move, False, {}
        else:
            return self.state, self.reward_impossible_move\
                , False, {}

    def check(self):
        if len(self.state) != len(self.target):
            return False
        for i in self.state:
            if i not in self.target:
                return False
        return True


    def reset(self):
        self.matrix, self.state, self.state_dict = self.random_state()
        return self.state


    def random_state(self):
        if not self.random:
            random.seed(0)
        matrix = np.zeros(shape=[self.N,self.N])
        state = set()
        state_dict = {}
        state.add('clear(0)')
        for i in range(1, self.N):
            c = random.randint(0, self.N - 1)
            r = 0
            while matrix[r, c] != 0:
                r += 1
            matrix[r, c] = i
            state_dict[i] = (r, c)
            if r == 0:
                state.add('on(%d,0)' % i)
            else:
                j = matrix[r - 1, c]
                state.remove('clear(%d)' % j)
                state.add('on(%d,%d)' % (i, j))
            state.add('clear(%d)' % i)

        return matrix, state, state_dict


    def render(self, mode='human', close=False):
        print(self.matrix)
        print(self.state)

#
# class MMEBlockWorld(gym.Env):
#
#     def __init__(self, N, episode_length, reward_end=100, reward_allowed_move=0, reward_impossible_move=-30):
#
#
#         self.reward_end = reward_end
#         self.reward_allowed_move = reward_allowed_move
#         self.reward_impossible_move = reward_impossible_move
#
#         self.episode_length = episode_length
#         self.bw = BlockWorld(N, episode_length, reward_end=reward_end,reward_allowed_move=reward_allowed_move,reward_impossible_move=reward_impossible_move)
#         self.N = N + 1 # + 1 for ground
#         self.A = ["move(%d,%d)" % (i,j) for i,j in product(range(1, N+1), range(0, N+1)) if i!=j]
#
#         d = mme.Domain("blocks", num_constants=self.N, constants=[str(s) for s in range(0, self.N)])
#         on = mme.Predicate("on",domains=(d,d))
#         clear = mme.Predicate("clear", domains = [d])
#
#         self.ontology = mme.Ontology(domains=[d], predicates=[on, clear])
#
#     def render(self, mode='human'):
#         # self.bw.render()
#         # input()
#         pass
#
#     def reset(self):
#         self._step=0
#         return self.__translate_state(self.bw.reset())
#
#     def __translate_state(self, s):
#         state = np.zeros([self.ontology.linear_size()], dtype=np.float32)
#         for atom in s:
#             state[ self.ontology.atom_string_to_id(atom+".")] = 1
#         return state
#
#     def step(self, action):
#         # print(self.A[action])
#         s, r, finish, _ = self.bw.step(self.A[action])
#         self._step+=1
#         if not finish and self._step > self.episode_length:
#             finish = True
#         # self.bw.render()
#         # print(r)
#         # input()
#         return self.__translate_state(s), r, finish, _
#


# if __name__ == '__main__':
#     b = MMEBlockWorld(5)
#     b.render()
#
#     while True:
#         f = input()
#         t = input()
#
#         action = "move(%s,%s)" % (f,t)
#         print(action)
#         print(b.step(b.A.index(action)))
#         print(b.bw.matrix)