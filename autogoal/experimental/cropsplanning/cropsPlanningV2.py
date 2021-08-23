from myTree import myTree
import numpy
from sortedcontainers import SortedList
import math
import abc
from autogoal.kb import SemanticType


class Algorithm(SemanticType):
    pass


class AlgorithmStructure(Algorithm):
    @abc.abstractmethod
    def planning(self, data):
        pass

    @abc.abstractmethod
    def maxRoad(self):
        pass

    @classmethod
    def _match(cls, x) -> bool:
        return issubclass(x, Filter)


class Filter(abc.ABC):
    pass


class cropsFilter(Filter):
    def __init__(self, crops, value_of_decisions, time_of_decicions,
                 crop_rotation, planning_time, number_of_neurons):

        self._value_of_decisions = value_of_decisions
        self._time_of_decicions = time_of_decicions
        self._current_time = 0
        self._crop_rotation = crop_rotation
        self._time_rotatiom = []
        self._planning_time = planning_time
        self._neighboring_crops = []
        self._harmful_crops = []
        self._id_crops = SortedList()
        self._id_mul = 1
        self._crops_state = []
        self._cultivation_time = []
        self._time_state = []
        self._sorted_field = SortedList()
        self._desicions = {}
        self._prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                       103, 107,
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
                       223, 227,
                       229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
                       347, 349,
                       353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
                       463, 467,
                       479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
                       607, 613, 617,
                       619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743,
                       751, 757, 761,
                       769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887,
                       907, 911, 919,
                       929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039,
                       1049, 1051,
                       1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171,
                       1181, 1187,
                       1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297,
                       1301, 1303,
                       1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447,
                       1451, 1453,
                       1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567,
                       1571, 1579,
                       1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697,
                       1699, 1709,
                       1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847,
                       1861, 1867,
                       1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993]
        self.__start(crops)

        self.tree_size = number_of_neurons * (len(value_of_decisions) + 3)
        self._myTree = myTree(self.tree_size, len(value_of_decisions), 3)

    def __start(self, crops):
        for i in range(len(self._value_of_decisions)):
            self._id_crops.add(self._prime[i])
            self._id_mul *= self._prime[i]
            self._harmful_crops.append([])
            self._neighboring_crops.append([])

        for i in range(crops):
            self._crops_state.append(self._id_mul)
            self._sorted_field.add([0, i, 0])
            self._cultivation_time.append(0)
            rotation = []
            for j in range(len(self._value_of_decisions)):
                rotation.append(0)
            self._time_rotatiom.append(rotation)

        self._desicions[0] = SortedList()

    def decision(self, desicions, max_value_position, state_pos, tree_pos, epsilon):
        if max_value_position == -1 or epsilon < numpy.random.random():
            while (True):

                desicion_pos = desicions[numpy.random.randint(0, len(desicions))]

                if self._myTree.data_set[tree_pos + desicion_pos + 1] == -1:
                    if self._validation(state_pos, self._id_crops[desicion_pos]):
                        return desicion_pos, False
                    else:
                        return desicion_pos, True

                else:
                    return desicion_pos, False
        return max_value_position, False

    def _validation(self, pos, id_crop):
        if math.ceil(self._crops_state[pos] % id_crop) != 0:
            return False
        return True

    def backUpdate(self, pos, desicion, time):
        self._crops_state[pos] = self._crops_state[pos] / self._id_crops[desicion]
        self._cultivation_time[pos] = time + self._time_of_decicions[desicion]

        neighboring_crops = self._neighboring_crops[desicion]
        for i in range(len(neighboring_crops)):
            for j in range(len(self._harmful_crops[desicion])):
                self._crops_state[neighboring_crops[i]] /= self._id_crops[self._harmful_crops[desicion][j]]

        self._time_rotatiom[pos][desicion] = self._crop_rotation[desicion] + self._cultivation_time[pos]

    def frontUpdate(self, pos, time, desicion):
        if time != 0:
            neighboring_crops = self._neighboring_crops[desicion]
            for i in range(len(neighboring_crops)):
                for j in range(len(self._harmful_crops[desicion])):
                    self._crops_state[neighboring_crops[i]] *= self._id_crops[self._harmful_crops[desicion][j]]

            for i in range(len(self._time_rotatiom[pos])):
                if self._time_rotatiom[pos][i] != 0 and self._time_rotatiom[pos][i] <= time:
                    self._crops_state[pos] *= self._id_crops[i]

    def planning(self, epsilon):

        for i in range(len(self._value_of_decisions)):
            self._desicions[0].add(i)
        while True:
            way = [0]
            pos = 0
            planificatiom_value = 0
            if self._myTree.isEnty(0):
                return 0
            while True:
                next_node = self._myTree._data_set_position_occupied + len(self._value_of_decisions) + 3
                if next_node > self.tree_size or len(self._desicions[0]) == 0:
                    return 0

                crops = self._sorted_field[0]
                desicions: SortedList = self._desicions[pos]

                time = crops[0]
                self.frontUpdate(crops[1], time, crops[2])

                max_value_position = self._myTree.maxValuePosition(pos)

                if max_value_position != -1 and not desicions.__contains__(max_value_position):
                    max_value_position = -1

                myPosDecision = self.decision(desicions, max_value_position, crops[1], pos, epsilon)

                way.append(myPosDecision[0])
                if myPosDecision[1]:
                    is_right = False
                    break

                planificatiom_value += self._value_of_decisions[myPosDecision[0]]
                self.backUpdate(crops[1], myPosDecision[0], time)
                self._sorted_field.discard(crops)

                crops[0] += self._time_of_decicions[myPosDecision[0]]
                crops[2] = myPosDecision[0]

                if self._planning_time > crops[0]:
                    self._sorted_field.add(crops)

                if len(self._sorted_field) == 0:
                    is_right = True
                    break

                pos = self._myTree.nextNode(pos, myPosDecision[0] + 1, False)
                if pos == -1:
                    return 0
                way.append(pos)
                if self._desicions.get(pos) is None:

                    self._desicions[pos] = SortedList()
                    for i in range(len(self._value_of_decisions)):
                        self._desicions[pos].add(i)

            self._sorted_field = SortedList()
            self._time_rotatiom = []
            for i in range(len(self._crops_state)):
                self._sorted_field.add([0, i, 0])
                self._crops_state[i] = self._id_mul
                self._cultivation_time[i] = 0
                rotation = []
                for j in range(len(self._value_of_decisions)):
                    rotation.append(0)
                self._time_rotatiom.append(rotation)

            change_number_of_children = True
            change_max_value = is_right
            for i in range(int(len(way) / 2)):
                desicions_pos = way[len(way) - 1 - (2 * (i))]
                tree_pos = way[len(way) - 1 - (2 * (i)) - 1]

                if change_number_of_children:
                    self._desicions[tree_pos].discard(desicions_pos)
                    change_number_of_children = self._myTree.lostChildren(tree_pos)

                if change_max_value:
                    change_max_value = self._myTree.updateNode(tree_pos, desicions_pos, planificatiom_value)

                if not change_max_value and not change_number_of_children:
                    break
            way.clear()

    def maxRoad(self):
        return self._myTree.maxRoad()


























class Environment(SemanticType):






























    pass


class GameStructure(Environment):
    """The Game structure defines an Environment for 2-Players Game board
       with perfect information available. This should match our GameInterface
       which allow the implementatio of arbitrary games.
    Args:
        Environment ([type]): [description]
    """

    @classmethod
    def _match(cls, x) -> bool:
        return issubclass(x, Game)


class Game(abc.ABC):
    pass


class TicTacToeGame(Game):
    pass


if __name__ == "__main__":
    crops = 3
    planning_time = 10
    epsilon = 1 - (0.1 + 0.1 / 4)
    number_of_neurons = 100

    value_of_decisions = [1, 2]
    time_of_decicions = [10, 10]
    crop_rotation = [0, 0]

    x = cropsFilter(crops, value_of_decisions, time_of_decicions,
                    crop_rotation, planning_time, number_of_neurons)

    x.planning(epsilon)

    print(x.maxRoad(), x._myTree.data_set)
