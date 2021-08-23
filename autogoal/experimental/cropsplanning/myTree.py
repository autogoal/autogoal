# from numba import jitclass, types, typed

# spec = [('_data_set', types.ListType(types.int64)),
#         ('_data_set_position_occupied', types.int64),
#         ('_number_of_children', types.int64),
#         ('_node_data_size', types.int64)]


# @jitclass(spec)
class myTree:
    def __init__(self, n_size, number_of_children, node_data_size):
        self._data_set = []
        for i in range(n_size):
            self._data_set.append(-1)
        self._data_set[0] = -55

        self._data_set[number_of_children + 3] = number_of_children
        self._data_set_position_occupied = number_of_children + node_data_size + 1
        self._number_of_children = number_of_children
        self._node_data_size = node_data_size

    def addNode(self, myId, fathers_position):
        if self._data_set_position_occupied + self._number_of_children + 4 <= len(self._data_set):

            data_set_position_occupied = self._data_set_position_occupied
            new_data_set_position_occupied = data_set_position_occupied + self._number_of_children + 4

            self._data_set[fathers_position] = data_set_position_occupied
            self._data_set[data_set_position_occupied] = myId

            if self._data_set[data_set_position_occupied + self._number_of_children + 3] != -1000:
                self._data_set[data_set_position_occupied + self._number_of_children + 3] \
                    = self._number_of_children
            self._data_set_position_occupied = new_data_set_position_occupied
            return False
        return True

    def addLeaf(self, id, fathers_position):
        if self._data_set_position_occupied + 2 <= len(self._data_set):
            self._data_set[fathers_position] = self._data_set_position_occupied
            self._data_set[self._data_set_position_occupied] = id
            if self._data_set[self._data_set_position_occupied + 1] == -1:
                self._data_set[self._data_set_position_occupied + 1] = 0
            self._data_set_position_occupied += 2
            return 1
        return 0

    def nextPos(self, nodePos, sonPos):
        if nodePos + sonPos < len(self._data_set):
            return self._data_set[nodePos + sonPos]
        return None

    def nextNode(self, nodePos, sonPos, isLeaf):
        if self._data_set[nodePos + sonPos] == -1:
            is_not_right = False
            father_position = self._data_set_position_occupied
            if isLeaf:
                is_not_right = self.addLeaf(-99, nodePos + sonPos)
            else:
                is_not_right = self.addNode(-77, nodePos + sonPos)
            if is_not_right:
                return -1
            return father_position
        else:
            return self._data_set[nodePos + sonPos]

    def updateDataSet(self, value, calculated_path):
        calculated_path_pos = len(calculated_path) - 1
        for x in range(int(len(calculated_path) / 2)):

            data_set_pos = calculated_path[calculated_path_pos]
            father_pos = calculated_path[calculated_path_pos - 2]

            if self._data_set[data_set_pos] == -99:
                self._data_set[data_set_pos + 1] = value
                self._data_set[father_pos + self._number_of_children + 3] -= 1
            elif self._data_set[data_set_pos + self._number_of_children + 1] < value:
                self._data_set[data_set_pos + self._number_of_children + 1] = value
                self._data_set[data_set_pos + self._number_of_children + 2] \
                    = calculated_path[calculated_path_pos + 1]

            if calculated_path_pos != int(len(calculated_path)) - 1 and \
                    self._data_set[data_set_pos + self._number_of_children + 3] == 0:
                self._data_set[data_set_pos + self._number_of_children + 3] = -100
                if calculated_path_pos != 1:
                    self._data_set[father_pos + self._number_of_children + 3] -= 1

            calculated_path_pos -= 2

        if self._data_set[self._number_of_children + 1] < value:
            self._data_set[self._number_of_children + 1] = value
            self._data_set[self._number_of_children + 2] = calculated_path[0]

        if self._data_set[calculated_path[1] + self._number_of_children + 3] == -100:
            self._data_set[self._number_of_children + 3] -= 1

    def maxValuePosition(self, pos):
        maxValuePosition = pos + self._number_of_children + 1
        return self._data_set[maxValuePosition]

    def isEnty(self, pos):
        return self._data_set[pos + 3 + self._number_of_children] == -100 or self._data_set[
            pos + 3 + self._number_of_children] == 0

    def lostChildren(self, pos):
        self._data_set[pos + 3 + self._number_of_children] -= 1

        if self._data_set[pos + 3 + self._number_of_children] == 0:
            return True

        return False

    def updateNode(self, pos_tree, desicions_pos, value):
        if self._data_set[pos_tree + self._number_of_children + 2] < value:
            self._data_set[pos_tree + self._number_of_children + 2] = value
            self._data_set[pos_tree + self._number_of_children + 1] = desicions_pos
            return True
        return False

    def maxRoad(self):
        result = []
        pos = 0
        while pos != -1:
            posicion = self._data_set[pos + self._number_of_children + 1]
            result.append(posicion)
            pos = self._data_set[pos + posicion + 1]
        return result
    @property
    def data_set(self):
        return self._data_set

    @property
    def data_set_position_occupied(self):
        return self._data_set_position_occupied

