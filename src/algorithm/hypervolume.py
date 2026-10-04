"""
Hypervolume indicator (dimension-sweep algorithm).

Based on:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez.
    "An improved dimension-sweep algorithm for the hypervolume indicator."
    IEEE Congress on Evolutionary Computation, 2006.

Minimisation is assumed throughout.  Before computation the front is shifted
so that the reference point becomes [0, ..., 0].

Classes
-------
HyperVolume   - Public interface: compute(front) -> float
_MultiList    - Internal doubly-linked multi-list used by the sweep algorithm
"""


class HyperVolume:
    """Compute the hypervolume dominated by a non-dominated front."""

    def __init__(self, referencePoint):
        self.referencePoint = referencePoint
        self.list = []

    def compute(self, front):
        """Return the hypervolume dominated by *front* relative to the reference point."""

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        referencePoint = self.referencePoint
        dimensions = len(referencePoint)

        # All points in the front are considered relevant.
        relevantPoints = front

        if any(referencePoint):
            # Shift so that referencePoint == [0, ..., 0]; avoids explicit
            # subtraction inside the recursive sweep.
            relevantPoints -= referencePoint

        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        return self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)

    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive dimension-sweep hypervolume computation.

        The reference point is implicitly [0, ..., 0] after the shift in compute().
        """
        hvol = 0.0
        sentinel = self.list.sentinel

        if length == 0:
            return hvol
        elif dimIndex == 0:
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (
                q.cargo[dimIndex] > bounds[dimIndex]
                or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]
            ):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = (
                    qPrevDimIndex.volume[dimIndex]
                    + qPrevDimIndex.area[dimIndex]
                    * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
                )
            else:
                qArea[0] = 1
                qArea[1 : dimIndex + 1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol

    def preProcess(self, front):
        """Build the _MultiList structure required for the sweep."""
        dimensions = len(self.referencePoint)
        nodeList = _MultiList(dimensions)
        nodes = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList

    def sortByDimension(self, nodes, i):
        """Sort *nodes* in-place by the i-th coordinate of their cargo point."""
        decorated = [(node.cargo[i], node) for node in nodes]
        decorated.sort()
        nodes[:] = [node for (_, node) in decorated]


class _MultiList:
    """Several doubly-linked lists that share common nodes.

    Every node belongs to all lists simultaneously, each with its own
    next/prev pointers.  This is the data structure required by the
    dimension-sweep HV algorithm.
    """

    class Node:
        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

        def __str__(self):
            return str(self.cargo)

        def __lt__(self, other):
            return all(self.cargo < other.cargo)

    def __init__(self, numberLists):
        self.numberLists = numberLists
        self.sentinel = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists

    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        return "\n".join(strings)

    def __len__(self):
        return self.numberLists

    def getLength(self, i):
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node

    def extend(self, nodes, index):
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            sentinel.prev[index] = node
            lastButOne.next[index] = node

    def remove(self, node, index, bounds):
        """Unlink *node* from all lists in [0, index) and update bounds."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """Re-link *node* into all lists in [0, index) at its original position."""
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
