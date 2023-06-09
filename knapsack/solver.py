#!/usr/bin/python
# -*- coding: utf-8 -*-

# FINAL SOLUTION 

from collections import namedtuple
import numpy as np
import time

Item = namedtuple("Item", ['index', 'value', 'weight'])

class Node:
    def __init__(self, level, value, weight, taken):
        self.level = level
        self.value = value
        self.weight = weight
        self.taken = taken

def solve_it(input_data):
    # Parse the input
    lines = input_data.strip().split('\n')
    item_count, capacity = map(int, lines[0].split())

    items = []
    for line in lines[1:]:
        value, weight = map(int, line.split())
        items.append(Item(len(items), value, weight))

    # Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda x: x.value / x.weight, reverse=True)

    # Initialize variables
    best_value = 0
    best_taken = np.zeros(item_count, dtype=int)
    queue = []

    # Set the start time
    start_time = time.time()

    # Create the root node
    root = Node(level=0, value=0, weight=0, taken=np.zeros(item_count, dtype=int))
    queue.append(root)

    # Explore the search tree using Branch and Bound
    while queue:
        # Check if the time limit has been reached
        elapsed_time = time.time() - start_time
        if elapsed_time > 120:  # 2 minutes = 120 seconds
            break

        node = queue.pop(0)

        if node.level < item_count:
            # Explore the left child (take the item)
            left_child = Node(
                level=node.level + 1,
                value=node.value + items[node.level].value,
                weight=node.weight + items[node.level].weight,
                taken=node.taken.copy(),
            )
            left_child.taken[items[node.level].index] = 1

            if left_child.weight <= capacity:
                if left_child.value > best_value:
                    best_value = left_child.value
                    best_taken = left_child.taken
                if left_child.value + bound(node.level + 1, item_count, items, capacity - left_child.weight) > best_value:
                    queue.append(left_child)

            # Explore the right child (do not take the item)
            right_child = Node(
                level=node.level + 1,
                value=node.value,
                weight=node.weight,
                taken=node.taken.copy(),
            )

            if right_child.value + bound(node.level + 1, item_count, items, capacity - right_child.weight) > best_value:
                queue.append(right_child)

    # Prepare the solution in the specified output format
    output_data = str(int(best_value)) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, best_taken))

    return output_data

def bound(level, item_count, items, capacity):
    """
    Compute the upper bound for the remaining items using the fractional knapsack approach.
    """
    value_bound = 0
    weight = 0

    for i in range(level, item_count):
        if weight + items[i].weight <= capacity:
            value_bound += items[i].value
            weight += items[i].weight
        else:
            remaining_capacity = capacity - weight
            value_bound += items[i].value * (remaining_capacity / items[i].weight)
            break

    return value_bound


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory.')








'''

OLD SOLUTION #2 USING BRANCH AND BOUND

import numpy as np
from collections import namedtuple
import time
import heapq

Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # Sort items in descending order of value-to-weight ratio
    items.sort(key=lambda x: x.value / x.weight, reverse=True)

    # Convert items to NumPy arrays for efficient computations
    values = np.array([item.value for item in items])
    weights = np.array([item.weight for item in items])

    # Initialize variables
    best_value = 0
    best_taken = np.zeros(item_count, dtype=int)

    # Define a node in the search tree
    class Node:
        def __init__(self, level, value, weight, taken):
            self.level = level
            self.value = value
            self.weight = weight
            self.taken = taken

        def bound(self):
            remaining_items = item_count - self.level - 1
            remaining_values = values[self.level+1:]
            remaining_weights = weights[self.level+1:]
            if self.weight >= capacity:
                return 0
            bound_value = self.value
            bound_weight = self.weight
            sorted_indices = np.argsort(remaining_values / remaining_weights)[::-1]
            for i in sorted_indices:
                if bound_weight + remaining_weights[i] <= capacity:
                    bound_value += remaining_values[i]
                    bound_weight += remaining_weights[i]
                else:
                    bound_value += remaining_values[i] / remaining_weights[i] * (capacity - bound_weight)
                    break
            return bound_value

        def __lt__(self, other):
            return self.bound() < other.bound()

    # Initialize the priority queue
    priority_queue = []

    # Create the root node
    root = Node(-1, 0, 0, np.zeros(item_count, dtype=int))

    # Add the root node to the priority queue
    heapq.heappush(priority_queue, root)

    # Branch and bound algorithm with time limit
    start_time = time.time()
    time_limit = 60  # 1 minute

    while priority_queue and time.time() - start_time < time_limit:
        node = heapq.heappop(priority_queue)

        if node.bound() < best_value:
            continue

        if node.level == item_count - 1:
            if node.value > best_value:
                best_value = node.value
                best_taken = node.taken
            continue

        # Explore the left child (with the item included)
        left_child = Node(node.level + 1,
                        node.value + values[node.level + 1],
                        node.weight + weights[node.level + 1],
                        node.taken.copy())
        left_child.taken[node.level + 1] = 1

        if left_child.weight <= capacity and left_child.value > best_value:
            best_value = left_child.value
            best_taken = left_child.taken

        if left_child.bound() >= best_value:
            heapq.heappush(priority_queue, left_child)

        # Explore the right child (with the item excluded)
        right_child = Node(node.level + 1, node.value, node.weight, node.taken.copy())

        if right_child.bound() >= best_value:
            heapq.heappush(priority_queue, right_child)

    # Prepare the solution in the specified output format
    obj = sum(best_taken)
    # Assume optimal solution 
    opt = 1
    output_data = f"{obj} {opt}\n"
    output_data += " ".join(map(str, best_taken))

    return output_data

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (e.g. python solver.py ./data/ks_4_0)')


OLD #1 SOLUTION USING DP 


from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # parse the input
    lines = input_data.split("\n")
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    # set an empty list
    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # initialize the dynamic programming table from the course video
    table = [[0] * (capacity+1) for _ in range(item_count+1)]

    # fill the table using DP
    for i in range(1, item_count+1):
        item = items[i-1]
        for w in range(1, capacity+1):
            if item.weight > w:
                table[i][w] = table[i-1][w]
            else:
                table[i][w] = max(table[i-1][w], table[i-1][w-item.weight] + item.value)

    # find the optimal solution and its value
    value = table[-1][-1]
    taken = [0] * item_count
    w = capacity

    for i in range(item_count, 0, -1):
        if value <= 0:
            break
        if value == table[i-1][w]:
            continue
        else:
            taken[i-1] = 1
            value -= items[i-1].value
            w -= items[i-1].weight

    # prepare the solution in the specified output format
    obj = sum(taken)
    # assume the solution is optimal based on code
    opt = 1
    output_data = f"{obj} {opt}\n{' '.join(map(str, taken))}"
    # return final answer 
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


'''

