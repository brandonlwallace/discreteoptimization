#!/usr/bin/python
# -*- coding: utf-8 -*-

# Final solution using SA, 2-opt, and an initial greedy solution as the starting point

import math
import numpy as np
from collections import namedtuple
import random
import sys
import time

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    # utilzing np.linalg.norm to calculate the Euclidean norm of the vector, which is equivalent to the distance between the two points
    # this is more efficent; avoid looping over individual coordinates and perform the distance calculation.
    vector = np.array([point1.x - point2.x, point1.y - point2.y])
    return np.linalg.norm(vector)

def calculate_distance(points, solution):
    distance = length(points[solution[-1]], points[solution[0]])
    for index in range(len(solution) - 1):
        distance += length(points[solution[index]], points[solution[index+1]])
    return distance

def greedy_tour_builder(points):
    num_nodes = len(points)
    visited = [False] * num_nodes
    solution = [None] * num_nodes
    solution[0] = 0  # Start with the first node
    visited[0] = True

    for i in range(1, num_nodes):
        current_node = solution[i - 1]
        min_distance = -1e9  # Set to a low value
        next_node = None

        for j in range(num_nodes):
            if not visited[j]:
                distance = length(points[current_node], points[j])
                if distance > min_distance:  # Check for a closer node
                    min_distance = distance
                    next_node = j

        solution[i] = next_node
        visited[next_node] = True

    return solution

def two_opt_swap(solution, i, j):
    new_solution = solution.copy()
    new_solution[i:j+1] = reversed(solution[i:j+1])
    return new_solution

def simulated_annealing(points, initial_solution, initial_temperature, cooling_factor, num_iterations, time_limit):
    start_time = time.time()
    current_solution = initial_solution
    best_solution = current_solution
    current_distance = calculate_distance(points, current_solution)
    best_distance = current_distance
    temperature = initial_temperature
    num_moves = 0

    while time.time() - start_time < time_limit:
        for _ in range(num_iterations):
            i, j = random.sample(range(len(points)), 2)
            new_solution = two_opt_swap(current_solution, i, j)
            new_distance = calculate_distance(points, new_solution)
            delta = new_distance - current_distance

            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_solution = new_solution
                current_distance = new_distance
                num_moves += 1

                if current_distance < best_distance:
                    best_solution = current_solution
                    best_distance = current_distance

            temperature *= cooling_factor

    return best_solution, best_distance, num_moves

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    start_time = time.time()  # Track the start time
    time_limit = 120  # Set the time limit to 2 minutes

    while time.time() - start_time < time_limit:

        nodeCount = int(lines[0])

        points = []
        for i in range(1, nodeCount+1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))

        # Build initial greedy solution
        initial_solution = greedy_tour_builder(points)

        # Set SA parameters
        initial_temperature = 10000.0
        cooling_factor = 0.75
        num_iterations = 10000
        time_limit = 60  # in seconds

        # Run multiple restarts of SA
        best_solution = None
        best_distance = float('inf')
        num_moves = 0
        num_restarts = 10

        for _ in range(num_restarts):
            sa_solution, sa_distance, sa_moves = simulated_annealing(points, initial_solution, initial_temperature,
                                                                    cooling_factor, num_iterations, time_limit)

            if sa_distance < best_distance:
                best_solution = sa_solution
                best_distance = sa_distance
                num_moves = sa_moves

        # Perform a single round of greedy 2-opt
        for i in range(len(points)):
            for j in range(i + 2, len(points)):
                new_solution = two_opt_swap(best_solution, i, j)
                new_distance = calculate_distance(points, new_solution)
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance

        # Prepare the solution in the specified output format
        output_data = '%.2f' % best_distance + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, best_solution))

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')



'''

Version 1 attempt at solution using Google OR Tools

import math
import sys
import time
from collections import namedtuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def create_data_model(points):
    data = {}
    data['distance_matrix'] = [
        [length(points[i], points[j]) for j in range(len(points))] for i in range(len(points))
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def apply_2_opt(solution, distance_matrix):
    best_route = solution[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j - i == 1:
                    continue
                new_route = best_route[:]
                new_route[i:j] = best_route[j - 1:i - 1:-1]
                new_length = calculate_route_length(new_route, distance_matrix)
                if new_length < calculate_route_length(best_route, distance_matrix):
                    best_route = new_route
                    improved = True
        solution = best_route
    return solution

def calculate_route_length(route, distance_matrix):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i]][route[i + 1]]
    length += distance_matrix[route[-1]][route[0]]
    return length

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # Create the data model
    data = create_data_model(points)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes
        return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set maximum time allowed for the solver to find a solution
    time_limit_seconds = 10
    start_time = time.time()

    # Solve the problem
    solution = None
    while time.time() - start_time < time_limit_seconds:
        solution = routing.Solve()
        if solution:
            break

    # Prepare the solution in the specified output format
    if solution:
        obj = solution.ObjectiveValue()
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        
        # Apply 2-opt optimization to improve the tour length
        distance_matrix = data['distance_matrix']
        route = apply_2_opt(route, distance_matrix)

        output_data = f"{obj} 1\n"
        output_data += ' '.join(map(str, route))
    else:
        output_data = "No solution found."

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory.')





'''