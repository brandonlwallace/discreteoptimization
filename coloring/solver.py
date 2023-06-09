#!/usr/bin/python
# -*- coding: utf-8 -*-


from ortools.sat.python import cp_model
import sys

def solve_it(input_data):
    # parse the input
    lines = input_data.strip().split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # create a CP-SAT model
    model = cp_model.CpModel()

    # create variables for node colors
    colors = [model.NewIntVar(0, node_count - 1, f'color_{i}') for i in range(node_count)]

    # add constraints for adjacent nodes to have different colors
    for edge in edges:
        model.Add(colors[edge[0]] != colors[edge[1]])

    # minimize the number of colors used
    obj = model.NewIntVar(0, node_count, 'obj')
    model.AddMaxEquality(obj, colors)
    model.Minimize(obj)

    # create a solver and set a time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180  # Set the time limit 

    # solve the model
    status = solver.Solve(model)

    # prepare the solution in the specified output format
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = [solver.Value(color) for color in colors]
        num_colors = max(solution) + 1
        output_data = f'{num_colors} 1\n'
        output_data += ' '.join(map(str, solution))
    else:
        output_data = 'approach not working; rework solution'

    return output_data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')





'''
# INITIAL TRIVAL SOLUTION 

def solve_it(input_data):
    # Parse the input
    lines = input_data.strip().split('\n')
    node_count, edge_count = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        u, v = map(int, line.split())
        edges.append((u, v))

    # Sort nodes by degree (descending order)
    nodes = list(range(node_count))
    nodes.sort(key=lambda x: sum(u == x or v == x for u, v in edges), reverse=True)

    # Assign colors to nodes
    colors = [0] * node_count
    for node in nodes:
        # Find the smallest color that can be assigned to the current node
        available_colors = set(range(node_count))
        for u, v in edges:
            if u == node and colors[v] in available_colors:
                available_colors.remove(colors[v])
            elif v == node and colors[u] in available_colors:
                available_colors.remove(colors[u])

        # Assign the smallest available color to the current node
        colors[node] = min(available_colors)

    # Prepare the solution in the specified output format
    obj = max(colors) + 1
    opt = 1
    output_data = f"{obj} {opt}\n"
    output_data += " ".join(map(str, colors))

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
