#!/usr/bin/python
# -*- coding: utf-8 -*-


# Final Solution Using Gurobi Solver

from collections import namedtuple
import math
from gurobipy import *

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])



def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):

    # Parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # Build the model
    m = Model()

    # Add decision variables
    x = {}
    y = {}
    d = {}
    for j in range(facility_count):
        x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)
    for i in range(customer_count):
        for j in range(facility_count):
            y[(i, j)] = m.addVar(vtype=GRB.BINARY, name="y%d,%d" % (i, j))
            d[(i, j)] = length(customers[i].location, facilities[j].location)
    m.update()

    # Set objective function
    m.setObjective(quicksum(
        facilities[j].setup_cost * x[j] + quicksum(d[(i, j)] * y[(i, j)] for i in range(customer_count)) for j in
        range(facility_count)), GRB.MINIMIZE)

    # Add constraints
    for i in range(customer_count):
        m.addConstr(quicksum(y[(i, j)] for j in range(facility_count)) == 1)
    for i in range(customer_count):
        for j in range(facility_count):
            m.addConstr(y[(i, j)] <= x[j])

    for j in facilities:
        m.addConstr(quicksum(y[(i.index, j.index)] * i.demand for i in customers) <= j.capacity)

    # Optimize the model
    m.optimize()

    # Get the solution
    obj = m.objVal
    solution = []
    for c in customers:
        for f in facilities:
            if y[(c.index, f.index)].X == 1:
                solution.append(f.index)

    # Prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')


'''

# First Solution with Google OR Tools

from collections import namedtuple
import math
import sys
import numpy as np
import time
from ortools.linear_solver import pywraplp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
    
    # Pre-compute distance matrix
    distance_matrix = np.zeros((facility_count, customer_count))
    for i in range(facility_count):
        for j in range(customer_count):
            distance_matrix[i, j] = length(facilities[i].location, customers[j].location)

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Create variables
    x = {}
    for i in range(facility_count):
        for j in range(customer_count):
            x[i, j] = solver.BoolVar('x[%i,%i]' % (i, j))

    # Create constraints
    # Capacity constraints
    for i in range(facility_count):
        solver.Add(sum(x[i, j] * customers[j].demand for j in range(customer_count)) <= facilities[i].capacity)

    # Assignment constraints
    for j in range(customer_count):
        solver.Add(sum(x[i, j] for i in range(facility_count)) == 1)

    # Objective function
    objective = solver.Objective()
    for i in range(facility_count):
        for j in range(customer_count):
            objective.SetCoefficient(x[i, j], distance_matrix[i, j])
    objective.SetMinimization()

    # Solve the problem
    solver.Solve()

    # Get the solution
    solution = np.zeros(customer_count, dtype=np.int32)
    for j in range(customer_count):
        for i in range(facility_count):
            if x[i, j].solution_value() == 1:
                solution[j] = i
                break

    # Calculate the objective value
    obj = sum(facilities[i].setup_cost for i in range(facility_count))
    for j in range(customer_count):
        obj += distance_matrix[solution[j], j]

    # Prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')


'''