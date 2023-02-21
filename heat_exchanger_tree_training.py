# The following import contains the data parsing function
# import linear_tree_data

# The following import contains the different formulation functions
# import linear_tree_formulations

import numpy as np
from sklearn.linear_model import *
from lineartree import LinearTreeClassifier, LinearTreeRegressor
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=4)
from pyomo.util.model_size import build_model_size_report

# Read the data from the csv file
data = np.genfromtxt('Data\\PHTSV_Table_HMAX_Adjusted.csv', delimiter=',')
data = data[1:-1, :]

# 2 D array containing Pressure and Enthalpy data
P_H = data[:, 0:2]

# Scale the pressure to bar and scale the enthalpy to kJ/mol
P_H[:, 0] = P_H[:, 0] / 1e5
P_H[:, 1] = P_H[:, 1] / 1000
minP = np.min(data[:, 0])
maxP = np.max(data[:, 0])
minH = np.min(data[:, 1])
maxH = np.max(data[:, 1])
print(minH)
print(maxH)
print(minP)
print(maxP)

# Vector containing Temperatures
T = data[:, 2]
minT = np.min(T)
maxT = np.max(T)

regr = LinearTreeRegressor(LinearRegression(), criterion='mse', max_bins=60,
                           min_samples_leaf=8, max_depth=8, min_impurity_decrease=0.1)
regr.fit(P_H, T)
T_hat = regr.predict(P_H)

plt.scatter(T, T_hat)
plt.show()

# print(np.sum(np.square(T - T_hat)))
#print(T- T_hat)
print(np.max(np.abs(T - T_hat)))
print(np.mean(np.square(T - T_hat)))
spts, lvs, th = linear_tree_data.parse_Tree_Data(regr, write_to_Pickle=True)

print(len(lvs))

"""
Below is some testing code I did on the large model. Can just load the pickle files
"""

# m = pyo.ConcreteModel()

# m.P = pyo.Var(bounds = (minP, maxP))
# m.H = pyo.Var(bounds = (minH, maxH))
# m.T = pyo.Var(bounds = (minT, maxT))

# m.obj = pyo.Objective(expr = m.T)

# m.b = linear_tree_formulations.miqp_formulation(spts, lvs, th)

# m.Pressure_inputlink = pyo.Constraint(expr = m.P == m.b.input_variable[0])
# m.Enthalpy_inputlink = pyo.Constraint(expr = m.H == m.b.input_variable[1])
# m.Temperature_outputlink = pyo.Constraint(expr = m.T == m.b.output_variable)

# m.P.fix(8.6047e6/1e5)
# m.T.fix(866)

# solver = pyo.SolverFactory('gurobi')
# results = solver.solve(m, tee=True)

# m.H.display()

###########################################
# YUFENG VISUALIZATION
###########################################
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv('Data\\PHTSV_Table_HMAX_Adjusted.csv')


def DT_boundary(data, boundary1, boundary2):
    # with open(file, 'rb') as f:
    #    data = pickle.load(f)
    plt.figure()
    for key in data:
        x1 = np.linspace(data[key]['bounds'][1][0], data[key]['bounds'][1][1])
        y1 = data[key]['bounds'][0][0] + x1 * 0
        plt.plot(x1, y1, 'black')
        x2 = np.linspace(data[key]['bounds'][1][0], data[key]['bounds'][1][1])
        y2 = data[key]['bounds'][0][1] + x2 * 0
        plt.plot(x2, y2, 'black')
        y3 = np.linspace(data[key]['bounds'][0][0], data[key]['bounds'][0][1])
        x3 = data[key]['bounds'][1][0] + x2 * 0
        plt.plot(x3, y3, 'black')
        y4 = np.linspace(data[key]['bounds'][0][0], data[key]['bounds'][0][1])
        x4 = data[key]['bounds'][1][1] + x3 * 0
        plt.plot(x4, y4, 'black')
        plt.xlim(boundary1[0], boundary1[1])
        plt.ylim(boundary2[0], boundary2[1])

    x = df['H(J/mol)'] / 1000
    x = np.array(x)
    x.shape = (100, 100)

    y = df['P(Pa)'] / 100000
    y = np.array(y)
    y.shape = (100, 100)

    z = df['T(K)']
    z = np.array(z)
    z.shape = (100, 100)

    plt.contourf(x, y, z, 1000)
    plt.ylabel('Pressure [bar]')
    plt.xlabel('Enthalpy [kJ/mol]')
    plt.colorbar(label='Temperature (K)')
    plt.show()

    plt.figure()
    plt.contourf(x, y, z, 1000)
    plt.ylabel('Pressure [bar]')
    plt.xlabel('Enthalpy [kJ/mol]')
    plt.colorbar(label='Temperature (K)')
    plt.xlim(boundary1[0], boundary1[1])
    plt.ylim(boundary2[0], boundary2[1])
    plt.show()

    plt.figure()
    for key in data:
        x1 = np.linspace(data[key]['bounds'][1][0], data[key]['bounds'][1][1])
        y1 = data[key]['bounds'][0][0] + x1 * 0
        plt.plot(x1, y1, 'black')
        x2 = np.linspace(data[key]['bounds'][1][0], data[key]['bounds'][1][1])
        y2 = data[key]['bounds'][0][1] + x2 * 0
        plt.plot(x2, y2, 'black')
        y3 = np.linspace(data[key]['bounds'][0][0], data[key]['bounds'][0][1])
        x3 = data[key]['bounds'][1][0] + x2 * 0
        plt.plot(x3, y3, 'black')
        y4 = np.linspace(data[key]['bounds'][0][0], data[key]['bounds'][0][1])
        x4 = data[key]['bounds'][1][1] + x3 * 0
        plt.plot(x4, y4, 'black')
        plt.xlim(boundary1[0], boundary1[1])
        plt.ylim(boundary2[0], boundary2[1])
    plt.ylabel('Pressure [bar]')
    plt.xlabel('Enthalpy [kJ/mol]')
    plt.title('')
    plt.show()


DT_boundary(lvs, [minH, maxH], [minP, maxP])
