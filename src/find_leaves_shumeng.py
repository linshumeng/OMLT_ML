import numpy as np
import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
pp = pprint.PrettyPrinter(indent=4)
import lightgbm as lgb
import pyomo.environ as pe

# Read the data from the csv file
data = np.genfromtxt('data/PHTSV_Table_HMAX_Adjusted.csv', delimiter=',')
data = data[1:, :]

# 2 D array containing Pressure and Enthalpy data
P_H = data[:, 0:2]

# Scale the pressure to bar and scale the enthalpy to kJ/mol
P_H[:, 0] = P_H[:, 0] / 1e5
P_H[:, 1] = P_H[:, 1] / 1000
minP = np.min(data[:, 0])
maxP = np.max(data[:, 0])
minH = np.min(data[:, 1])
maxH = np.max(data[:, 1])
# print(minH)
# print(maxH)
# print(minP)
# print(maxP)

# Vector containing Temperatures
T = data[:, 2]
minT = np.min(T)
maxT = np.max(T)

# Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    P_H, T, test_size=0.2, random_state=42)

lgb_dic = {}
train_data_linear = lgb.Dataset(
    X_train, label=y_train)
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.4,
    'num_leaves': 20,
    "verbosity": -1,
    'min_samples_leaf': 8,
    'max_bin': 60,
    'num_iterations': 10,
    'max_depth': 10,
}
model_linear = lgb.train(params, train_data_linear)
y_pred_linear = model_linear.predict(X_test)
print(
    f"Linear trees error: {round(mean_squared_error(y_test, y_pred_linear),3)}")

# Save and load onnx
from onnxmltools.convert.lightgbm.convert import convert
from skl2onnx.common.data_types import FloatTensorType

lgb_model = model_linear
float_tensor_type = FloatTensorType([None, lgb_model.num_feature()])
initial_types = [('float_input', float_tensor_type)]
onnx_model = convert(lgb_model,
                     initial_types=initial_types,
                     target_opset=8)

graph = onnx_model.graph

# Parse the onnx, default left true
# iterate through inputs of the graph
for input in graph.input:
    print(input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                n_input = d.dim_value
                print(d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                # unknown dimension with symbolic name
                print(d.dim_param, end=", ")
            else:
                print("?", end=", ")  # unknown dimension with no name
    else:
        print("unknown rank", end="")

# print(n_input)


def _node_attributes(node):
    attr = dict()
    for at in node.attribute:
        attr[at.name] = at
    return attr


root_node = graph.node[0]
attr = _node_attributes(root_node)

# attr
base_value = (
    np.array(attr["base_values"].floats)[
        0] if "base_values" in attr else 0.0
)


nodes_feature_ids = np.array(attr["nodes_featureids"].ints)
nodes_values = np.array(attr["nodes_values"].floats)
nodes_modes = np.array(attr["nodes_modes"].strings)
nodes_tree_ids = np.array(attr["nodes_treeids"].ints)
nodes_node_ids = np.array(attr["nodes_nodeids"].ints)
nodes_false_node_ids = np.array(attr["nodes_falsenodeids"].ints)
nodes_true_node_ids = np.array(attr["nodes_truenodeids"].ints)

n_targets = attr["n_targets"].i  # assert is 1 or not
target_ids = np.array(attr["target_ids"].ints)  # assert is same or not
target_node_ids = np.array(attr["target_nodeids"].ints)
target_tree_ids = np.array(attr["target_treeids"].ints)
target_weights = np.array(attr["target_weights"].floats)

nodes_leaf_mask = nodes_modes == b"LEAF"
nodes_branch_mask = nodes_modes == b"BRANCH_LEQ"

tree_ids = set(nodes_tree_ids)
feature_ids = set(nodes_feature_ids)

# Save tree structure into splits_dic, leaves_dic
from collections import defaultdict
from collections import deque

splits_dic = defaultdict(dict)
leaves_dic = defaultdict(dict)
for i in tree_ids:
    node = nodes_node_ids[nodes_tree_ids == i]
    feature = nodes_feature_ids[nodes_tree_ids == i]
    value = nodes_values[nodes_tree_ids == i]
    mode = nodes_modes[nodes_tree_ids == i]
    target_weight = target_weights[target_tree_ids == i]
    count = 0
    count_leaf = 0
    queue = deque([node[count]])
    while queue:
        cur = queue[0]
        queue.popleft()
        if mode[cur] == b'BRANCH_LEQ':
            splits_dic[i][cur] = {'th': value[cur],
                                  'col': feature[cur],
                                  'children': [None, None]}
            queue.appendleft(node[count + 2])
            splits_dic[i][cur]['children'][0] = node[count + 1]
            queue.appendleft(node[count + 1])
            splits_dic[i][cur]['children'][1] = node[count + 2]
            count += 2
        else:
            leaves_dic[i][cur] = {'val': target_weight[count_leaf]}
            count_leaf += 1

# Helper function to find all leaves under a node


def find_leaves(input_node, t):
    """
    input is a node and the tree_id, return is all the children leaves saved in a list.
    """
    root_node = input_node
    leaves_list = []
    queue = [root_node]
    while queue:
        node = queue.pop()
        node_left = node['children'][0]
        node_right = node['children'][1]
        if node_left in leaves_dic[t]:
            leaves_list.append(node_left)
        else:
            queue.append(splits_dic[t][node_left])
        if node_right in leaves_dic[t]:
            leaves_list.append(node_right)
        else:
            queue.append(splits_dic[t][node_right])
    return leaves_list


# Iterate through all trees and all splits
for t in tree_ids:
    for s in splits_dic[t].keys():
        cur_node = splits_dic[t][s]
        cur_node_left = cur_node['children'][0]  # find its left child
        cur_node_right = cur_node['children'][1]  # find its right child
        # create the list to save leaves
        cur_node['left_leaves'], cur_node['right_leaves'] = [], []
        if cur_node_left in leaves_dic[t]:  # if left child is a leaf node
            cur_node['left_leaves'].append(cur_node_left)
        else:  # traverse its left node by calling function to find all the leaves from its left node
            cur_node['left_leaves'] = find_leaves(
                splits_dic[t][cur_node_left], t)
        if cur_node_right in leaves_dic[t]:  # if right child is a leaf node
            cur_node['right_leaves'].append(cur_node_right)
        else:  # traverse its right node by calling function to find all the leaves from its right node
            cur_node['right_leaves'] = find_leaves(
                splits_dic[t][cur_node_right], t)

# Visualize the LightGBM tree structure
# import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\\bin'
# # lgb.plot_tree(model_linear)
# # for i in range(params['num_iterations']):
# # p = lgb.create_tree_digraph(model_linear, params['num_iterations'] - 1)
# p = lgb.create_tree_digraph(model_linear, -1)
# p

# ------ The way you find all children
# for i in tree_ids:
#     splits = splits_dic[i]
#     leaves = leaves_dic[i]
#     for split in splits:
#         left_child = splits[split]['children'][0]
#         right_child = splits[split]['children'][1]
#         if left_child in splits:
#             splits[left_child]['parent'] = split
#         else:
#             leaves[left_child]['parent'] = split

#         if right_child in splits:
#             splits[right_child]['parent'] = split
#         else:
#             leaves[right_child]['parent'] = split

# def find_all_children_splits(split, splits_dict):
#     """
#     This helper function finds all multigeneration children splits for an
#     argument split.

#     Arguments:
#         split --The split for which you are trying to find children splits
#         splits_dict -- A dictionary of all the splits in the tree

#     Returns:
#         A list containing the Node IDs of all children splits
#     """
#     all_splits = []

#     # Check if the immediate left child of the argument split is also a split.
#     # If so append to the list then use recursion to generate the remainder
#     left_child = splits_dict[split]['children'][0]
#     if left_child in splits_dict:
#         all_splits.append(left_child)
#         all_splits.extend(find_all_children_splits(left_child, splits_dict))

#     # Same as above but with right child
#     right_child = splits_dict[split]['children'][1]
#     if right_child in splits_dict:
#         all_splits.append(right_child)
#         all_splits.extend(find_all_children_splits(right_child, splits_dict))

#     return all_splits

# def find_all_children_leaves(split, splits_dict, leaves_dict):
#     """
#     This helper function finds all multigeneration children leaves for an
#     argument split.

#     Arguments:
#         split -- The split for which you are trying to find children leaves
#         splits_dict -- A dictionary of all the split info in the tree
#         leaves_dict -- A dictionary of all the leaf info in the tree

#     Returns:
#         A list containing all the Node IDs of all children leaves
#     """
#     all_leaves = []

#     # Find all the splits that are children of the relevant split
#     all_splits = find_all_children_splits(split, splits_dict)

#     # Ensure the current split is included
#     if split not in all_splits:
#         all_splits.append(split)

#     # For each leaf, check if the parents appear in the list of children
#     # splits (all_splits). If so, it must be a leaf of the argument split
#     for leaf in leaves_dict:
#         if leaves_dict[leaf]['parent'] in all_splits:
#             all_leaves.append(leaf)

#     return all_leaves

# for i in tree_ids:
#     splits = splits_dic[i]
#     leaves = leaves_dic[i]
#     for split in splits:
#         # print("split:" + str(split))
#         left_child = splits[split]['children'][0]
#         right_child = splits[split]['children'][1]

#         if left_child in splits:
#             # means left_child is split
#             splits[split]['left_leaves'] = find_all_children_leaves(
#                 left_child, splits, leaves
#             )
#         else:
#             # means left_child is leaf
#             splits[split]['left_leaves'] = [left_child]
#             # print("left_child" + str(left_child))

#         if right_child in splits:
#             splits[split]['right_leaves'] = find_all_children_leaves(
#                 right_child, splits, leaves
#             )
#         else:
#             splits[split]['right_leaves'] = [right_child]
#             # print("right_child" + str(right_child))


# ------ End

# Find bounds
# take care of this features, it is only the # of features in a tree not for all
features = np.arange(0, len(set(nodes_feature_ids)))
for i in tree_ids:
    splits = splits_dic[i]
    leaves = leaves_dic[i]
    for leaf in leaves:
        leaves[leaf]['bounds'] = {}
    for th in features:
        for leaf in leaves:
            leaves[leaf]['bounds'][th] = [None, None]
# for i in tree_ids:
#     splits = splits_dic[i]
#     leaves = leaves_dic[i]
    for split in splits:
        var = splits[split]['col']
        for leaf in splits[split]['left_leaves']:
            leaves[leaf]['bounds'][var][1] = splits[split]['th']

        for leaf in splits[split]['right_leaves']:
            leaves[leaf]['bounds'][var][0] = splits[split]['th']
# print(leaves_dic, splits_dic)
# ------ Pyomo section
# Reassign none bounds


def reassign_none_bounds(leaves, input_bounds):
    """
    This helper function reassigns bounds that are None to the bounds
    input by the user

    Arguments:
        leaves -- The dictionary of leaf information. Attribute of the 
            LinearTreeModel object
        input_bounds -- The nested dictionary

    Returns:
        The modified leaves dict without any bounds that are listed as None
    """
    L = np.array(list(leaves.keys()))
    features = np.arange(0, n_input)

    for l in L:
        for f in features:
            if leaves[l]['bounds'][f][0] == None:
                leaves[l]['bounds'][f][0] = input_bounds[f][0]
            if leaves[l]['bounds'][f][1] == None:
                leaves[l]['bounds'][f][1] = input_bounds[f][1]

    return leaves


input_bounds = {0: (minP, maxP),
                1: (minH, maxH)}
print(input_bounds)

for t in tree_ids:
    leaves_dic[t] = reassign_none_bounds(leaves_dic[t], input_bounds)

# pp.pprint(leaves_dic)

# Create model
m = pe.ConcreteModel()
m.T = pe.Var()
m.P = pe.Var()
m.H = pe.Var()


def create_tree_block(leaves_dic):
    b = pe.Block(concrete=True)

    tree_leaf_set = []
    for t in tree_ids:
        for l in leaves_dic[t].keys():
            tree_leaf_set.append((t, l))

    b.z = pe.Var(tree_leaf_set, within=pe.Binary)
    b.d = pe.Var(tree_ids)
    # b.z.pprint()

    # print(tf_set)
    b.P_H = pe.Var(features, within=pe.NonNegativeReals)
    # b.P_H.pprint()
    b.T = pe.Var(within=pe.NonNegativeReals)

    def lowerBounds(m, t, f):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(leaves_dic[t][l]['bounds'][f][0] * m.z[t, l] for l in L) <= m.P_H[f]

    b.lbCon = pe.Constraint(tree_ids, features, rule=lowerBounds)

    def upperBounds(m, t, f):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(leaves_dic[t][l]['bounds'][f][1] * m.z[t, l] for l in L) >= m.P_H[f]
    b.ubCon = pe.Constraint(tree_ids, features, rule=upperBounds)

    def outPuts(m, t):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(m.z[t, l] * leaves_dic[t][l]['val'] for l in L) == b.d[t]
    b.outputCon = pe.Constraint(tree_ids, rule=outPuts)

    def onlyOne(m, t):
        leaves = leaves_dic[t]
        L = np.array(list(leaves.keys()))
        return sum(b.z[t, l] for l in L) == 1
    b.onlyOneCon = pe.Constraint(tree_ids, rule=onlyOne)

    b.final_sum = pe.Constraint(expr=b.T == sum(b.d[t] for t in tree_ids))

    return b


m.tree_model = create_tree_block(leaves_dic)

m.linkP = pe.Constraint(expr=m.P == m.tree_model.P_H[0])
m.linkH = pe.Constraint(expr=m.H == m.tree_model.P_H[1])
m.linkT = pe.Constraint(expr=m.T == m.tree_model.T)

m.P.fix(5)
m.H.fix(50)

m.obj = pe.Objective(expr=1)

solver = pe.SolverFactory('gurobi')
results = solver.solve(m)

out = model_linear.predict(np.array([5, 50]).reshape(1, -1))
print(out, pe.value(m.T))
