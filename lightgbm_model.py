import numpy as np
import lightgbm as lgb

def find_all_children_splits(split, splits_dict):
    """
    This helper function finds all multigeneration children splits for an 
    argument split.

    Arguments:
        split --The split for which you are trying to find children splits
        splits_dict -- A dictionary of all the splits in the tree
    
    Returns:
        A list containing the Node IDs of all children splits
    """
    all_splits = []

    # Check if the immediate left child of the argument split is also a split.
    # If so append to the list then use recursion to generate the remainder
    left_child = splits_dict[split]['children'][0]
    if left_child in splits_dict:
        all_splits.append(left_child)
        all_splits.extend(find_all_children_splits(left_child, splits_dict))

    # Same as above but with right child
    right_child = splits_dict[split]['children'][1]
    if right_child in splits_dict:
        all_splits.append(right_child)
        all_splits.extend(find_all_children_splits(right_child, splits_dict))

    return all_splits

def find_all_children_leaves(split, splits_dict, leaves_dict):
    """
    This helper function finds all multigeneration children leaves for an 
    argument split.

    Arguments:
        split -- The split for which you are trying to find children leaves
        splits_dict -- A dictionary of all the split info in the tree
        leaves_dict -- A dictionary of all the leaf info in the tree

    Returns:
        A list containing all the Node IDs of all children leaves
    """
    all_leaves = []

    # Find all the splits that are children of the relevant split
    all_splits = find_all_children_splits(split, splits_dict)

    # Ensure the current split is included
    if split not in all_splits:
        all_splits.append(split)

    # For each leaf, check if the parents appear in the list of children
    # splits (all_splits). If so, it must be a leaf of the argument split
    for leaf in leaves_dict:
        if leaves_dict[leaf]['parent'] in all_splits:
            all_leaves.append(leaf)

    return all_leaves

def parse_model(model):
    if str(type(model)) == "<class 'lightgbm.basic.Booster'>":
        whole_model = model.dump_model()
    else:
        model = lgb.Booster(model_file = model)
        # change the model to json format
        whole_model = model.dump_model()

    tree = {}
    for i in range(whole_model['tree_info'][-1]['tree_index']+1):

        node  = whole_model['tree_info'][i]["tree_structure"]

        queue = [node]
        splits = {}

        # the very first node
        splits["split"+str(queue[0]["split_index"])] = {'th': queue[0]["threshold"],
                                        'col': queue[0]["split_feature"] }

        # flow though the tree
        while queue:
            
            # left child
            if "left_child" in queue[0].keys():
                queue.append(queue[0]["left_child"])
                # child is a split
                if "split_index" in queue[0]["left_child"].keys():
                    splits["split"+str(queue[0]["left_child"]["split_index"])] = {'parent': "split"+str(queue[0]["split_index"]),
                                                                'direction': 'left',
                                                                'th': queue[0]["left_child"]["threshold"], 
                                                                'col': queue[0]["left_child"]["split_feature"]}
                # child is a leaf
                else:
                    splits["leaf"+str(queue[0]["left_child"]["leaf_index"])] = {'parent': "split"+str(queue[0]["split_index"]),
                                                                'direction': 'left', 
                                                                'intercept': queue[0]["left_child"]["leaf_const"], 
                                                                'slope': queue[0]["left_child"]["leaf_coeff"]}
                    
            # right child
            if "right_child" in queue[0].keys():
                queue.append(queue[0]["right_child"])      
                # child is a split
                if "split_index" in queue[0]["right_child"].keys():
                    splits["split"+str(queue[0]["right_child"]["split_index"])] = {'parent': "split"+str(queue[0]["split_index"]),
                                                                'direction': 'right',
                                                                'th': queue[0]["right_child"]["threshold"], 
                                                                'col': queue[0]["right_child"]["split_feature"]}
                # child is a leaf
                else:
                    splits["leaf"+str(queue[0]["right_child"]["leaf_index"])] = {'parent': "split"+str(queue[0]["split_index"]),
                                                                'direction': 'right',
                                                                'intercept': queue[0]["right_child"]["leaf_const"], 
                                                                'slope': queue[0]["right_child"]["leaf_coeff"]}
            # delet the first node
            queue.pop(0)

            tree['tree'+str(i)] = splits

    nested_splits = {}
    nested_leaves = {}
    nested_thresholds = {}

    n_inputs = model.num_feature()
    for index in tree:

        splits = tree[index]
        for i in splits:
            # print(i)
            if 'parent' in splits[i].keys():
                splits[splits[i]['parent']]['children'] = []

        for i in splits:
            # print(i)
            if 'parent' in splits[i].keys():
                if splits[i]['direction'] == 'left':     
                    splits[splits[i]['parent']]['children'].insert(0,i)
                if splits[i]['direction'] == 'right':     
                    splits[splits[i]['parent']]['children'].insert(11,i)

        leaves = {}
        for i in splits.keys():
            if i[0] == 'l':
                leaves[i] = splits[i]

        for leaf in leaves:
            del splits[leaf]

        for split in splits:
            # print("split:" + str(split))
            left_child = splits[split]['children'][0]
            right_child = splits[split]['children'][1]
            
            if left_child in splits:
                # means left_child is split
                splits[split]['left_leaves'] = find_all_children_leaves(
                    left_child, splits, leaves
                    )
            else:
                # means left_child is leaf
                splits[split]['left_leaves'] = [left_child]
                # print("left_child" + str(left_child))
            
            if right_child in splits:
                splits[split]['right_leaves'] = find_all_children_leaves(
                    right_child, splits, leaves
                    )
            else:
                splits[split]['right_leaves'] = [right_child]
                # print("right_child" + str(right_child))

        splitting_thresholds = {}
        for split in splits:
            var = splits[split]['col']
            splitting_thresholds[var] = {}
        for split in splits:
            var = splits[split]['col']
            splitting_thresholds[var][split] = splits[split]['th']

        for var in splitting_thresholds:
            splitting_thresholds[var] = dict(sorted(splitting_thresholds[var].items(), key=lambda x: x[1]))

        for split in splits:
            var = splits[split]['col']
            splits[split]['y_index'] = []
            splits[split]['y_index'].append(var)
            splits[split]['y_index'].append(
                list(splitting_thresholds[var]).index(split)
            )

        features = np.arange(0,n_inputs)

        for leaf in leaves:
            leaves[leaf]['bounds'] = {}
        for th in features:
            for leaf in leaves:
                leaves[leaf]['bounds'][th] = [None, None]
        
        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(splits)
        # pp.pprint(leaves)
        for split in splits:
            var = splits[split]['col']
            for leaf in splits[split]['left_leaves']:
                leaves[leaf]['bounds'][var][1] = splits[split]['th']

            for leaf in splits[split]['right_leaves']:
                leaves[leaf]['bounds'][var][0] = splits[split]['th']

        nested_splits['tree' + str(index)] = splits
        nested_leaves['tree' + str(index)] = leaves
        nested_thresholds['tree' + str(index)] = splitting_thresholds

    return nested_splits, nested_leaves, nested_thresholds
    