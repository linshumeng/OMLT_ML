# onnx

# onnx for LightGBM

- required arguments: model, ('variable_name', data_type)

- problem: onnx cannot save 'linear_tree', 'leaf_const', 'leaf_coeff' arguments, but only 'leaf_value', so it is hard to reproduce the tree structure

- solution: we save LightGBM model in txt and load it use the default loader in LightGBM, or we can write a parser in omlt to do so
