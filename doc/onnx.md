# onnx

# onnx for LightGBM

- required arguments: model, ('variable_name', data_type)

- problem: cannot save linear_tree argument

- solution: save LightGBM model in txt and load it use the default loader in LightGBM
