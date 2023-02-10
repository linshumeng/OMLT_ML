# [OMLT](https://github.com/linshumeng/OMLT/tree/main/tests)

## io

### test_input_bounds

I think it is a built-in function, and every io will need this. So we don't need to imitate it.

### test_keras_reader

They load existing nn model, check the result of _load_keras_sequential_, e.g. # of layers, activation function and the shape of weights.

### test_onnx_parser

They load existing nn model, check the result of _load_onnx_neural_network_, e.g. # of layer, activation function and the shape of weights.

## gbt

### test_gbt_formulation

They load existing gbt model from onnx, reformulate the model, and then check the result, e.g. # of variable, # of constraint, # of node(?), # of feature, # of singe leaf, # of left split, # of right splt, # of categorical, # of variable lower bound, # of variable upper bound.

## nn

### train_keras_models

They use keras to generate nns for other test problems as benchmark.

### test_layer

They test the _input_indexes_ and the result after different layers.

### test_onnx

They check _load_onnx_neural_network_ and use different activation function, linear, relu, sigmoid. They also check the _scaled_input_bound_ when nn model is load through _load_onnx_neural_network_with_bounds_.

### test_relu

They hard code input and output, compare the output from reluformulation(different representation for relu) with the hard code.

```
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    ReluBigMFormulation,
    ReluComplementarityFormulation,
    ReluPartitionFormulation,
)
```

### test_network_definition

They check the scaling in building nn.

### test_nn_formulation

They hard code input and output, compare the output from reformulation(different representation for nn) with the hard code.

```
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    NetworkDefinition,
    ReducedSpaceNNFormulation,
    ReducedSpaceSmoothNNFormulation,
)
```

### test_keras

They compare the output from reformulation with the keras predicted one.

# [scikit-learn](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/tree/tests)

- toy sample: X, y, T
- iris dataset
- diabetes dataset

# Future work

- Generate a tree model
  - for the following tests
- Result from tree parser
  - dict has every item
- Result from reformulation
  - Same as hard code GT
  - Same as prediction from package
  - Scalling
