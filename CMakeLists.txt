cmake_minimum_required(VERSION 3.29)
project(neural_network)

set(CMAKE_CXX_STANDARD 23)

add_executable(neural_network
        src/main.cpp
        src/nn/activation_function.hpp src/nn/activation_function.cpp
        src/mlp/layer.hpp src/mlp/layer.cpp
        src/mlp/perceptron.hpp src/mlp/perceptron.cpp
        src/common/index_generator.hpp src/common/index_generator.cpp
        src/common/mnist_loader.hpp src/common/mnist_loader.cpp
)
