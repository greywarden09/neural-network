#ifndef LAYER_HPP
#define LAYER_HPP
#include "perceptron.hpp"

namespace mlp {
    class Layer {
        std::vector<Perceptron> neurons;
        float learningRate;

    public:
        Layer(const uint32_t&, const uint32_t&, const nn::ActivationFunction&, float&);
    };
}

#endif
