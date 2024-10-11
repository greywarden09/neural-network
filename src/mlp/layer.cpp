#include "layer.hpp"

namespace mlp {
    Layer::Layer(const uint32_t &inputs,
                 const uint32_t &neurons,
                 const nn::ActivationFunction &activationFunction,
                 float &learningRate)
        : neurons(neurons), learningRate(learningRate) {
        for (auto i = 0; i < neurons; i++) {
            this->neurons.emplace_back(inputs, learningRate, activationFunction);
        }
    }
}
