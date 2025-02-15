#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP
#include <vector>

#include "../nn/activation_function.hpp"

namespace mlp {
    using f_vector = std::vector<float>;
    class Perceptron {
        f_vector weights;
        float bias;
        float learningRate;
        const nn::ActivationFunction &activation;

        void initializeWeights();

    public:
        Perceptron();
        Perceptron(const uint32_t&, const float&, const nn::ActivationFunction&);
        float forward(const f_vector&) const;
        void backward(const f_vector&, const float&, const float&);

        f_vector getWeights() const;
    };
}

#endif
