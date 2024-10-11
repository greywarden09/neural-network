#include "perceptron.hpp"

#include <chrono>
#include <random>

namespace mlp {
    void Perceptron::initializeWeights() {
        std::normal_distribution<float> distribution(0.0, 1.0);
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

        bias = distribution(generator);
        std::ranges::generate(weights, [&]() { return distribution(generator); });
    }

    Perceptron::Perceptron(const uint32_t &inputs,
                           const float &learningRate,
                           const nn::ActivationFunction &activationFunction)
        : weights(inputs), bias(0.0f), learningRate(learningRate), activation(activationFunction) {
        initializeWeights();
    }

    float Perceptron::forward(const f_vector &inputs) const {
        auto weightedSum = bias;
        for (auto i = 0; i < weights.size(); ++i) {
            weightedSum += weights[i] * inputs[i];
        }
        return activation.activation(weightedSum);
    }

    void Perceptron::backward(const f_vector &inputs, const float &output, const float &target) {
        const auto error = target - output;
        const auto gradient = error * activation.derivative(output);

        for (auto i = 0; i < weights.size(); ++i) {
            weights[i] += learningRate * gradient * inputs[i];
        }

        bias += learningRate * gradient;
    }
}
