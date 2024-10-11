#include "activation_function.hpp"

#include <algorithm>
#include <cmath>

namespace nn {
    float SigmoidActivationFunction::activation(const float &x) const {
        return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
    }

    float SigmoidActivationFunction::derivative(const float &x) const {
        const auto sigmoid = activation(x);
        return static_cast<float>(sigmoid * (1.0 - sigmoid));
    }

    float ReLUActivationFunction::activation(const float &x) const {
        return std::max(0.0f, x);
    }

    float ReLUActivationFunction::derivative(const float &x) const {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    f_vector SoftMaxActivationFunction::activation(const f_vector &inputs) {
        f_vector outputs(inputs.size());
        const auto maxInput = *std::ranges::max_element(inputs);
        auto sumExp = 0.0f;
        for (auto i = 0; i < inputs.size(); i++) {
            outputs[i] = std::exp(inputs[i] - maxInput);
            sumExp += outputs[i];
        }

        for (auto &output : outputs) {
            output /= sumExp;
        }

        return outputs;
    }

}
