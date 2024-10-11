#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP
#include <vector>

namespace nn {
    using f_vector = std::vector<float>;


    class ActivationFunction {
    public:
        virtual float activation(const float &) const = 0;

        virtual float derivative(const float &) const = 0;

        virtual ~ActivationFunction() = default;
    };

    class SigmoidActivationFunction final : public ActivationFunction {
    public:
        float activation(const float &) const override;

        float derivative(const float &) const override;
    };

    class ReLUActivationFunction final : public ActivationFunction {
    public:
        float activation(const float &) const override;

        float derivative(const float &) const override;
    };

    class SoftMaxActivationFunction final {
    public:
        static f_vector activation(const f_vector &);
    };
}

#endif
