#include <fstream>
#include <iostream>

#include "mlp/perceptron.hpp"

int main() {
    const auto perceptron = new mlp::Perceptron(10, 0.1f, nn::SigmoidActivationFunction());
    auto weights = perceptron->getWeights();

    std::ofstream fs("perceptron.bin", std::ios::out | std::ios::binary);
    for (const auto weight : weights) {
        fs.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }
    fs.close();

    //
    std::ifstream ifs("perceptron.bin", std::ios::binary);
    std::vector<float> loadedWeights(10);
    for (auto i = 0; i < 10; i++) {
        ifs.read(reinterpret_cast<char*>(&loadedWeights[i]), sizeof(float));
    }

    delete perceptron;
}
