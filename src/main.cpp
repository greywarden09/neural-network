#include <iostream>

#include "common/mnist_loader.hpp"

int main() {
    using nn_common::MNISTLoader;

    const MNISTLoader loader(R"(C:\Users\lasma\work\neural-network\data)");
    auto data = loader.loadTrainingData();
    const auto x = data[0].first;
    std::cout << static_cast<int>(data[0].first) << std::endl;
}
