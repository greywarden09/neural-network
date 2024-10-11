#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <string>
#include <vector>

namespace nn_common {
    using TrainingModel = std::vector<std::pair<uint8_t, std::vector<uint8_t>>>;
    using String = std::string;

    class MNISTLoader {
        String path;
        String trainImagesPath;
        String trainLabelsPath;

        static uint32_t bigEndianToLittleEndian(const uint32_t&);
    public:
        explicit MNISTLoader(String path);

        TrainingModel loadTrainingData() const;
    };
}

#endif
