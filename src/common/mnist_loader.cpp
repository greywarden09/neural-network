#include "mnist_loader.hpp"

#include <filesystem>
#include <utility>
#include <fstream>
#include <iostream>


namespace nn_common {
    String trainImages = "train-images.idx3-ubyte";
    String trainLabels = "train-labels.idx1-ubyte";

    uint32_t MNISTLoader::bigEndianToLittleEndian(const uint32_t &value) {
        const uint8_t c1 = value & 255;
        const uint8_t c2 = value >> 8 & 255;
        const uint8_t c3 = value >> 16 & 255;
        const uint8_t c4 = value >> 24 & 255;

        return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) + (static_cast<int>(c3) << 8) + c4;
    }

    MNISTLoader::MNISTLoader(String path): path(std::move(path)) {
    }


    TrainingModel MNISTLoader::loadTrainingData() const {
        TrainingModel dataset;

        std::filesystem::path dataDir = path;
        auto trainImagesPath = (path / std::filesystem::path(trainImages)).string();
        auto trainLabelsPath = (path / std::filesystem::path(trainLabels)).string();

        std::ifstream trainImages(trainImagesPath, std::ios::binary);
        std::ifstream trainLabels(trainLabelsPath, std::ios::binary);
        if (trainImages.is_open() && trainLabels.is_open()) {
            uint32_t magicNumber = 0;
            uint32_t nRows = 0;
            uint32_t nCols = 0;
            uint32_t numberOfImages = 0;
            uint32_t numberOfLabels = 0;
            uint32_t imageSize = 0;

            trainImages.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
            magicNumber = bigEndianToLittleEndian(magicNumber);

            if (magicNumber != 2051) {
                throw std::runtime_error("Invalid magic number - images");
            }

            trainLabels.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
            magicNumber = bigEndianToLittleEndian(magicNumber);

            if (magicNumber != 2049) {
                throw std::runtime_error("Invalid magic number - labels");
            }

            trainImages.read(reinterpret_cast<char *>(&numberOfImages), sizeof(numberOfImages)),
                    numberOfImages = bigEndianToLittleEndian(numberOfImages);

            trainImages.read(reinterpret_cast<char *>(&nRows), sizeof(nRows)), nRows = bigEndianToLittleEndian(nRows);
            trainImages.read(reinterpret_cast<char *>(&nCols), sizeof(nCols)), nCols = bigEndianToLittleEndian(nCols);

            imageSize = nRows * nCols;

            trainLabels.read(reinterpret_cast<char *>(&numberOfLabels), sizeof(numberOfLabels)), numberOfLabels =
                    bigEndianToLittleEndian(numberOfLabels);

            if (numberOfImages != numberOfLabels) {
                throw std::runtime_error("Number of labels and images does not match");
            }

            for (auto i = 0; i < numberOfImages; i++) {
                auto buffer = new uint8_t[imageSize];
                uint8_t label = 0;
                trainImages.read(reinterpret_cast<char *>(buffer), imageSize);
                trainLabels.read(reinterpret_cast<char *>(&label), 1);
                std::vector<uint8_t> data(imageSize);
                std::ranges::copy(&buffer[0], &buffer[imageSize - 1], back_inserter(data));
                dataset.emplace_back(label, data);

                delete[] buffer;
            }
        } else {
            throw std::runtime_error("Failed to open training images or labels file");
        }

        return dataset;
    }
}
