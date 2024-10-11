#include "index_generator.hpp"

#include <tuple>

namespace nn_common {
    struct IndexGenerator::iterator {
        unsigned int x, y, z, X, Y, Z;

        bool operator!=(const iterator &other) const {
            return x != other.x || y != other.y || z != other.z;
        }

        void operator++() {
            if (++z >= Z) {
                z = 0;
                if (++y >= Y) {
                    y = 0;
                    ++x;
                }
            }
        }

        std::tuple<int, int, int> operator*() const {
            return std::make_tuple(x, y, z);
        }
    };

    IndexGenerator::IndexGenerator(const unsigned int x, const unsigned int y, const unsigned int z): X(x), Y(y), Z(z) {
    }

    IndexGenerator::IndexGenerator(const unsigned int x,
                                   const unsigned int y): IndexGenerator(x, y, 0) {
    }

    IndexGenerator::iterator IndexGenerator::begin() const {
        return iterator{0, 0, 0, X, Y, Z};
    }

    IndexGenerator::iterator IndexGenerator::end() const {
        return iterator{X, 0, 0, X, Y, Z};
    }
}
