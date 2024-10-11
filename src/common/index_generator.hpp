#ifndef INDEX_GENERATOR_HPP
#define INDEX_GENERATOR_HPP

namespace nn_common {
    class IndexGenerator {
    public:
        struct iterator;

        IndexGenerator(unsigned int x, unsigned int y, unsigned int z);

        IndexGenerator(unsigned int x, unsigned int y);

        [[nodiscard]] iterator begin() const;

        [[nodiscard]] iterator end() const;

    private:
        unsigned int X, Y, Z;
    };
}


#endif

