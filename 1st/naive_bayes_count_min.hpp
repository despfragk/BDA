#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"
#include <random>
#ifndef ROTL64
#include "murmurhash.hpp"
#endif

namespace bdap {
    uint64_t MurmurHash3_x64_128(const void* key, size_t len, uint32_t seed);

    class CountMinSketch {
    private:
        int numHashes_;
        int size_;
        std::vector<std::vector<int> > table;

    public:
        CountMinSketch(int numHashes, int size) : numHashes_(numHashes), size_(size) {
            table.resize(numHashes);
            for (int i = 0; i < numHashes; i++) {
                table[i].resize(size, 0);
            }
        }

        void update(const std::string& ngram, int count) {
            uint64_t hashValue;
            int index;
            for (int i = 0; i < numHashes_; i++) {
                hashValue = MurmurHash3_x64_128(ngram.data(), ngram.size(), i);
                index = hashValue % size_;
                table[i][index] += count;
            }
        }

        int estimate(const std::string& ngram) const {
            uint64_t hashValue;
            int index, min = std::numeric_limits<int>::max();
            for (int i = 0; i < numHashes_; i++) {
                hashValue = MurmurHash3_x64_128(ngram.data(), ngram.size(), i);
                index = hashValue % size_;
                min = std::min(min, table[i][index]);
            }
            return min;
        }
    };
} // namespace bdap