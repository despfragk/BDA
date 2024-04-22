#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    CountMinSketch cms_; // Assuming you have a Count-Min Sketch implementation

public:
    PerceptronCountMin(int ngram, int num_hashes, int log_num_buckets, double learning_rate)
        : BaseClf<PerceptronCountMin>(0.0), // Specify the template argument
          ngram_(ngram),
          log_num_buckets_(log_num_buckets),
          learning_rate_(learning_rate),
          bias_(0.0),
          weights_(1 << log_num_buckets, 0.0),
          cms_(num_hashes, log_num_buckets) // Initialize Count-Min Sketch
    {
    }

    void update_(const Email& email) {
        EmailIter iterator(email, ngram_);
        double prediction = predict_(email);
        int label = email.is_spam() ? 1 : -1;
        double error = label - prediction;

        while (iterator) {
            auto ngram = iterator.next();
            size_t hash = bdap::BaseClf<PerceptronCountMin>::hash(ngram, 0) % (1 << log_num_buckets_);
            weights_[hash] += learning_rate_ * error;
        }

        bias_ += learning_rate_ * error;
    }

    double predict_(const Email& email) const {
        EmailIter iterator(email, ngram_);
        double score = bias_;

        while (iterator) {
            auto ngram = iterator.next();
            size_t hash = bdap::BaseClf<PerceptronCountMin>::hash(ngram, 0) % (1 << log_num_buckets_);
            score += weights_[hash];
        }

        return score;
    }
};

} // namespace bdap
