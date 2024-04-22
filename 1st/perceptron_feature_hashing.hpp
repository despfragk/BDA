#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;

    int seed_;

public:
    /** Do not change the signature of the constructor! */
    PerceptronFeatureHashing(int ngram, int log_num_buckets, double learning_rate)
        : BaseClf(0.0 /* set appropriate threshold */)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0xa738cc)
    {
        // set all weights to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email) {
        EmailIter iterator(email, ngram_);
        double prediction = predict_(email);
        int target = email.is_spam() ? 1 : -1;
        if (prediction * target <= 0) {
            while (iterator) {
                auto ngram = iterator.next();
                size_t bucket = get_bucket(ngram);
                weights_[bucket] += learning_rate_ * target;
            }
            bias_ += learning_rate_ * target;
        }
    }

    double predict_(const Email& email) const {
        EmailIter iterator(email, ngram_);
        double score = bias_;
        while (iterator) {
            auto ngram = iterator.next();
            size_t bucket = get_bucket(ngram);
            score += weights_[bucket];
        }
        return score;
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        // limit the range of the hash values here
        return hash & ((1 << log_num_buckets_) - 1);
    }
};

} // namespace bdap
