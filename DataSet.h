#pragma once
#include <vector>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>

// PLACEHOLDER: swap this for your Column/DataSet from the analytics library.
// Minimum the regressor needs: row access, feature count, target access.
class DataSet {
public:
    DataSet(std::vector<std::vector<double>> X, std::vector<double> y)
        : X_(std::move(X)), y_(std::move(y)) {}

    std::size_t size() const { return X_.size(); }
    std::size_t numFeatures() const { return X_.empty() ? 0 : X_[0].size(); }
    const std::vector<double>& row(std::size_t i) const { return X_[i]; }
    double target(std::size_t i) const { return y_[i]; }
    const std::vector<std::vector<double>>& features() const { return X_; }
    const std::vector<double>& targets() const { return y_; }

    // Standardize each column to mean 0, std 1. Stores mean_/std_ so the
    // same scaling can later be applied to fresh/test-only data if needed.
    void normalize() {
        if (X_.empty()) return;
        std::size_t n = size(), f = numFeatures();
        mean_.assign(f, 0.0);
        std_.assign(f, 0.0);

        for (std::size_t j = 0; j < f; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += X_[i][j];
            mean_[j] = sum / n;
        }
        for (std::size_t j = 0; j < f; ++j) {
            double sq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = X_[i][j] - mean_[j];
                sq += d * d;
            }
            double variance = sq / n;
            std_[j] = std::sqrt(variance);
            if (std_[j] < 1e-12) std_[j] = 1.0; // avoid div-by-zero on a constant column
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < f; ++j)
                X_[i][j] = (X_[i][j] - mean_[j]) / std_[j];
    }

    // Returns {trainSet, testSet}. trainFraction in (0,1), e.g. 0.8.
    // Shuffles row indices with a fixed seed so results are reproducible
    // (swap in std::random_device if you want a different split each run).
    std::pair<DataSet, DataSet> trainTestSplit(double trainFraction,
                                                unsigned seed = 42) const {
        if (trainFraction <= 0.0 || trainFraction >= 1.0)
            throw std::invalid_argument("trainFraction must be in (0, 1)");

        std::vector<std::size_t> idx(size());
        for (std::size_t i = 0; i < size(); ++i) idx[i] = i;
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        std::size_t splitIdx = static_cast<std::size_t>(trainFraction * size());
        std::vector<std::vector<double>> trainX, testX;
        std::vector<double> trainY, testY;
        trainX.reserve(splitIdx); trainY.reserve(splitIdx);
        testX.reserve(size() - splitIdx); testY.reserve(size() - splitIdx);

        for (std::size_t k = 0; k < idx.size(); ++k) {
            std::size_t i = idx[k];
            if (k < splitIdx) { trainX.push_back(X_[i]); trainY.push_back(y_[i]); }
            else              { testX.push_back(X_[i]);  testY.push_back(y_[i]); }
        }
        return { DataSet(std::move(trainX), std::move(trainY)),
                 DataSet(std::move(testX), std::move(testY)) };
    }

private:
    std::vector<std::vector<double>> X_;
    std::vector<double> y_;
    std::vector<double> mean_, std_; // filled by normalize()
};
