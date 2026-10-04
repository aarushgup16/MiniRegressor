#pragma once
#include <vector>
#include <cstddef>

// Model only knows: params + how to predict + how to compute param gradients.
// It does NOT know how to train itself (that's the Optimizer/Trainer's job).
class LinearModel {
public:
    explicit LinearModel(std::size_t numFeatures)
        : weights_(numFeatures, 0.0), bias_(0.0) {}

    double predict(const std::vector<double>& x) const {
        double out = bias_;
        for (std::size_t j = 0; j < weights_.size(); ++j)
            out += weights_[j] * x[j];
        return out;
    }

    // Given dLoss/dPred for one sample, fill gradW (size = #features) and gradB.
    void backward(const std::vector<double>& x, double dLossdPred,
                  std::vector<double>& gradW, double& gradB) const {
        gradW.resize(weights_.size());
        for (std::size_t j = 0; j < weights_.size(); ++j)
            gradW[j] = dLossdPred * x[j];
        gradB = dLossdPred;
    }

    std::vector<double>& weights() { return weights_; }
    double& bias() { return bias_; }
    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }

private:
    std::vector<double> weights_;
    double bias_;
};
