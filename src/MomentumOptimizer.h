#pragma once
#include "Optimizer.h"
#include <vector>
#include <cstddef>

// SGD with momentum. Keeps a running "velocity" per parameter so updates
// build speed in a consistent direction and damp out noisy gradients.
//   v = beta*v + (1-beta)*grad
//   param -= lr * v
class MomentumOptimizer : public Optimizer {
public:
    MomentumOptimizer(double learningRate, double beta = 0.9)
        : lr_(learningRate), beta_(beta) {}

    void update(std::vector<double>& weights, double& bias,
                const std::vector<double>& gradW, double gradB) override {
        if (velocityW_.size() != weights.size())
            velocityW_.assign(weights.size(), 0.0);

        for (std::size_t j = 0; j < weights.size(); ++j) {
            velocityW_[j] = beta_ * velocityW_[j] + (1.0 - beta_) * gradW[j];
            weights[j] -= lr_ * velocityW_[j];
        }
        velocityB_ = beta_ * velocityB_ + (1.0 - beta_) * gradB;
        bias -= lr_ * velocityB_;
    }
    const char* name() const override { return "Momentum"; }

private:
    double lr_;
    double beta_;
    std::vector<double> velocityW_;
    double velocityB_ = 0.0;
};
