#pragma once
#include "Optimizer.h"
#include <cstddef>

class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(double learningRate) : lr_(learningRate) {}

    void update(std::vector<double>& weights, double& bias,
                const std::vector<double>& gradW, double gradB) override {
        for (std::size_t j = 0; j < weights.size(); ++j)
            weights[j] -= lr_ * gradW[j];
        bias -= lr_ * gradB;
    }
    const char* name() const override { return "SGD"; }

private:
    double lr_;
};
