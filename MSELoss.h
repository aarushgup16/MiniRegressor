#pragma once
#include "ILossFunction.h"
#include <cstddef>

class MSELoss : public ILossFunction {
public:
    double compute(const std::vector<double>& yPred,
                   const std::vector<double>& yTrue) const override {
        if (yPred.empty()) return 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < yPred.size(); ++i) {
            double diff = yPred[i] - yTrue[i];
            sum += diff * diff;
        }
        return sum / yPred.size();
    }
    double gradient(double yPred, double yTrue) const override {
        // d/dPred [ (pred - true)^2 ] = 2*(pred - true)
        return 2.0 * (yPred - yTrue);
    }
    const char* name() const override { return "MSE"; }
};
