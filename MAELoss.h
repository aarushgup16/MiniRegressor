#pragma once
#include "ILossFunction.h"
#include <cmath>
#include <cstddef>

// Same interface as MSELoss -> drop-in swap, nothing else in the pipeline
// needs to change. That's the point of ILossFunction.
class MAELoss : public ILossFunction {
public:
    double compute(const std::vector<double>& yPred,
                   const std::vector<double>& yTrue) const override {
        if (yPred.empty()) return 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < yPred.size(); ++i)
            sum += std::fabs(yPred[i] - yTrue[i]);
        return sum / yPred.size();
    }
    double gradient(double yPred, double yTrue) const override {
        double diff = yPred - yTrue;
        if (diff > 0.0) return 1.0;
        if (diff < 0.0) return -1.0;
        return 0.0; // subgradient at the kink
    }
    const char* name() const override { return "MAE"; }
};
