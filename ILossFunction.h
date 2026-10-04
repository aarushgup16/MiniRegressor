#pragma once
#include <vector>

// Interface: any loss must say how bad predictions are AND how to improve them.
class ILossFunction {
public:
    virtual ~ILossFunction() = default;

    // scalar loss over a batch
    virtual double compute(const std::vector<double>& yPred,
                           const std::vector<double>& yTrue) const = 0;

    // dLoss/dyPred for ONE sample (chain rule starts here)
    virtual double gradient(double yPred, double yTrue) const = 0;

    virtual const char* name() const = 0;
};
