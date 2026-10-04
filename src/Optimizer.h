#pragma once
#include <vector>

// Interface: given gradients, nudge the params.
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double>& weights, double& bias,
                        const std::vector<double>& gradW, double gradB) = 0;
    virtual const char* name() const = 0;
};
