#pragma once
#include <vector>
#include <cstddef>

// Evaluation metrics, kept separate from ILossFunction since these are for
// reporting/judging a trained model, not for driving gradient steps.
namespace Metrics {

    inline double mse(const std::vector<double>& yPred,
                       const std::vector<double>& yTrue) {
        if (yPred.empty()) return 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < yPred.size(); ++i) {
            double diff = yPred[i] - yTrue[i];
            sum += diff * diff;
        }
        return sum / yPred.size();
    }

    // R^2 = 1 - (sum of squared residuals) / (total sum of squares).
    // Fraction of variance in y explained by the model: 1.0 = perfect fit,
    // 0.0 = no better than predicting the mean, can go negative if worse.
    inline double r2Score(const std::vector<double>& yPred,
                           const std::vector<double>& yTrue) {
        if (yPred.empty()) return 0.0;
        double meanY = 0.0;
        for (double y : yTrue) meanY += y;
        meanY /= yTrue.size();

        double ssRes = 0.0, ssTot = 0.0;
        for (std::size_t i = 0; i < yPred.size(); ++i) {
            double resid = yTrue[i] - yPred[i];
            double total = yTrue[i] - meanY;
            ssRes += resid * resid;
            ssTot += total * total;
        }
        if (ssTot < 1e-12) return 0.0; // all y identical -> R^2 undefined, define as 0
        return 1.0 - ssRes / ssTot;
    }

} // namespace Metrics
