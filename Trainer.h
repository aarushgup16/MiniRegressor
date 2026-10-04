#pragma once
#include <vector>
#include <cstddef>
#include "DataSet.h"
#include "LinearModel.h"
#include "ILossFunction.h"
#include "Optimizer.h"
#include "Metrics.h"

struct EpochLog {
    int epoch;
    double trainLoss;
    double testLoss;   // -1 if no test set was supplied
    double r2;          // R^2 on the train set this epoch
};

class Trainer {
public:
    // batchSize: 1 = pure SGD, trainData.size() = full-batch GD,
    // anything in between = mini-batch GD.
    Trainer(ILossFunction& loss, Optimizer& opt, std::size_t batchSize = 1)
        : loss_(loss), opt_(opt), batchSize_(batchSize) {}

    // Returns the full per-epoch history so it can be plotted/inspected.
    // testData is optional (pass nullptr to skip test-loss tracking).
    std::vector<EpochLog> fit(LinearModel& model, const DataSet& trainData,
                               int epochs, const DataSet* testData = nullptr) {
        std::vector<EpochLog> history;
        std::size_t n = trainData.size();
        std::size_t bs = batchSize_ == 0 ? 1 : batchSize_;

        for (int e = 0; e < epochs; ++e) {
            for (std::size_t start = 0; start < n; start += bs) {
                std::size_t end = std::min(start + bs, n);
                std::size_t actualBatch = end - start;

                std::vector<double> sumGradW(model.weights().size(), 0.0);
                double sumGradB = 0.0;

                for (std::size_t i = start; i < end; ++i) {
                    const auto& x = trainData.row(i);
                    double y = trainData.target(i);
                    double pred = model.predict(x);
                    double dL = loss_.gradient(pred, y);

                    std::vector<double> gradW;
                    double gradB;
                    model.backward(x, dL, gradW, gradB);

                    for (std::size_t j = 0; j < gradW.size(); ++j)
                        sumGradW[j] += gradW[j];
                    sumGradB += gradB;
                }

                // average gradient over the batch
                for (double& g : sumGradW) g /= static_cast<double>(actualBatch);
                sumGradB /= static_cast<double>(actualBatch);

                opt_.update(model.weights(), model.bias(), sumGradW, sumGradB);
            }

            // end-of-epoch bookkeeping
            std::vector<double> trainPreds(n);
            for (std::size_t i = 0; i < n; ++i)
                trainPreds[i] = model.predict(trainData.row(i));

            double trainLoss = loss_.compute(trainPreds, trainData.targets());
            double r2 = Metrics::r2Score(trainPreds, trainData.targets());

            double testLoss = -1.0;
            if (testData != nullptr) {
                std::vector<double> testPreds(testData->size());
                for (std::size_t i = 0; i < testData->size(); ++i)
                    testPreds[i] = model.predict(testData->row(i));
                testLoss = loss_.compute(testPreds, testData->targets());
            }

            history.push_back({e, trainLoss, testLoss, r2});
        }
        return history;
    }

private:
    ILossFunction& loss_;
    Optimizer& opt_;
    std::size_t batchSize_;
};
