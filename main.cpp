#include <iostream>
#include <iomanip>
#include "DataSet.h"
#include "LinearModel.h"
#include "MSELoss.h"
#include "MAELoss.h"
#include "SGDOptimizer.h"
#include "MomentumOptimizer.h"
#include "Trainer.h"

int main() {
    // Toy house-price data: {area, bedrooms} -> price (in lakhs)
    DataSet fullData(
        {{1000, 2}, {1500, 3}, {2000, 3}, {2500, 4}, {3000, 4}, {3500, 5},
         {1200, 2}, {1800, 3}, {2200, 3}, {2800, 4}},
        {200, 300, 380, 500, 540, 610, 230, 340, 400, 520});

    fullData.normalize();
    auto [trainData, testData] = fullData.trainTestSplit(0.8);

    LinearModel model(fullData.numFeatures());

    MAELoss loss;
    //We can change the loss function to compare

    // MomentumOptimizer opt(0.1, 0.9);
    SGDOptimizer opt(0.1);            //Can change the optimizer

    std::size_t batchSize = 4; 
    Trainer trainer(loss, opt, batchSize);

    auto history = trainer.fit(model, trainData, /*epochs=*/200, &testData);

    std::cout << std::fixed << std::setprecision(4);
    for (const auto& log : history) {
        if (log.epoch % 20 != 0 && log.epoch != static_cast<int>(history.size()) - 1)
            continue; // print every 20th epoch + the last one, keep output readable
        std::cout << "epoch " << std::setw(3) << log.epoch
                  << " | train_loss=" << log.trainLoss
                  << " | test_loss=" << log.testLoss
                  << " | r2=" << log.r2 << "\n";
    }

    std::cout << "\nLearned weights: [";
    for (double w : model.weights()) std::cout << w << " ";
    std::cout << "], bias: " << model.bias() << "\n";

    std::cout << "Loss function: " << loss.name()
              << ", Optimizer: " << opt.name() << "\n";

    return 0;
}
