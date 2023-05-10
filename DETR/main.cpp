// DETR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "definitions.h"
#include "conditional_detr_net.h"
#include "layers.h"

#include "hungarian_optimize_lsap.h"

int main()
{
    std::vector<double> costs({ 4,1,3,2,0,5,3,2,2 });
    auto cost = torch::from_blob(costs.data(), { 3,3 }, torch::kDouble);
    std::cout << "COST: " << cost << std::endl;
    LinearSumAssignment line = LinearSumAssignment(cost, false);

	/*conditional_detrnet* conditional = new conditional_detrnet(37);
	conditional->run();*/


    /*torch::Device device("cpu");
    torch::Tensor t = torch::rand({ 2, 3, 224, 224 }).to(device);
    //ResNet<BasicBlock<torch::nn::BatchNorm2dImpl>, torch::nn::BatchNorm2dImpl> resnet = resnet18<torch::nn::BatchNorm2dImpl>();
    ResNet<BottleNeck<FrozenBatchNorm2dImpl>, FrozenBatchNorm2dImpl> resnet = resnet50<FrozenBatchNorm2dImpl>();
    resnet.fc = nullptr;
    resnet.unregister_module("fc");
    resnet.to(device);
    t = resnet.forward(t);
    std::cout << t.sizes() << std::endl;*/

    std::cout << "FINISH" << std::endl;
    system("PAUSE");
}
