// DETR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "conditional_detr_net.h"
/*
class dataset_test : public torch::data::Dataset<dataset_test, torch::data::Example<torch::Tensor, int>>
{
protected:
	std::vector<torch::data::Example<torch::Tensor, int>> examples_;
	int siz_data_;
public:
	dataset_test()
	{
		for(int i=0;i<100;i++)
		{
			//torch::OrderedDict<std::string, int> order;
			//order.insert("myorder" + std::to_string(i), i + 5);
			//if(i % 5 == 0)
				//order.insert("luero" + std::to_string(i+1), i+1 + 5);
			examples_.push_back({ torch::randint(10, {2,1}), (i+5)});
		}
		std::cout << "Allorder is added" << std::endl;
	}
	
	torch::optional<size_t> size()const override {
		return examples_.size();
	}
	ExampleType get(size_t index) override
	{
		return examples_[index];
	}
	~dataset_test()
	{
		examples_.clear();
	}
};
*/
int main()
{
	/*auto tensor = torch::rand({ 3,4,4 }, torch::kFloat);
	auto pad_img = torch::zeros({ 1, tensor.size(0),tensor.size(1),tensor.size(2) }, tensor.dtype());
	auto mask = torch::ones({ 1, 4,4 }, torch::kBool);
	pad_img = pad_img.index({ torch::indexing::Slice(torch::indexing::None, tensor.size(0)), torch::indexing::Slice(torch::indexing::None, tensor.size(1)),torch::indexing::Slice(torch::indexing::None, tensor.size(2)) }).copy_(tensor);
	mask.index_put_({ torch::indexing::Slice(torch::indexing::None, tensor.size(1)), torch::indexing::Slice(torch::indexing::None, tensor.size(2)) }, false);
	//pad_img.copy_(tensor);
	std::cout << tensor << std::endl;
	std::cout << pad_img << std::endl;
	std::cout << mask << std::endl;*/
	//tensors_.copy_()

	conditional_detrnet conditional = conditional_detrnet(37);
	conditional.run();

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
