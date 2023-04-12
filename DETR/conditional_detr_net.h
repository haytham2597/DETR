#pragma once

#ifndef MODULES_CONDITIONAL_DETR_NET_H
#define MODULES_CONDITIONAL_DETR_NET_H

#include "backbone.h"
#include "conditional_detr.h"
#include "custom_stack.h"
#include "dataset.h"

typedef struct _configs {
	double lr;
	int nEpochs = 50;
	int seed;
	double weight_decays = 1e-4;
	/**
	 * \brief if trainBatchSize == -1 is autobatch
	 */
	int trainBatchSize = 1;
	int testBatchSize;
	bool use_fp32 = true;
	bool augments = false;
	std::string save_path;
	//decays decay_settings;
}config;

//Class that use for dataset, augmentation, net, forward, train, etc. Loading all modules
class conditional_detrnet : public torch::optim::LRScheduler
{
private:
	
	torch::Device device;
	dataset_detr dataset_train_ = dataset_detr();
	dataset_detr dataset_val = dataset_detr();
	Backbone backbone_ = Backbone();
	ConditionalDETR conditional_detr_;
	conditional_detr::Transformer transformer_ = conditional_detr::Transformer(256);
	torch::optim::Optimizer* optimizer = nullptr;
	SetCriterion criterion;
	//torch::optim::LRScheduler lr_scheduler = torch::optim::LRScheduler();
	std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<dataset_detr, CustomStackV2<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>>, torch::data::samplers::RandomSampler>> dataloader = nullptr;
protected:
	config params;

public:
	std::vector<double> get_lrs() override
	{
		std::vector<double> lrs;
		return lrs;
	}
	conditional_detrnet() : LRScheduler(*optimizer), device(torch::kCPU)
	{
		if (torch::cuda::is_available())
			device = torch::kCUDA;
	}
	conditional_detrnet(int num_class, int num_queries = 300) : conditional_detrnet()
	{
		conditional_detr_ = ConditionalDETR(backbone_, transformer_, num_class, num_queries);
		criterion = SetCriterion(num_class);

		build_module();

		std::vector<std::string> images;
		cv::glob("D://Datasets//ContainerGeneratorv10//images//train", images);
		std::vector<std::string> labels;
		cv::glob("D://Datasets//ContainerGeneratorv10//labels//train", labels);
		dataset_train_.add_data(images, labels);
	}
	void build_module()
	{
		conditional_detr_.to(device);
		criterion.to(device);
		if (!params.use_fp32) {
			conditional_detr_.to(torch::kHalf);
			criterion.to(torch::kHalf);
		}

		int64_t num_parameters=0;
		for(auto v : conditional_detr_.parameters())
		{
			if(v.requires_grad())
				num_parameters+=v.numel();
		}
		std::cout << "Number of parameters: " << std::to_string(num_parameters) << std::endl;

		if (optimizer == nullptr)
			optimizer = new torch::optim::AdamW(conditional_detr_.parameters(), torch::optim::AdamWOptions(params.lr).weight_decay(params.weight_decays));
	}
	void add_data(std::string yolopath, bool usefp32)
	{
		
		//cv::glob()
		//dataset_.add_data()
	}
	void run()
	{
		for(int i=0;i<params.nEpochs;i++)
		{
			conditional_detr_.train();
			criterion.train();
			//train epoch

			trainnet();
			valnet();

			this->step(); //scheduler step
		}
	}
	void trainnet()
	{
		if(dataloader == nullptr)
		{
			
			/*auto transf = torch::data::transforms::Stack<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>();*/
			auto transf = dataset_train_.map(CustomStackV2<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>());
			/*auto myset = dataset_train_.map(torch::data::transforms::Collate<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>([](std::vector<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>> e){
				
				return std::move(e.front());
			}));*/
			//auto dla = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(transf), params.trainBatchSize);;
			dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(transf), params.trainBatchSize);
		}
		const auto start = std::chrono::high_resolution_clock::now();
		int batch_index = 0;
		for (auto& batch : *dataloader)
		{
			auto da = batch.data.to(this->device);
			for (int i = 0; i < batch.target.size(); i++)
				for (auto n : batch.target[i])
					n.value().to(this->device);

			auto sample = conditional_detr_.forward(NestedTensor(da));
			auto loss = criterion.forward(sample, batch.target);
			
			optimizer->zero_grad();
			std::vector<torch::Tensor> losses;
			for (auto v : loss)
			for (auto u : criterion.weight_dict_) {
				if (u.first == v.first) {
					std::cout << "Loss name: " << v.first << " val: " << v.second << std::endl;
					losses.push_back(u.second * v.second);
				}
			}

			torch::Tensor losses_sum;
			for(int i=0;i<losses.size();i++)
				losses_sum += losses[i];

			losses_sum.backward();
			optimizer->step();
			std::cout << losses_sum.sizes() << std::endl;
			std::cout << losses_sum << std::endl;
			std::cout << "Batch Index: " << batch_index << std::endl;
			batch_index++;
		}
	}
	void valnet()
	{

	}
};

#endif