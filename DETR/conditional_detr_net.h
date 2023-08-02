#pragma once

#ifndef MODULES_CONDITIONAL_DETR_NET_H
#define MODULES_CONDITIONAL_DETR_NET_H

#include "models/backbone.h"
#include "models/conditional_detr.h"
#include "libs/datasets/custom_stack.h"
#include "libs/datasets/dataset.h"
#include "cmath"
#include <cuda.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>

#ifndef LIBS_DEFINITIONS_H
#include "libs/definitions.h"
#endif

#include "libs/layers.h"
#include "libs/util/nested_tensor.h"

typedef struct _configs {
	double lr;
	int nEpochs = 50;
	int seed;
	double weight_decays = 1e-4;
	/**
	 * \brief if trainBatchSize == -1 is autobatch
	 */
	int trainBatchSize = 8;
	int testBatchSize;
	bool use_fp32 = true;
	bool augments = false;
	std::string save_path;
	//decays decay_settings;
}config;

//Class that use for dataset, augmentation, net, forward, train, etc. Loading all modules
//class conditional_detrnet : public torch::optim::LRScheduler
class conditional_detrnet
{
private:
	
	torch::Device device;
	dataset_detr dataset_train_ = dataset_detr();
	//dataset_detr dataset_val = dataset_detr();
	Backbone backbone_ = Backbone();
	ConditionalDETR conditional_detr_;
	conditional_detr::Transformer* transformer_ = new conditional_detr::Transformer(256);
	torch::optim::Optimizer* optimizer = nullptr;
	SetCriterion criterion;
	//torch::optim::LRScheduler lr_scheduler = torch::optim::LRScheduler();
	std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<dataset_detr, CustomStackV2<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>>, torch::data::samplers::RandomSampler>> dataloader = nullptr;
protected:
	config params;

public:
	/*std::vector<double> get_lrs() override
	{
		std::vector<double> lrs;
		return lrs;
	}*/
	/*conditional_detrnet() : LRScheduler(*optimizer), device(torch::kCPU)
	{
		if (torch::cuda::is_available())
			device = torch::kCUDA;
	}*/
	conditional_detrnet() : device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU)
	{
		/*if (torch::cuda::is_available())
			device = torch::kCUDA;*/
	}
	conditional_detrnet(int num_class, int num_queries = 300) : conditional_detrnet()
	{
		conditional_detr_ = ConditionalDETR(backbone_, transformer_, num_class, num_queries);
		criterion = SetCriterion(num_class);

		build_module();

		/*if(torch::cuda::is_available())
		{
			torch::cuda::synchronize();
			c10::cuda::CUDACachingAllocator::emptyCache();
		}*/
	}
	void build_module()
	{
		conditional_detr_.to(device);
		//criterion.to(device);
		if (!params.use_fp32) {
			conditional_detr_.to(torch::kHalf);
			//criterion.to(torch::kHalf);
		}

		int64_t num_parameters=0;
		for(const auto& v : conditional_detr_.parameters())
			if(v.requires_grad())
				num_parameters+=v.numel();
		MESSAGE_LOG("Number of Parameters: " + std::to_string(num_parameters))
		//std::cout << "Number of parameters: " << std::to_string(num_parameters) << std::endl;
		if (optimizer == nullptr)
			optimizer = new torch::optim::AdamW(conditional_detr_.parameters(), torch::optim::AdamWOptions(params.lr).weight_decay(params.weight_decays));
	}

	void run()
	{
		for(int i=0;i<params.nEpochs;i++)
		{
			/*conditional_detr_.train();
			criterion.train();*/
			//train epoch
			trainnet();
			valnet();

			//this->step(); //scheduler step
		}
	}
	void trainnet()
	{
		MESSAGE_LOG("Trainnet")
		
		//torch::NoGradGuard noGrad;
		std::vector<std::string> images;
		cv::glob("D://Datasets//TZ_Contenedores_Chasis//ContainerGeneratorv10//images//train", images);
		std::vector<std::string> labels;
		cv::glob("D://Datasets//TZ_Contenedores_Chasis//ContainerGeneratorv10//labels//train", labels);
		dataset_train_.add_data(images, labels);

		if(dataloader == nullptr)
		{
			auto transf = dataset_train_.map(CustomStackV2<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>());
			dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(transf), params.trainBatchSize);
		}
		//std::cout << "PRUEBA: " << dataset_train_->get(0).target["labels"] << std::endl;
 		const auto start = std::chrono::high_resolution_clock::now();
		int batch_index = 0;
		conditional_detr_.train();
		criterion.train();
		for (auto& batch : *dataloader)
		{
			auto nested = NestedTensor(batch.data);
			nested.to(this->device);
			//auto da = batch.data.to(this->device);
			
			/*for (uint64_t i = 0; i < batch.target.size(); i++) {
				for (auto& n : batch.target[i]) {
					
					//n.value().to(this->device);
					//MESSAGE_LOG("BatchTarget Name and Device: " + n.key() + ", " + std::to_string(n.value().get_device()))
					if (n.key() == "labels") {
						std::cout << "BatchTarget: [" << n.key() << "]" << n.value() << std::endl;
					}
				}
			}*/
			//std::cout << "Size da.size(): " << da.sizes() << std::endl;
			auto sample = conditional_detr_.forward(nested);
			auto loss = criterion.forward(sample, batch.target);

			
			torch::Tensor losses = torch::zeros(at::IntArrayRef({}), torch::kFloat);
			//std::vector<torch::Tensor> losses;
			for (const auto& v : loss) {

				std::cout << v.first << " value: " << v.second << std::endl;
				for (const auto& u : criterion.weight_dict_) {
					if (u.first == v.first) {
						losses += u.second * v.second.to(torch::kCPU);
						//losses.push_back(u.second * v.second);
					}
				}
			}
			
			
			optimizer->zero_grad();
			std::cout << "Losses: " << losses << std::endl;
			losses.backward();
			/*if(losses.size() == 1)
			{
				std::cout << "Losses: " << losses << std::endl;
				losses[0].backward();
			}
			else if(losses.size() > 1)
			{
				for (uint64_t i = 0; i < losses.size(); i++)
					std::cout << "Losses size matrix: " << losses[i].sizes() << std::endl;

				if (losses[0].sizes() != torch::IntArrayRef({}))
				{
					
					std::cout << "lero: " << lero << std::endl;
					lero = lero.sum();

					MESSAGE_LOG(lero.sizes())
					MESSAGE_LOG(lero)
					//std::cout << "Lero: " << lero << std::endl;
					lero.backward();
				}
			}*/
			optimizer->step();
			std::cout << "Batch Index: " << batch_index << std::endl;
			batch_index++;

			//torch::cuda::synchronize();
		}
		std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms" << std::endl;
	}
	void valnet()
	{

	}
};

#endif