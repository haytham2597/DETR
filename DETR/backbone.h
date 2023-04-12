#pragma once

#ifndef MODELS_BACKBONE_H
#define MODELS_BACKBONE_H

#include "nested_tensor.h"
#include "position_encoding.h"
#include "resnet.h"

class BackboneBase : public torch::nn::Module
{
public:
	std::vector<int> strides;
	std::vector<int> num_channels;
	ResNet<BottleNeck<FrozenBatchNorm2dImpl>, FrozenBatchNorm2dImpl> resnet_ = resnet50<FrozenBatchNorm2dImpl>();

	BackboneBase(bool train_backbone = true)
	{
		if (!train_backbone) 
			for (auto v : resnet_.named_parameters())
				if (v.key() != "layer2" || v.key() != "layer3" || v.key() != "layer4")
					v.value().requires_grad_(false);

		strides = { 32 };
		num_channels = { 2048 };
		//resnet_.unregister_module("layer4");
		resnet_.unregister_module("fc");
		resnet_.fc = nullptr;
		resnet_.to(torch::kCUDA);
	}

	NestedTensor forward(NestedTensor nested)
	{
		auto x = this->resnet_.forward(nested.tensors_);
		return { x, nested.masks_ };
		/*if (nested.masks_.defined())
		{
			auto m = nested.masks_;
			//auto size = x.index({torch::indexing::None });
			//std::cout << size.sizes() << std::endl;
			nested.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat32), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{x.size(-2), x.size(-1)})).to(torch::kBool)[0];
		}
		return {x, nested.masks_};*/
	}
};

class Backbone : public BackboneBase
{
public:
	Backbone() : BackboneBase(true)
	{
		
	}
	Backbone(bool train_backbone, bool dilation=false) : BackboneBase(train_backbone) //el return_interm_layers es para panópticos segmentation
	{
		
	}
};

class Joiner
{
private:
	BackboneBase backbone_;
	PositionEncoding position_embedding_;
public:
	Joiner(){}
	Joiner(BackboneBase backbone, PositionEncoding position_embedding)
	{
		backbone_ = backbone;
		position_embedding_ = position_embedding;
	}
	std::tuple<NestedTensor, torch::Tensor> forward(NestedTensor nested)
	{
		auto nest = backbone_.forward(nested);
		auto m = nest.masks_;
		//nest.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{nest.tensors_.size(-2), nest.tensors_.size(-1)})).to(torch::kBool)[0];
		nest.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{nest.tensors_.size(-2), nest.tensors_.size(-1)})).to(torch::kBool).index({torch::indexing::Slice(0)});

		auto pos = position_embedding_.forward(nest).to(nest.tensors_.dtype());
		return std::make_tuple(nest, pos);
		/*nested = backbone_.forward(nested.tensors_);
		auto pos =position_embedding_.forward(nested).to(nested.tensors_.dtype());
		if (pos.defined())
		{
			auto m = pos;
			//auto size = x.index({torch::indexing::None });
			//std::cout << size.sizes() << std::endl;
			nested.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat32), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{nested.tensors_.size(-2), nested.tensors_.size(-1)})).to(torch::kBool)[0];
		}
		return nested;*/
	}
};

#endif