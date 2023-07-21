#pragma once

#ifndef MODELS_BACKBONE_H
#define MODELS_BACKBONE_H

#include <utility>

#include "../libs/util/nested_tensor.h"
#include "../libs/layers.h"
#include "position_encoding.h"
#include "resnet.h"


class BackboneBase : public torch::nn::Module
{
public:
	std::vector<int> strides;
	int num_channels;
	ResNet<BottleNeck<FrozenBatchNorm2dImpl>, FrozenBatchNorm2dImpl> resnet_ = resnet50<FrozenBatchNorm2dImpl>();

	BackboneBase(bool train_backbone = true)
	{
		if (!train_backbone) 
			for (auto v : resnet_.named_parameters())
				if (v.key() != "layer2" || v.key() != "layer3" || v.key() != "layer4")
					v.value() = v.value().requires_grad_(false);

		strides = { 32 };
		num_channels = 2048;
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
	PositionEmbeddingSine position_embedding_;
public:
	Joiner(){}
	Joiner(BackboneBase backbone, PositionEmbeddingSine position_embedding)
	{
		backbone_ = std::move(backbone);
		position_embedding_ = std::move(position_embedding);
	}
	std::pair<NestedTensor, torch::Tensor> forward(const NestedTensor nested)
	{
		auto nest = backbone_.forward(nested);
		auto m = nest.masks_;
		auto sizTens = nest.tensors_.sizes();
		auto interpolate = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat32), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ sizTens[sizTens.size() - 2], sizTens[sizTens.size() - 1] })));
		interpolate = interpolate.to(torch::kBool);
		/*std::cout << "Size interpolate: " << interpolate.sizes() << std::endl;
		std::cout << "Size interpolate[0]: " << interpolate[0].sizes() << std::endl;
		std::cout << "Size interpolate indexing[0]: " << interpolate.index({0})[0].sizes() << std::endl;*/
		nest.masks_ = interpolate.index({ 0 }).to(nest.tensors_.device());
		//nest.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{nest.tensors_.size(-2), nest.tensors_.size(-1)})).to(torch::kBool)[0];
		/*nest.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{nest.tensors_.size(-2), nest.tensors_.size(-1)})).to(torch::kBool).index({torch::indexing::Slice(0)});
		auto siz1 = nest.tensors_.sizes();
		
		std::cout << "Size lero: " << siz1[siz1.size()-2] << std::endl;
		std::cout << "Size lero1: " << siz1[siz1.size() - 1] << std::endl;
		std::cout << "Size mask: " << nest.tensors_.index({torch::indexing::Slice(-2, torch::indexing::None)}).sizes() << std::endl;*/
		//nest.masks_ = torch::nn::functional::interpolate(m.index({ torch::indexing::None }).to(torch::kFloat), torch::nn::functional::InterpolateFuncOptions().size(nest.tensors_.index({torch::indexing::Slice(-2, torch::indexing::None)}).sizes().vec())).to(torch::kBool).index({torch::indexing::Slice(0)});

		auto pos = position_embedding_.forward(nest).to(nest.tensors_.dtype());
		//std::cout << "Pos from Joiner sizes: " << pos.sizes() << std::endl;
		return std::make_pair(nest, pos);
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