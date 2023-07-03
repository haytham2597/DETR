#pragma once

#ifndef LIBS_NESTED_TENSOR_H
#define LIBS_NESTED_TENSOR_H

#include "torch/torch.h"

class NestedTensor {
public:
	torch::Tensor tensors_={};
	torch::Tensor masks_={};

	NestedTensor(const torch::Tensor& tensors)
	{
		MESSAGE_LOG_ObJ("Sizes tensor:",tensors.sizes())
		//this->tensors_ = tensors;
		this->tensors_ = tensors;
		auto mask = torch::ones({ tensors.size(0),tensors.size(2),tensors.size(3) }, torch::kBool).fill_(false).to(tensors_.device());
		this->masks_ = mask;
		/*auto tensor = torch::zeros(tensors.sizes(),  torch::TensorOptions(tensors.dtype()).device(tensors.device()));
		auto mask = torch::ones(tensors.sizes(), torch::TensorOptions(torch::kBool).device(tensors.device()));
		torch::Tensor pad_img = torch::zeros(tensors.sizes(), torch::TensorOptions(tensors.dtype()).device(tensors.device()));
		MESSAGE_LOG_ObJ("TensorSize:", tensor.sizes())
		MESSAGE_LOG_ObJ("MaskSize:", mask.sizes())
		MESSAGE_LOG_ObJ("pad_img:", pad_img.sizes())
		for(int i=0;i<tensors.size(0);i++)
		{
			pad_img.copy_(tensors[i].index({ tensors[i].size(0), tensors[i].size(1) ,tensors[i].size(2)}));
			mask.index_put_({ tensors[i].size(1), tensors[i].size(2) }, false);
		}
		this->tensors_ = tensor;
		this->masks_ = mask;*/
	}
	NestedTensor(torch::Tensor tensors, torch::Tensor masks) {
		tensors_ = tensors;
		masks_ = masks;
	}
	std::tuple<torch::Tensor, torch::Tensor> decompose() {
		return std::make_tuple(this->tensors_, this->masks_);
	}
	std::tuple<torch::Tensor, torch::Tensor> to(torch::Device dev, bool non_blocking = false) {
		tensors_ = tensors_.to(dev, non_blocking);
		if (this->masks_.defined() && this->masks_.numel() != 0)
			this->masks_ = this->masks_.to(dev, non_blocking);

		/*auto cast_tensors = tensors_.to(dev, non_blocking);
		torch::Tensor cast_masks;
		if(this->masks_.defined() && this->masks_.numel() != 0)
			cast_masks = this->masks_.to(dev, non_blocking);*/
		return std::make_tuple(tensors_, masks_);
	}
};

#endif