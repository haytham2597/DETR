#pragma once

#ifndef LIBS_NESTED_TENSOR_H
#define LIBS_NESTED_TENSOR_H

#include "torch/torch.h"

class NestedTensor {
public:
	torch::Tensor tensors_={};
	torch::Tensor masks_={};

	NestedTensor(torch::Tensor tensors)
	{
		this->tensors_ = tensors;
		auto pad_img = torch::zeros({ 1, tensors.size(0),tensors.size(1),tensors.size(2) }, tensors.dtype());
		auto mask = torch::ones({ 1, 1,tensors.size(1) ,tensors.size(2) }, torch::kBool);
		pad_img = pad_img.index({ torch::indexing::Slice(torch::indexing::None, tensors.size(0)), torch::indexing::Slice(torch::indexing::None, tensors.size(1)),torch::indexing::Slice(torch::indexing::None, tensors.size(2)) }).copy_(tensors);
		mask.index_put_({ torch::indexing::Slice(torch::indexing::None, tensors.size(1)), torch::indexing::Slice(torch::indexing::None, tensors.size(2)) }, false);
		
		this->masks_ = mask;
	}
	NestedTensor(torch::Tensor tensors, torch::Tensor masks) {
		tensors_ = tensors;
		masks_ = masks;
	}
	std::tuple<torch::Tensor, torch::Tensor> decompose() {
		return std::make_tuple(this->tensors_, this->masks_);
	}
	std::tuple<torch::Tensor, torch::Tensor> to(torch::Device dev, bool non_blocking = false) {
		auto cast_tensors = tensors_.to(dev, non_blocking);
		torch::Tensor cast_masks;
		if(this->masks_.defined() && this->masks_.numel() != 0)
			cast_masks = this->masks_.to(dev, non_blocking);
		return std::make_tuple(cast_tensors, cast_masks);
	}
};

#endif