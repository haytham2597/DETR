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
		const int64_t b = tensors.size(0);
		const int64_t c = tensors.size(1);
		const int64_t h = tensors.size(2);
		const int64_t w = tensors.size(3);
		this->tensors_ = tensors;
		auto mask = torch::ones({ b,h,w}, torch::kBool).fill_(false).to(tensors_.device());
		this->masks_ = mask;
		//auto pad_img = torch::zeros({ tensors.sizes() }, tensors.dtype());
		//pad_img = pad_img.index({ torch::indexing::Slice(torch::indexing::None, tensors.size(0)), torch::indexing::Slice(torch::indexing::None, tensors.size(1)),torch::indexing::Slice(torch::indexing::None, tensors.size(2)) }).copy_(tensors);
		//mask.index_put_({ torch::indexing::Slice(torch::indexing::None, tensors.size(1)), torch::indexing::Slice(torch::indexing::None, tensors.size(2)) }, false);
		
		//this->masks_ = mask;
	}
	NestedTensor(std::vector<torch::Tensor> tensors)
	{
		
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