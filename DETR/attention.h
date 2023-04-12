#pragma once

#ifndef LIBS_ATTENTION_H
#define LIBS_ATTENTION_H

#include <torch/torch.h>
/*
 *
 * MultiheadAttention that support query, key, and value to have different dimensions.
 * Query, key, and value projections are removed.
 * Mostly copy-paste from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L873
 * and https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4837
 **/
class MultiHeadAttentionImpl : public torch::nn::Module
{
private:
	int embed_dim_, kdim_, vdim_, num_heads_, head_dim_;
	double dropout_;
	bool qkv_same_embed_dim_;
	torch::nn::Linear out_proj{ nullptr };
public:
	MultiHeadAttentionImpl()
	{
		
	}
	MultiHeadAttentionImpl(int embed_dim, int num_heads, double dropout = 0., bool bias = true, bool add_bias_kv = false, bool add_zero_attn = false, int kdim = -1, int vdim = -1)
	{
		this->embed_dim_ = embed_dim;
		kdim_ = kdim != -1 ? kdim : embed_dim;
		vdim_ = vdim != -1 ? vdim : embed_dim;
		qkv_same_embed_dim_ = kdim_ == embed_dim_ && vdim_ == embed_dim_;
			
		this->num_heads_ = num_heads;
		this->dropout_ = dropout;
		this->head_dim_ = embed_dim_ / num_heads_;
		assert(head_dim_ * num_heads_ == embed_dim_);

		out_proj = torch::nn::Linear(torch::nn::LinearOptions(vdim_, vdim_));

	}
	std::tuple<torch::Tensor, torch::optional<torch::Tensor>> forward(torch::Tensor query, torch::Tensor key, torch::Tensor value = {}, bool need_weights = true, torch::Tensor attn_mask = {})
	{
		if(!qkv_same_embed_dim_)
		{
			
		}else
		{
			
		}
	}
};
TORCH_MODULE(MultiHeadAttention);
#endif