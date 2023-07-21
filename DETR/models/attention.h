#pragma once

#ifndef LIBS_ATTENTION_H
#define LIBS_ATTENTION_H

#include <torch/torch.h>
//#include "m_assert.h"
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
	torch::Tensor in_proj_bias_{}, in_proj_weight_{}, bias_k_{}, bias_v_{}, q_proj_weight_{}, k_proj_weight{}, v_proj_weight{};
	bool add_zero_attn_ = false;
	bool training_ = true;
	void reset_parameters()
	{
		torch::nn::init::constant_(this->out_proj->bias, 0.);
	}
public:
	MultiHeadAttentionImpl() = default;

	MultiHeadAttentionImpl(int embed_dim, int num_heads, double dropout = 0., bool bias = true, bool add_bias_kv = false, bool add_zero_attn = false, int kdim = -1, int vdim = -1)
	{
		this->embed_dim_ = embed_dim;
		kdim_ = kdim != -1 ? kdim : embed_dim;
		vdim_ = vdim != -1 ? vdim : embed_dim;
		qkv_same_embed_dim_ = kdim_ == embed_dim_ && vdim_ == embed_dim_;
			
		this->num_heads_ = num_heads;
		this->dropout_ = dropout;
		this->head_dim_ = embed_dim / num_heads_;
		assert(head_dim_ * num_heads_ == embed_dim_);
		out_proj = torch::nn::Linear(torch::nn::LinearOptions(vdim_, vdim_));
		reset_parameters();
	}

	std::tuple<torch::Tensor, torch::optional<torch::Tensor>> multi_head_attention_forward(
		torch::Tensor query,
		torch::Tensor key,
		torch::Tensor value,
		int embed_dim_to_check,
		int num_heads,
		torch::Tensor in_proj_weight,
		torch::Tensor in_proj_bias,
		bool add_zero_attn,
		float dropout_p,
		torch::Tensor out_proj_weight,
		torch::Tensor out_proj_bias,
		int out_dim,
		bool training = true,
		bool need_weights = true,
		bool use_separate_proj_weight = false,
		torch::Tensor q_proj_weight = {},
		torch::Tensor k_proj_weight = {},
		torch::Tensor v_proj_weight = {},
		torch::Tensor static_k = {},
		torch::Tensor static_v = {},
		torch::Tensor attn_mask = {},
		torch::Tensor key_padding_mask = {},
		torch::Tensor bias_v = {},
		torch::Tensor bias_k = {}
	)
	{
		
		auto tgt_len = query.size(0);
		auto bsz = query.size(1);
		const auto embed_dimension = query.size(2);
		//MESSAGE_LOG("Embed_Dim: " + std::to_string(embed_dimension))
		assert(embed_dimension == embed_dim_to_check);
		assert(key.size(0) == value.size(0) && key.size(1) == value.size(1));

		auto head_dim = embed_dimension / num_heads;
		auto v_head_dim = out_dim / num_heads;
		//embed_dim must be divisible by num_heads
		//M_Assert(head_dim * num_heads_ == embed_dim, "embed_dim must be divisible by num_heads");

		auto scaling = std::pow(static_cast<double>(head_dim), -0.5);

		auto q = query * scaling;
		auto k = key;
		auto v = value;
		if(attn_mask.defined())
		{
			//M_Assert(attn_mask.dtype() == torch::kFloat32 || attn_mask.dtype() == torch::kFloat64 || attn_mask.dtype() == torch::kFloat16 || attn_mask.dtype() == torch::kUInt8 || attn_mask.dtype() == torch::kBool, "Only float, bool and int supported not: " + std::to_string(attn_mask.dtype().name()));
			if(attn_mask.dtype() == torch::kUInt8)
				attn_mask = attn_mask.to(torch::kBool);
			if(attn_mask.dim() == 2)
			{
				attn_mask = attn_mask.unsqueeze(0);
				//List runtime error attention.py line 297
			}
		}
		if(key_padding_mask.defined() && key_padding_mask.dtype() == torch::kUInt8)
			key_padding_mask = key_padding_mask.to(torch::kBool);
		if(bias_k.defined() && bias_v.defined())
		{
			if(static_k.defined() && static_v.defined())
			{
				k = torch::cat({ k, bias_k.repeat({1,bsz, 1}) });
				v = torch::cat({ v, bias_v.repeat({1,bsz, 1}) });
				if(attn_mask.defined())
					attn_mask = torch::nn::functional::pad(attn_mask, torch::nn::functional::PadFuncOptions(std::vector<int64_t>({ 0,1 })));
				if(key_padding_mask.defined())
					key_padding_mask = torch::nn::functional::pad(key_padding_mask, torch::nn::functional::PadFuncOptions(std::vector<int64_t>({ 0,1 })));
			}else
			{
				throw std::exception("Static V and K is none");
			}
		}else
		{
			//throw std::exception("Bias k and v is none");
		}

		q = q.contiguous().view({ tgt_len, bsz * num_heads, head_dim }).transpose(0, 1);
		if (k.defined())
			k = k.contiguous().view({ -1, bsz * num_heads, head_dim }).transpose(0, 1);
		if (v.defined())
			v = v.contiguous().view({ -1, bsz * num_heads, v_head_dim }).transpose(0, 1);
		
		if(static_k.defined())
		{
			assert(static_k.size(0) == bsz * num_heads);
			assert(static_v.size(2) == head_dim);
			k = static_k;
		}
		if(static_v.defined())
		{
			assert(static_v.size(0) == bsz * num_heads);
			assert(static_v.size(2) == v_head_dim);
			v = static_v;
		}
		auto src_len = k.size(1);
		if(key_padding_mask.defined())
		{
			assert(key_padding_mask.size(0) == bsz);
			assert(key_padding_mask.size(1) == src_len);
		}
		/*if(add_zero_attn)
		{
			src_len += 1;
			k = torch::cat({k, torch::zeros({k.size(0), 1})})
		}*/

		auto attn_output_weights = torch::bmm(q, k.transpose(1,2));
		if(attn_mask.defined())
		{
			if (attn_mask.dtype() == torch::kBool)
				attn_output_weights.masked_fill_(attn_mask, float(-INFINITY));
			else
				attn_output_weights += attn_mask;
		}
		if(key_padding_mask.defined())
		{
			attn_output_weights = attn_output_weights.view({ bsz, num_heads, tgt_len, src_len });
			attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float(-INFINITY));
			attn_output_weights = attn_output_weights.view({ bsz * num_heads, tgt_len, src_len });
		}
		attn_output_weights = torch::softmax(attn_output_weights, -1);
		attn_output_weights = torch::dropout(attn_output_weights, dropout_p, training);
		auto attn_output = torch::bmm(attn_output_weights, v);

		attn_output = attn_output.transpose(0, 1).contiguous().view({ tgt_len, bsz, out_dim});

		/*std::cout << "Attn_Output device: " << attn_output.get_device() << std::endl;
		std::cout << "out_proj_weight device: " << out_proj_weight.get_device() << std::endl;
		std::cout << "out_proj_bias device: " << out_proj_bias.get_device() << std::endl;*/
		out_proj_weight = out_proj_weight.to(torch::kCUDA);
		out_proj_bias = out_proj_bias.to(torch::kCUDA);
		attn_output = torch::nn::functional::linear(attn_output, out_proj_weight, out_proj_bias);
		//attn_output = torch::linear(attn_output, out_proj_weight, out_proj_bias);
		if(need_weights)
		{
			attn_output_weights = attn_output_weights.view({ bsz, num_heads, tgt_len, src_len });
			return std::make_tuple(attn_output, attn_output_weights.sum(1) / num_heads);
		}
		return std::make_tuple(attn_output, torch::Tensor{});
	}
	std::tuple<torch::Tensor, torch::optional<torch::Tensor>> forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor key_padding_mask = {}, bool need_weights = true, torch::Tensor attn_mask = {})
	{
		if(!qkv_same_embed_dim_)
		{
			return multi_head_attention_forward(query, key, value, embed_dim_, num_heads_, in_proj_weight_, in_proj_bias_, add_zero_attn_, dropout_, out_proj->weight, out_proj->bias, vdim_, training_, need_weights, true, q_proj_weight_, k_proj_weight, v_proj_weight, {},{},attn_mask, key_padding_mask);
		}
		else
		{
			return multi_head_attention_forward(query, key, value, embed_dim_, num_heads_, in_proj_weight_, in_proj_bias_, add_zero_attn_, dropout_, out_proj->weight, out_proj->bias, vdim_, training_, need_weights, false, {}, {}, {}, {}, {}, attn_mask, key_padding_mask);
		}
	}
};
TORCH_MODULE(MultiHeadAttention);
#endif