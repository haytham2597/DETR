#pragma once

#ifndef MODELS_POSITION_ENCODING
#define MODELS_POSITION_ENCODING

//#include <torch/torch.h>
#include "nested_tensor.h"


typedef torch::indexing::Slice sli;
constexpr auto non = torch::indexing::None;

class PositionEncoding : public torch::nn::Module
{
public:
	PositionEncoding() = default;
	virtual torch::Tensor forward(NestedTensor nested_tensor){
		return nested_tensor.tensors_;
	}
	//virtual torch::Tensor forward(NestedTensor nested_tensor) = 0;
};

/**
 * \brief [WRONG FUNCTIONAL IN PYTHON THE CUMSUM NOT WORKING]
 * [STABLE WORK HERE]
 */
class PositionEmbeddingSine : public torch::nn::Module
{
private:
	int num_pos_feats_;
	int temperature_;
	bool normalize_;
	double scale_;
	double eps_ = 1e-6;
public:
	PositionEmbeddingSine(int num_pos_feats = 64, int temperature = 10000, bool normalize = true, double scale = -1)
	{
		this->num_pos_feats_ = num_pos_feats;
		this->temperature_ = temperature;
		this->normalize_ = normalize;
		this->scale_ = scale;
		if(this->scale_ == static_cast<double>(-1))
			this->scale_ = 2 * M_PI;
	}
	
	torch::Tensor forward(NestedTensor nested_tensor)
	{
		//auto tuple = nested_tensor.decompose();
		auto x = nested_tensor.tensors_;
		auto mask = nested_tensor.masks_;
		/*std::cout << "x size: " << x.sizes() << std::endl;
		std::cout << "mask size: " << mask.sizes() << std::endl;*/
		if (!mask.defined() || mask.numel() == 0)
			throw std::exception("Mask undefined");
		auto not_mask = ~mask;
		auto y_embed = not_mask.cumsum(1, torch::kFloat32);
		auto x_embed = not_mask.cumsum(2, torch::kFloat32);
		if(this->normalize_)
		{
			y_embed = y_embed / (y_embed.index({ sli(), sli(-1, non), sli() }) + eps_) * scale_;
			x_embed = x_embed / (x_embed.index({ sli(),sli(), sli(-1, non) }) + eps_) * scale_;
		}
		auto dim_t = torch::arange(num_pos_feats_, torch::TensorOptions(torch::kFloat)).to(x.device()); //Add device... Line 48 position_encoding.py
		dim_t = torch::pow(temperature_, (2 * dim_t.div(2, "floor") / num_pos_feats_));
		
		/*std::cout << "Device x_embed: " << x_embed.get_device() << std::endl;
		std::cout << "Device dim_t: " << dim_t.get_device() << std::endl;
		std::cout << "x_embed size: " << x_embed.sizes() << std::endl;
		std::cout << " size: " << dim_t.sizes() << std::endl;*/
		auto pos_x = x_embed.index({ sli(), sli(), sli(), non }).div(dim_t);
		auto pos_y = y_embed.index({ sli(), sli(), sli(), non }).div(dim_t);
		pos_x = torch::stack({ pos_x.index({sli(), sli(), sli(), sli(0, non, 2)}).sin(), pos_x.index({sli(), sli(), sli(), sli(1, non, 2)}).cos() }, 4).flatten(3);
		pos_y = torch::stack({ pos_y.index({sli(), sli(), sli(), sli(0, non, 2)}).sin(), pos_y.index({sli(), sli(), sli(), sli(1, non, 2)}).cos() }, 4).flatten(3);
		auto pos = torch::cat({ pos_y, pos_x }, 3).permute({ 0,3,1,2 });
		return pos;
	}
};

/**
 * \brief
 * [STABLE AND OK]
 */
class PositionEmbeddingLearned : public PositionEncoding
{
private:
	torch::nn::Embedding row_embed{ nullptr }, col_embed{ nullptr };
	static void init_weights(torch::nn::Module& module)
	{
		if (auto* embed = module.as<torch::nn::Embedding>())
			torch::nn::init::uniform_(embed->weight);
	}
public:
	PositionEmbeddingLearned(int num_pos_feats = 256)
	{
		row_embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(50, num_pos_feats));
		col_embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(50, num_pos_feats));
		register_module("row_embed", row_embed);
		register_module("col_embed", col_embed);
		this->apply(init_weights);
	}
	torch::Tensor forward(NestedTensor tensorList) override
	{
		auto x = tensorList.tensors_;
		//h, w = x.shape[-2:]
		//auto siz = x.index({ sli(-2, non) }).sizes();
		auto h = x.size(-2);
		auto w = x.size(-1);
		/*auto h = tensorList.tensors_.size(0);
		auto w = tensorList.tensors_.size(1);*/
		auto i = torch::arange(w, x.device()); //device, position_encoding.py line 76
		auto j = torch::arange(h, x.device());
		auto x_emb = col_embed->forward(i);
		auto y_emb = row_embed->forward(j);
		auto pos = torch::cat({
				x_emb.unsqueeze(0).repeat({h,1,1}),
				y_emb.unsqueeze(1).repeat({1,w,1})
			}, 
		-1).permute({ 2,0,1 }).unsqueeze(0).repeat({ x.size(0), 1,1,1 });

		return pos;
	}
};

#endif