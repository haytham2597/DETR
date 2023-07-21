#pragma once

#ifndef MODELS_DEFORMABLE_TRANSFORMER
#define MODELS_DEFORMABLE_TRANSFORMER

#include <torch/torch.h>

class DeformableTransformer : public torch::nn::Module
{
private:
	torch::nn::Linear reference_points{ nullptr };

	torch::nn::Linear enc_output{ nullptr };
	torch::nn::LayerNorm enc_output_norm{ nullptr };
	torch::nn::Linear pos_trans{ nullptr };
	torch::nn::LayerNorm pos_trans_norm{ nullptr };

	deformable_detr::DeformableTransformerEncoder encoder{nullptr};
	deformable_detr::DeformableTransformerDecoder decoder{ nullptr };
	torch::Tensor level_embed_;
	void reset_parameters()
	{
		for(auto p : this->parameters())
		{
			if (p.dim() > 1)
				torch::nn::init::xavier_uniform_(p);
		}
		for(auto m : this->modules())
		{
			if (m->as<MSDeformAttn>())
				m->as<MSDeformAttn>()->reset_parameters();
		}
		if(!two_stage_ && !reference_points.is_empty())
		{
			torch::nn::init::xavier_uniform_(reference_points->weight.data(), 1);
			torch::nn::init::constant_(reference_points->bias.data(), 0.);
		}
		torch::nn::init::normal_(level_embed_);
	}
public:
	bool two_stage_;
	int d_model_;
	
	DeformableTransformer(
		int d_model = 256, 
		int nhead = 8, 
		int num_encoder_layers = 6, 
		int num_decoder_layers = 6, 
		int dim_feedforward = 1024, 
		double dropout = 0.1, 
		std::string activation = "relu", 
		bool return_intermediate_dec = false,
		int num_feature_levels = 4,
		int dec_n_points = 4,
		int enc_n_points = 4,
		bool two_stage = false,
		int two_stage_num_proposals=300)
	{
		this->d_model_ = d_model;
		this->two_stage_ = two_stage;
		auto encoder_layer = deformable_detr::DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points);
		encoder = deformable_detr::DeformableTransformerEncoder(encoder_layer, num_encoder_layers);

		auto decoder_layer = deformable_detr::DeformableTransformDecodeLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points);
		decoder = deformable_detr::DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec);

		level_embed_= register_parameter("level_embed", torch::zeros({ num_feature_levels, d_model }));

		if (two_stage_)
		{
			enc_output = torch::nn::Linear(torch::nn::LinearOptions(d_model_, d_model_));
			enc_output_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}));
			pos_trans = torch::nn::Linear(torch::nn::LinearOptions(d_model_ * 2, d_model_ * 2));
			pos_trans_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model*2 }));

			register_module("enc_output", enc_output);
			register_module("enc_output_norm", enc_output_norm);
			register_module("pos_trans", pos_trans);
			register_module("pos_trans_norm", pos_trans_norm);
		}
		else
		{
			reference_points = torch::nn::Linear(torch::nn::LinearOptions(d_model_, 2));
			register_module("reference_points", reference_points);
		}
		reset_parameters();
	}
	torch::Tensor get_valid_ratios(torch::Tensor mask)
	{
		//size(0) -> Channel
		auto h = mask.size(1);
		auto w = mask.size(2);
		//auto valid_h = torch::sum(~mask.narrow_copy(1, , 2), 1);
		//WARNING: Fix this
		return mask;
	}
	std::tuple<torch::Tensor, torch::Tensor> get_encoder_output_proposals(torch::Tensor memory, torch::Tensor memory_padding_mask, torch::Tensor spatial_shapes)
	{
		auto n = memory.size(0);
		auto s = memory.size(1);
		auto c = memory.size(2);
		double base_scale = 4.0;
		std::vector<torch::Tensor> proposals;
		int cur_ = 0;
		
		for(int i=0;i<spatial_shapes.size(0);i++)
		{
			auto mem_pad = torch::zeros({ memory_padding_mask.size(0), memory_padding_mask.size(1) }, torch::kFloat32);
			auto h = spatial_shapes.size(0);
			auto w = spatial_shapes.size(1);
			for(int j=0;j<memory_padding_mask.size(0);j++)
			{
				for (int k = cur_; k < cur_ + h * w; k++)
					mem_pad[j,k] = memory_padding_mask[j, k];
			}
			mem_pad =mem_pad.view({ n, h,w,1 });

			//memory_padding_mask.narrow_copy(1,cur_, cur_+h*w)
		}
	}
	torch::Tensor forward(torch::Tensor srcs, torch::Tensor masks, torch::Tensor pos_embeds, torch::optional<torch::Tensor> query_embed = torch::nullopt)
	{
		std::vector<torch::Tensor> src_flatten;
		std::vector<torch::Tensor> mask_flatten;
		std::vector<torch::Tensor> lvl_pos_embed_flatten;
		std::vector<torch::Tensor> spatial_shapes;
		for(int i=0;i<static_cast<int>(srcs.size(0));i++)
		{
			auto bs = srcs[i].size(0);
			auto c = srcs[i].size(1);
			auto h = srcs[i].size(2);
			auto w = srcs[i].size(3);
			std::vector<long long> v({ h,w });
			spatial_shapes.push_back(torch::from_blob(v.data(), { 1,2 }, torch::kFloat32));
			auto src = srcs[i].flatten(2).transpose(1, 2);
			auto mask = masks[i].flatten(1);
			auto pos_embed = pos_embeds[i].flatten(2).transpose(1, 2);
			auto lvl_pos_embed = pos_embed + level_embed_[i].view({ 1,1,-1 });
			lvl_pos_embed_flatten.push_back(lvl_pos_embed);
			src_flatten.push_back(src);
			mask_flatten.push_back(mask);
		}

		auto src_flatt = torch::cat(src_flatten, 1);
		auto mask_flatt = torch::cat(mask_flatten, 1);
		auto lvl_pos_embed_flatt = torch::cat(lvl_pos_embed_flatten, 1);
		auto spatial_flatt = torch::cat(spatial_shapes);
		
		auto level_start_index = torch::stack({ spatial_flatt.new_zeros({1}), spatial_flatt.prod(1).cumsum(0).inverse() });
		std::vector<torch::Tensor> valid_ratios;
		for(int i=0;i<masks.size(0);i++)
			valid_ratios.push_back(get_valid_ratios(masks[i]));
		
		auto valid_ratio = torch::cat(valid_ratios, 1);

		auto memory = this->encoder->forward(src_flatt, spatial_flatt, level_start_index, valid_ratio, lvl_pos_embed_flatt, mask_flatt);
		auto bs = memory.size(0);
		auto c = memory.size(2);
		if(two_stage_)
		{
			
		}
	}
};

#endif