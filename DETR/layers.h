#pragma once

#ifndef LIBS_LAYERS_H
#define LIBS_LAYERS_H

#include <torch/torch.h>

#include "attention.h"
#include "ext.h"

/**
 * \brief [STABLE WORK]
 */
struct FrozenBatchNorm2dImpl : public torch::nn::Module
{
	double Eps;
	torch::Tensor weight, bias, running_var, running_mean;
	//FrozenBatchNorm2dImpl(){}
	FrozenBatchNorm2dImpl(int n, const double eps = 1e-5)
	{
		weight = this->register_buffer("weight", torch::ones({ n }));
		bias = this->register_buffer("bias", torch::zeros({ n }));
		running_mean = this->register_buffer("running_mean", torch::zeros({ n }));
		running_var = this->register_buffer("running_var", torch::ones({ n }));
		this->Eps = eps;
		//std::cout << "Frozen is initialized IMPL sizes weight: " << weight.sizes() << std::endl;
	}
	
	torch::Tensor forward(torch::Tensor x)
	{
		//for(auto v : this->buffers())
		//{
		//	std::cout << v.names() << std::endl;
		//	if (v.name() == "weight")
		//		w = v.reshape({1,-1,1,1});
		//	if (v.name() == "bias")
		//		b = v.reshape({1,-1,1,1});
		//	if (v.name() == "running_var")
		//		rv = v.reshape({ 1,-1,1,1 });
		//	if (v.name() == "running_mean")
		//		rm = v.reshape({ 1,-1,1,1 });
		//}
		auto w = weight.reshape({1,-1,1,1});
		auto b = bias.reshape({1,-1,1,1});
		auto rv = running_var.reshape({1,-1,1,1});
		auto rm = running_mean.reshape({1,-1,1,1});
		const auto scale = w * (rv + this->Eps).rsqrt();
		const auto bias = b - rm * scale;
		return x * scale + bias;
	}
};

TORCH_MODULE(FrozenBatchNorm2d);

/**
 * \brief Multi Layer Perceptron (also called FFN) [Theorically STABLE]
 */
struct MLPImpl : public torch::nn::Module
{
	torch::nn::Sequential layers;
	MLPImpl(int input_dim, int hidden_dim, int output_dim, int num_layers)
	{
		for(int i=0;i<num_layers;i++)
		{
			if (i == 0) {
				layers->push_back(torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim).bias(true)));
				layers->push_back(torch::nn::ReLU(torch::nn::ReLUOptions()));
				continue;
			}
			if(i == num_layers-1) //Last
			{
				layers->push_back(torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, output_dim).bias(true)));
				break;
			}
			layers->push_back(torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim).bias(true)));
			layers->push_back(torch::nn::ReLU(torch::nn::ReLUOptions()));
		}
		register_module("layer", layers);
	}
	torch::Tensor forward(torch::Tensor x)
	{
		return layers->forward(x);
	}
};

TORCH_MODULE(MLP);

//MS Deformation Attention
struct MSDeformAttnImpl : public torch::nn::Module
{
	int n_heads_, n_levels_, n_points_;
	void reset_parameters()
	{
		torch::nn::init::constant_(sampling_offsets->weight.data(), 0.);
		auto thetas = torch::arange(n_heads_, torch::kFloat32) * (2.0 * M_PI / n_heads_);
		auto grid_init = torch::cat({ thetas.cos(), thetas.sin() }, -1);
		grid_init = (grid_init / std::get<0>(grid_init.abs().max(-1, true))).view({ n_heads_, 1,1,2 }).repeat({ 1, n_levels_, n_points_, 1 });
		for (int i = 0; i < grid_init.size(0); i++)
		for (int j = 0; j < grid_init.size(1); j++)
		for (int n = 0; n < this->n_points_; n++)
		for (int k = 0; k < grid_init.size(3); k++)
			grid_init[i][j][n][k] *= n + 1;

		//TODO: Research ms_deform_attn.py line 70
		sampling_offsets->bias = grid_init.view({ -1 });

		torch::nn::init::constant_(attention_weights->weight.data(), 0.);
		torch::nn::init::constant_(attention_weights->bias.data(), 0.);
		torch::nn::init::xavier_uniform_(value_proj->weight.data());
		torch::nn::init::constant_(value_proj->bias.data(), 0.);
		torch::nn::init::xavier_uniform_(output_proj->weight.data());
		torch::nn::init::constant_(output_proj->bias.data(), 0.);
	}
	torch::nn::Linear sampling_offsets{ nullptr }, attention_weights{ nullptr }, value_proj{ nullptr }, output_proj{ nullptr };
	MSDeformAttnImpl(int d_model = 256, int n_levels = 4, int n_heads = 8, int n_points = 4)
	{
		if (d_model % n_heads != 0)
			throw std::exception("d_model must be divisible by n_heads");
		auto d_per_head = d_model / n_heads;
		if (!is_power_of_2(d_per_head))
			std::cout << "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation." << std::endl;

		int im2col_step = 64;
		this->n_heads_ = n_heads;
		this->n_levels_ = n_levels;
		this->n_points_ = n_points;
		sampling_offsets = torch::nn::Linear(torch::nn::LinearOptions(d_model, n_heads * n_levels * n_points*2));
		attention_weights = torch::nn::Linear(torch::nn::LinearOptions(d_model, n_heads * n_levels * n_points));
		value_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
		output_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
		register_module("sampling_offsets", sampling_offsets);
		register_module("attention_weights", attention_weights);
		register_module("value_proj", value_proj);
		register_module("output_proj", output_proj);
	}
	torch::Tensor forward(torch::Tensor query, torch::Tensor reference_points, torch::Tensor input_flatten, torch::Tensor input_spatial_shapes, torch::Tensor input_level_start_index, torch::optional<torch::Tensor> input_padding_mask = torch::nullopt)
	{
		//TODO: Determine ms_deform_attn.py 92
		/*
		:param query(N, Length_{ query }, C)
		: param reference_points(N, Length_{ query }, n_levels, 2), range in[0, 1], top - left(0, 0), bottom - right(1, 1), including padding area
			or (N, Length_{ query }, n_levels, 4), add additional(w, h) to form reference boxes
		: param input_flatten(N, \sum_{ l = 0 }^ {L - 1} H_l \cdot W_l, C)
		: param input_spatial_shapes(n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{ L - 1 }, W_{ L - 1 })]
		: param input_level_start_index(n_levels, ), [0, H_0 * W_0, H_0 * W_0 + H_1 * W_1, H_0 * W_0 + H_1 * W_1 + H_2 * W_2, ..., H_0 * W_0 + H_1 * W_1 + ... + H_{ L - 1 }*W_{ L - 1 }]
		: param input_padding_mask(N, \sum_{ l = 0 }^ {L - 1} H_l \cdot W_l), True for padding elements, False for non - padding elements

		: return output(N, Length_{ query }, C)
		*/
		int N = query.size(0);
		int Len_Q = query.size(1);
		auto value = this->value_proj->forward(input_flatten);
		auto sampl_offsets = this->sampling_offsets->forward(query).view({ N, Len_Q, this->n_heads_, this->n_levels_, this->n_points_, 2 });
		auto attn_weights = this->attention_weights->forward(query).view({ N, Len_Q, this->n_heads_, this->n_levels_ * this->n_points_ });
		attn_weights = torch::softmax(attn_weights, -1).view({ N, Len_Q, this->n_heads_, this->n_levels_, this->n_points_ });
		if(reference_points.size(-1) == 2)
		{
			//TODO: ms_deform_attn.py 103
			//auto offset_normalizer = torch::stack({input_spatial_shapes[]})
		}
	}
};
TORCH_MODULE(MSDeformAttn);

namespace deformable_detr {
	struct DeformableTransformerEncoderLayerImpl : public torch::nn::Module
	{
		MSDeformAttn attn{ nullptr };
		torch::nn::Dropout2d dropout2d{ nullptr };
		torch::nn::LayerNorm layer_norm{ nullptr };
		torch::nn::Sequential ffn_half;
		DeformableTransformerEncoderLayerImpl(int d_model = 256, int d_ffn = 1024, double dropout = 0.1, std::string activation = "relu", int n_levels = 4, int n_heads = 8, int n_points = 4)
		{
			//Self attention
			attn = MSDeformAttn(d_model, n_levels, n_heads, n_points);
			dropout2d = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));

			//ffn
			ffn_half->push_back(torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ffn)));
			ffn_half->push_back(torch::nn::ReLU(torch::nn::ReLU())); //in deformable_transformer.py 203 have 3 possible activation relu, glu, etc
			ffn_half->push_back(torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout)));
			ffn_half->push_back(torch::nn::Linear(torch::nn::LinearOptions(d_ffn, d_model)));
			//dropout2d3 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
		}
		torch::Tensor forward_ffn(torch::Tensor x)
		{
			auto src2 = this->ffn_half->forward(x);
			auto src = x + this->dropout2d->forward(src2);
			return layer_norm->forward(src);
		}
		torch::Tensor forward(torch::Tensor src, torch::Tensor pos, torch::Tensor reference_points, torch::Tensor spatial_shapes, torch::Tensor level_start_index, torch::optional<torch::Tensor> padding_mask = torch::nullopt)
		{
			auto src2 = this->attn->forward(with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask);
			src = src + dropout2d->forward(src2);
			src = layer_norm->forward(src);
			return forward_ffn(src);
		}
	};
	TORCH_MODULE(DeformableTransformerEncoderLayer);

	struct DeformableTransformerEncoderImpl : public torch::nn::Module
	{
		DeformableTransformerEncoderLayer encoder{ nullptr };

		torch::nn::Sequential layers;
		int num_layers_;
		DeformableTransformerEncoderImpl()
		{
			
		}
		DeformableTransformerEncoderImpl(DeformableTransformerEncoderLayer encoder_layer, int num_layers)
		{
			encoder = encoder_layer;
			//layers = get_clone(encoder_layer, num_layers);
			this->num_layers_ = num_layers;
		}

		torch::Tensor reference_points(torch::Tensor spatial_shapes, torch::Tensor valid_ratios)
		{
			std::vector<torch::Tensor> reference_lists;
			for (int i = 0; i < spatial_shapes.size(0); i++)
			{
				int h = spatial_shapes[i].size(0);
				int w = spatial_shapes[i].size(1);
				auto mesh = torch::meshgrid({ torch::linspace(0.5, h - 0.5, h, torch::kFloat32),torch::linspace(0.5, w - 0.5, w, torch::kFloat32) });
				auto ref_y = mesh[0].reshape({ -1 }); //todo: [none] what? deformable_transformer.py 244
				auto ref_x = mesh[1];
				auto ref = torch::stack({ ref_x, ref_y }, -1);
				reference_lists.push_back(ref);
			}
			return torch::cat({ reference_lists });
		}

		torch::Tensor forward(torch::Tensor src, torch::Tensor spatial_shapes, torch::Tensor level_start_index, torch::Tensor valid_ratios, torch::optional<torch::Tensor> pos = torch::nullopt, torch::optional<torch::Tensor> padding_masks = torch::nullopt)
		{
			auto out = src;
			auto reference_p = reference_points(spatial_shapes, valid_ratios);
			if (pos.has_value())
				for (int i = 0; i < num_layers_; i++)
					out = this->encoder->forward(out, pos.value(), reference_p, spatial_shapes, level_start_index, padding_masks);
			return out;
		}
	};
	TORCH_MODULE(DeformableTransformerEncoder);

	struct DeformableTransformDecodeLayerImpl : public torch::nn::Module
	{
		MSDeformAttn cross_attn{ nullptr };
		torch::nn::Dropout2d dropout2d{ nullptr };
		torch::nn::LayerNorm layer_norm{ nullptr };
		torch::nn::MultiheadAttention mha{ nullptr };
		torch::nn::Sequential ffn_half;
		/*template<typename Contained>
		torch::nn::Module activ{nullptr};*/

		DeformableTransformDecodeLayerImpl(int d_model = 256, int d_ffn = 1024, double dropout = 0.1, std::string activation = "relu", int n_levels = 4, int n_heads = 8, int n_points = 4)
		{
			/*if(activation == "relu")
			activ = get_activation<torch::nn::ReLUImpl>(activation);*/
			//cross attention
			cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points);
			dropout2d = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));

			//self attention
			mha = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(d_model, n_heads).dropout(dropout));

			//ffn
			ffn_half->push_back(torch::nn::Linear(torch::nn::LinearOptions(d_model, d_ffn)));

			ffn_half->push_back(torch::nn::ReLU(torch::nn::ReLU())); //in deformable_transformer.py 273 have 3 possible activation relu, glu, etc
			ffn_half->push_back(torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout)));
			ffn_half->push_back(torch::nn::Linear(torch::nn::LinearOptions(d_ffn, d_model)));
		}
		torch::Tensor with_pos_embed(torch::Tensor tensor, torch::optional<torch::Tensor> pos)
		{
			if (!pos.has_value()) //epmpty TODO:TORCH.LUMEL EMPTY
				return tensor;
			return tensor + pos.value();
		}
		torch::Tensor forward_ffn(torch::Tensor x)
		{
			auto tgt2 = ffn_half->forward(x);
			auto tgt = x + dropout2d->forward(tgt2);
			return layer_norm->forward(tgt);
		}
		torch::Tensor forward(torch::Tensor tgt, torch::optional<torch::Tensor> query_pos, torch::Tensor reference_points, torch::Tensor src, torch::Tensor src_spatial_shapes, torch::Tensor level_Start_index, torch::optional<torch::Tensor> src_padding_mask = torch::nullopt)
		{
			//self attention
			auto q = with_pos_embed(tgt, query_pos);
			auto tgt2 = std::get<0>(mha->forward(q.transpose(0, 1), q.transpose(0, 1), tgt.transpose(0, 1))).transpose(0, 1);
			tgt = tgt + dropout2d->forward(tgt2);
			tgt = layer_norm->forward(tgt);
			tgt2 = cross_attn->forward(with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_Start_index, src_padding_mask);
			tgt = tgt + dropout2d->forward(tgt2);
			tgt = layer_norm->forward(tgt);

			//ffn
			return forward_ffn(tgt);
		}
	};
	TORCH_MODULE(DeformableTransformDecodeLayer);

	/**
	 * \brief [UNFINISHED]
	 */
	struct DeformableTransformerDecoderImpl : public torch::nn::Module
	{
		DeformableTransformDecodeLayer decoder{ nullptr };
		int num_layers_;
		bool return_intermediate_ = false;
		MLP bbox_embed{ nullptr }; //en deformable_detr.py 55
		torch::nn::Linear class_embed{ nullptr };  //en deformable_detr.py 54
		DeformableTransformerDecoderImpl()
		{
			
		}
		DeformableTransformerDecoderImpl(DeformableTransformDecodeLayer decoder_layer, int num_layers, bool return_intermediate = false)
		{
			decoder = decoder_layer;
			this->num_layers_ = num_layers;
			this->return_intermediate_ = return_intermediate;
			//class_embed = torch::nn::Linear(torch::nn::LinearOptions())
		}
		std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor tgt, torch::Tensor reference_points, torch::Tensor src, torch::Tensor src_spatial_shapes, torch::Tensor src_level_start_index, torch::Tensor src_valid_ratios, torch::optional<torch::Tensor> query_pos = torch::nullopt, torch::optional<torch::Tensor> src_padding_mask = torch::nullopt)
		{
			auto output = tgt;
			std::vector<torch::Tensor> intermediate;
			std::vector<torch::Tensor> intermediate_reference_points;
			for (int i = 0; i < this->num_layers_; i++)
			{

				torch::Tensor reference_points_inputs;
				if (reference_points.size(-1) == 4)
				{
					//reference_points_input = reference_points[:, : , None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]

					auto refe = torch::from_blob(reference_points.data_ptr(), { reference_points.size(0), reference_points.size(1) }, torch::TensorOptions().dtype(torch::kFloat32));
					auto reference_points_inputs = refe * torch::cat({ src_valid_ratios, src_valid_ratios }, -1);
					reference_points_inputs = torch::from_blob(reference_points_inputs.data_ptr(), { reference_points_inputs.size(0) }, { torch::kFloat32 });
				}
				else
				{
					assert(reference_points.size(-1) == 2);
					auto refe = torch::from_blob(reference_points.data_ptr(), { reference_points.size(0), reference_points.size(1) }, torch::TensorOptions().dtype(torch::kFloat32));
					auto valid = torch::from_blob(src_valid_ratios.data_ptr(), { src_valid_ratios.size(0) }, { torch::kFloat32 });
					reference_points_inputs = refe * valid;
				}
				output = decoder->forward(output, query_pos, reference_points_inputs, src, src_spatial_shapes, src_level_start_index, src_padding_mask);

				if (!bbox_embed.is_empty())
				{
					//tmp = self.bbox_embed[lid](output) index LID?? deformable_transformer.py 342
					torch::Tensor new_reference_points;
					auto tmp = bbox_embed->forward(output);
					if (reference_points.size(-1) == 4)
					{
						new_reference_points = tmp + inverse_sigmoid(reference_points).sigmoid();
					}
					else
					{
						assert(reference_points.size(-1) == 2);
						new_reference_points = tmp;
						//deformable_transformer.py 349
						/*new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
						new_reference_points = new_reference_points.sigmoid()*/
					}
					reference_points = new_reference_points.detach();
				}

				if (this->return_intermediate_)
				{
					intermediate.push_back(output);
					intermediate_reference_points.push_back(reference_points);
				}

			}
			if (this->return_intermediate_)
				return std::make_tuple(torch::stack(intermediate), torch::stack(intermediate_reference_points));
			std::make_tuple(output, reference_points);
		}
	};
	TORCH_MODULE(DeformableTransformerDecoder);
}

namespace conditional_detr
{
	struct TransformerEncoderLayerImpl : public torch::nn::Module
	{
		torch::nn::MultiheadAttention self_attn{ nullptr };
		torch::nn::Linear linear1{ nullptr }, linear2{ nullptr };
		torch::nn::Dropout2d dropout0{ nullptr }, dropout1{nullptr}, dropout2{nullptr};
		torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr };
		bool normalize_before_;
		
		TransformerEncoderLayerImpl(int d_model, int nhead, int dim_feedforward = 2048, double dropout = 0.1, std::string activation="relu", bool normalize_before = false)
		{
			this->normalize_before_ = normalize_before;

			self_attn = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout));
			linear1 = torch::nn::Linear(torch::nn::LinearOptions(d_model, dim_feedforward));
			linear2 = torch::nn::Linear(torch::nn::LinearOptions(dim_feedforward, d_model));

			dropout0 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			dropout1 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			dropout2= torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));

			norm1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));
			norm2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));

			register_module("multiheadattention", self_attn);
			register_module("linear1", linear1);
			register_module("linear2", linear2);
			register_module("dropout0", dropout0);
			register_module("dropout1", dropout1);
			register_module("dropout2", dropout2);
			register_module("norm1", norm1);
			register_module("norm2", norm2);
		}

		torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}, torch::Tensor src_key_padding_mask = {}, torch::Tensor pos = {})
		{
			if(this->normalize_before_)
			{
				auto src2 = norm1->forward(src);
				auto q = with_pos_embed(src2, pos);
				auto k = q;
				src2 = std::get<0>(this->self_attn->forward(q, k, src2, src_key_padding_mask, true, src_mask));
				src = src + dropout1->forward(src2);
				src2 = norm2->forward(src);
				src2 = linear2->forward(dropout0->forward(torch::nn::ReLU(torch::nn::ReLUOptions())->forward(linear1->forward(src2))));
				return src + dropout2->forward(src2);
			}
			/*std::cout << "SRC Size: " << src.sizes() << std::endl;
			std::cout << "POS Size: " << pos.sizes() << std::endl;*/

			auto q = with_pos_embed(src, pos);
			auto k = q;
			auto src2 = std::get<0>(self_attn->forward(q, k, src, src_key_padding_mask, true, src_mask));
			src = src + dropout1->forward(src2);
			src = norm1->forward(src);
			src2 = linear2->forward(dropout0->forward(torch::nn::ReLU(torch::nn::ReLUOptions())->forward(linear1->forward(src))));
			src = src + dropout2->forward(src2);
			src = norm2->forward(src);
			return src;
		}
	};
	TORCH_MODULE(TransformerEncoderLayer);

	struct TransformerDecoderLayerImpl : public torch::nn::Module
	{
		//torch::nn::MultiheadAttention self_attn{ nullptr }, cross_attn{ nullptr };
		MultiHeadAttention self_attn{ nullptr }, cross_attn{ nullptr };
		torch::nn::Linear sa_qcontent_proj{ nullptr }, sa_qpos_proj{ nullptr }, sa_kcontent_proj{ nullptr }, sa_kpos_proj{ nullptr }, sa_v_proj{ nullptr };
		torch::nn::Linear ca_qcontent_proj{ nullptr }, ca_qpos_proj{ nullptr }, ca_kcontent_proj{ nullptr }, ca_kpos_proj{ nullptr }, ca_v_proj{ nullptr }, ca_qpos_sine_proj{nullptr};
		torch::nn::Linear linear1{ nullptr }, linear2{ nullptr };
		torch::nn::Dropout2d dropout1{ nullptr }, dropout2{ nullptr }, dropout3{ nullptr }, dropout4{ nullptr };
		torch::nn::LayerNorm norm1{ nullptr }, norm2{ nullptr }, norm3{ nullptr };
		torch::nn::ReLU activation{nullptr};
		bool normalize_before_;
		int nhead_;
	
		TransformerDecoderLayerImpl(int d_model, int nhead, int dim_feedforward = 2048, double dropout = 0.1, bool normalize_before = false)
		{
			activation = torch::nn::ReLU(torch::nn::ReLUOptions());
			//Decoder Self-Attention
			sa_qcontent_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			sa_qpos_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			sa_kcontent_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			sa_kpos_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			sa_v_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			//self_attn = MultiHeadAttention(d_model, nhead, dropout, d_model);
			self_attn = MultiHeadAttention(d_model, nhead, dropout, true, false, false, -1, d_model);
			//self_attn = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout).vdim(d_model));

			//Decoder Cross-Attention
			ca_qcontent_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			ca_qpos_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			ca_kcontent_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			ca_kpos_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			ca_v_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));
			ca_qpos_sine_proj = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model));

			//cross_attn = MultiHeadAttention(d_model * 2, nhead, dropout, d_model);
			cross_attn = MultiHeadAttention(d_model * 2, nhead, dropout, true, false, false, -1, d_model);
			//cross_attn = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(static_cast<int64_t>(d_model*2), nhead).dropout(dropout).vdim(d_model));

			this->normalize_before_ = normalize_before;
			this->nhead_ = nhead;
			//FeedForward model
			linear1 = torch::nn::Linear(torch::nn::LinearOptions(d_model, dim_feedforward));
			linear2 = torch::nn::Linear(torch::nn::LinearOptions(dim_feedforward, d_model));
			dropout1 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			dropout2 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			dropout3 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));
			dropout4 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropout));

			norm1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}));
			norm2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));
			norm3 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }));

			
			register_module("sa_qcontent_proj", sa_qcontent_proj);
			register_module("sa_qpos_proj", sa_qpos_proj);
			register_module("sa_kcontent_proj", sa_kcontent_proj);
			register_module("sa_kpos_proj", sa_kpos_proj);
			register_module("sa_v_proj", sa_v_proj);
			register_module("self_attn", self_attn);

			register_module("ca_qcontent_proj", ca_qcontent_proj);
			register_module("ca_qpos_proj", ca_qpos_proj);
			register_module("ca_kcontent_proj", ca_kcontent_proj);
			register_module("ca_kpos_proj", ca_kpos_proj);
			register_module("ca_v_proj", ca_v_proj);
			register_module("ca_qpos_sine_proj", ca_qpos_sine_proj);
			register_module("cross_attn", cross_attn);

			register_module("linear1", linear1);
			register_module("dropout1", dropout1);
			register_module("linear2", linear2);

			register_module("norm1", norm1);
			register_module("norm2", norm2);
			register_module("norm3", norm3);
			
			register_module("dropout2", dropout2);
			register_module("dropout3", dropout3);
			register_module("dropout4", dropout4);

			register_module("activation", activation);

		}

		torch::Tensor forward(torch::Tensor tgt, torch::Tensor memory, torch::Tensor tgt_mask = {}, torch::Tensor memory_mask = {}, torch::Tensor tgt_key_padding_mask = {}, torch::Tensor memory_key_padding_mask = {}, torch::Tensor pos = {}, torch::Tensor query_pos = {}, torch::Tensor query_sine_embed = {}, bool is_first = false)
		{
			if (this->normalize_before_)
			{
				auto tgt2 = this->norm1->forward(tgt);
				auto q = with_pos_embed(tgt2, query_pos);
				auto k = q;
				tgt2 = std::get<0>(self_attn->forward(q, k, tgt2, tgt_key_padding_mask, true, tgt_mask));
				tgt = tgt + this->dropout2->forward(tgt2);
				tgt2 = norm2->forward(tgt);
				tgt2 = std::get<0>(this->cross_attn->forward(with_pos_embed(tgt2, query_pos), with_pos_embed(memory, pos), memory, memory_key_padding_mask, true, memory_mask));
				tgt = tgt + dropout3->forward(tgt2);
				tgt2 = norm3->forward(tgt);
				tgt2 = linear2->forward(dropout1->forward(activation->forward(linear1->forward(tgt2))));
				tgt = tgt + dropout4->forward(tgt2);
				return tgt;
			}
			/*if (tgt.defined())
				std::cout << "TGT: " << tgt.sizes() << std::endl;
			if (memory.defined())
				std::cout << "memory: " << memory.sizes() << std::endl;
			if (tgt_mask.defined())
				std::cout << "tgt_mask: " << tgt_mask.sizes() << std::endl;
			if (memory_mask.defined())
				std::cout << "memory_mask: " << memory_mask.sizes() << std::endl;
			if (tgt_key_padding_mask.defined())
				std::cout << "tgt_key_padding_mask: " << tgt_key_padding_mask.sizes() << std::endl;
			if (memory_key_padding_mask.defined())
				std::cout << "memory_key_padding_mask: " << memory_key_padding_mask.sizes() << std::endl;
			if (pos.defined())
				std::cout << "pos: " << pos.sizes() << std::endl;
			if (query_pos.defined())
				std::cout << "query_pos: " << query_pos.sizes() << std::endl;
			if (query_sine_embed.defined())
				std::cout << "query_sine_embed: " << query_sine_embed.sizes() << std::endl;*/


			auto q_content = sa_qcontent_proj->forward(tgt);
			auto q_pos = sa_qpos_proj->forward(query_pos);
			auto k_content = sa_kcontent_proj->forward(tgt);
			auto k_pos = sa_kpos_proj->forward(query_pos);
			auto v = sa_v_proj->forward(tgt);
			auto num_queries = q_content.size(0);
			auto bs = q_content.size(1);
			auto n_model = q_content.size(2);
			auto hw = k_content.size(0);
			auto q = q_content + q_pos;
			auto k = k_content + k_pos;
			auto tgt2 = std::get<0>(self_attn->forward(q, k, v, tgt_key_padding_mask, true, tgt_mask));

			tgt = tgt + dropout2->forward(tgt2);
			tgt = norm1->forward(tgt);

			q_content = ca_qcontent_proj->forward(tgt);
			k_content = ca_kcontent_proj->forward(memory);
			v = ca_v_proj->forward(memory);

			num_queries = q_content.size(0);
			bs = q_content.size(1);
			n_model = q_content.size(2);
			hw = k_content.size(0);

			k_pos = ca_kpos_proj->forward(pos);
			if(is_first)
			{
				q_pos = ca_qpos_proj->forward(query_pos);
				q = q_content + q_pos;
				k = k_content + k_pos;
			}
			else
			{
				q = q_content;
				k = k_content;
			}
			q = q.view({ num_queries, bs, this->nhead_, static_cast<int64_t>(n_model / nhead_) });
			query_sine_embed = ca_qpos_sine_proj->forward(query_sine_embed);
			query_sine_embed = query_sine_embed.view({ num_queries, bs, nhead_, static_cast<int64_t>(n_model / nhead_) });
			q = torch::cat({ q, query_sine_embed }, 3).view({ num_queries, bs, n_model * 2 });
			k = k.view({ hw, bs, nhead_, static_cast<int64_t>(n_model / nhead_) });
			k_pos = k_pos.view({ hw, bs, nhead_, static_cast<int64_t>(n_model / nhead_) });
			k = torch::cat({ k, k_pos }, 3).view({ hw, bs, n_model * 2 });

			
			q = q.to(torch::kCUDA);
			k = k.to(torch::kCUDA);
			v = v.to(torch::kCUDA);
			memory_key_padding_mask = memory_key_padding_mask.to(torch::kCUDA);
			/*std::cout << "QUERY SHAPE: " << q.get_device() << std::endl;
			std::cout << "K SHAPE: " << k.get_device() << std::endl;
			std::cout << "V SHAPE: " << v.get_device() << std::endl;
			std::cout << "MemoryKey SHAPE: " << memory_key_padding_mask.sizes() << std::endl;*/
			//std::cout << "MemoryMask SHAPE: " << memory_mask.get_device() << std::endl;

			tgt2 = std::get<0>(cross_attn->forward(q, k, v, memory_key_padding_mask, true, memory_mask));

			tgt = tgt + dropout3->forward(tgt2);
			tgt = norm2->forward(tgt);
			tgt2 = linear2->forward(dropout1->forward(activation->forward(linear1->forward(tgt))));
			tgt = tgt + dropout4->forward(tgt2);
			return norm3->forward(tgt);
		}

	};
	TORCH_MODULE(TransformerDecoderLayer);

	struct TransformerEncoderImpl : public torch::nn::Module
	{
		torch::nn::LayerNorm norm_{ nullptr };
		int num_layer_;
		//TransformerEncoderLayerImpl encoder;
		//torch::nn::Sequential layers;
		std::vector<TransformerEncoderLayer> layers;
		TransformerEncoderImpl(TransformerEncoderLayer encoder_layer, int num_layer, torch::nn::LayerNorm norm = nullptr)
		{
			if (!norm.is_empty())
				norm_ = norm;
			this->num_layer_ = num_layer;
			//encoder = encoder_layer;
			for (int i = 0; i < num_layer_; i++) {
				layers.push_back(encoder_layer);
				register_module("layers"+std::to_string(i), encoder_layer);
				//layers->push_back(encoder_layer); //Será con TransformerEncoderLayerImpl???
				//register_module("encoder"+std::to_string(i), encoder);
			}
			
			if(!norm.is_empty())
				register_module("norm", norm_);
			
			//this->to(torch::kCUDA);
		}
		torch::Tensor forward(torch::Tensor src, torch::Tensor mask = {}, torch::Tensor src_key_padding_mask = {}, torch::Tensor pos = {})
		{
			auto output = src;
			for(int i=0;i<layers.size();i++)
				output = layers[i]->forward(output, mask, src_key_padding_mask, pos);
			//output = layers->forward(output, mask, src_key_padding_mask, pos);
			/*for (int i = 0; i < num_layer_; i++)
				output = encoder->forward(output, mask, src_key_padding_mask, pos);*/
			if (!norm_.is_empty())
				output = norm_->forward(output);
			return output;
		}
	};
	TORCH_MODULE(TransformerEncoder);

	struct TransformerDecoderImpl : public torch::nn::Module
	{
		int num_layers_;
		torch::nn::LayerNorm norm_{nullptr};
		bool return_intermediate_;
		MLP query_scale{ nullptr }, ref_point_head{ nullptr };
		TransformerDecoderLayer decoder{ nullptr };
		//torch::nn::Sequential layers;
		std::vector<TransformerDecoderLayer> layers;
		TransformerDecoderImpl(TransformerDecoderLayer decoder_layer, int num_layers, torch::nn::LayerNorm norm= nullptr, bool return_intermediate = true, int d_model =256)
		{
			this->num_layers_ = num_layers;
			if (!norm.is_empty())
				norm_ = norm;
			
			return_intermediate_ = return_intermediate;
			query_scale = MLP(d_model, d_model, d_model, 2);
			ref_point_head = MLP(d_model, d_model, 2, 2);

			register_module("query_scale", query_scale);
			register_module("ref_point_head", ref_point_head);
			if (!norm.is_empty()) {
				register_module("norm", norm_);
				norm_->to(torch::kCUDA);
			}
			for (int i = 0; i < num_layers_; i++) {
				layers.push_back(decoder_layer);
				register_module("layers" + std::to_string(i), decoder_layer);
			}
			//decoder_layer->unregister_module("ca_qpos_proj");
			//decoder = decoder_layer;
			//for(int i=0;i<num_layers_;i++)
				//layers->push_back(decoder);
			//register_module("layers", layers);
			//decoder_layer->ca_qpos_proj.
		}
		torch::Tensor gen_sinembed_for_position(torch::Tensor pos_tensor)
		{
			double scale = 2 * M_PI;
			auto dim_t = torch::arange(128, torch::TensorOptions(torch::kFloat32).device(pos_tensor.device()));
			dim_t = torch::pow(10000, (2 * (dim_t.div(2).to(torch::kInt32)) / 128));
			auto x_embed = pos_tensor.index({ torch::indexing::Slice(),  torch::indexing::Slice(), 0}) * scale; 
			auto y_embed = pos_tensor.index({ torch::indexing::Slice(),  torch::indexing::Slice(), 1 }) * scale;
			/*auto x_embed = pos_tensor.select(2, 0) * scale; //https://pytorch.org/docs/stable/generated/torch.select.html equivalent to tensor[:,:,index]
			auto y_embed = pos_tensor.select(2, 1) * scale; //https://pytorch.org/docs/stable/generated/torch.select.html equivalent to tensor[:,:,index]*/

			auto pos_x = x_embed.index({torch::indexing::Slice(), torch::indexing::Slice() , torch::indexing::None }) / dim_t; //How is x_embed[:,:, None] ??? should i use Reshape?
			auto pos_y = y_embed.index({ torch::indexing::Slice(), torch::indexing::Slice() , torch::indexing::None }) / dim_t;
			pos_x = torch::stack({ pos_x.index({torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2) }).sin(), pos_x.index({torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2) }).cos()}, 3).flatten(2); //https://github.com/Atten4Vis/ConditionalDETR/blob/main/models/transformer.py 43
			pos_y = torch::stack({ pos_y.index({torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2) }).sin(), pos_y.index({torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2) }).cos() }, 3).flatten(2);
			return torch::cat({ pos_x, pos_y }, 2);
		}
		std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor tgt, torch::Tensor memory, torch::Tensor tgt_mask = {}, torch::Tensor memory_mask = {}, torch::Tensor tgt_key_padding_mask = {}, torch::Tensor memory_key_padding_mask = {}, torch::Tensor pos = {}, torch::Tensor query_pos = {})
		{
			auto output = tgt;
			std::vector<torch::Tensor> intermediates;
			auto reference_points= ref_point_head->forward(query_pos).sigmoid().transpose(0,1); // [num_queries, batch_size, 2]
			for(int i=0;i<this->num_layers_;i++)
			{
				//torch::Tensor obj_center; //WARNING: IMPLEMENT OBJ_CENTER reference_points[..., :2].transpose(0, 1)
				auto obj_center = reference_points.index({ "...", torch::indexing::Slice(torch::indexing::None, 2) }).transpose(0, 1); //https://pytorch.org/cppdocs/notes/tensor_indexing.html
				
				auto query_sine_embed = gen_sinembed_for_position(obj_center);
				if (i == 0) {
					query_sine_embed *= 1;
				}else
				{
					query_sine_embed = query_sine_embed * query_scale->forward(output);
				}
				
				output = this->layers[i]->forward(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, i == 0);
				if (return_intermediate_)
					intermediates.push_back(norm_->forward(output));
			}
			if(!norm_.is_empty())
			{
				output = norm_->forward(output);
				if(return_intermediate_)
				{
					intermediates.pop_back();
					intermediates.push_back(output);
				}
			}
			if (return_intermediate_)
				return std::make_tuple(torch::stack(intermediates).transpose(1, 2), reference_points);
				//return torch::stack({ torch::stack(intermediates).transpose(1,2) , reference_points });
			//return output.unsqueeze(0);
			return std::make_tuple(output.unsqueeze(0), torch::Tensor{});
			//Tiene que tener como salida una TUPLA
		}
	};
	TORCH_MODULE(TransformerDecoder);

	class TransformerImpl : public torch::nn::Module
	{
	private:
		void reset_parameters()
		{
			for (auto p : this->parameters())
				if (p.dim() > 1)
					torch::nn::init::xavier_uniform_(p);
		}
	public:
		TransformerEncoder encoder{ nullptr };
		TransformerDecoder decoder{ nullptr };
		int d_model_, nhead_, dec_layers_;
		TransformerImpl(int d_model= 512, int nhead = 8, int num_queries = 300, int num_encoder_layers = 6, int num_decoder_layers = 6, int dim_feedforward = 2048, double dropout = 0.1, std::string activation = "relu", bool normalize_before = false, bool return_intermediate_dec = true)
		{
			//auto transform_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before);
			//transform_encoder->to(torch::kCUDA);
			encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before), num_encoder_layers, normalize_before ? torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })) : torch::nn::LayerNorm(nullptr));
			decoder = TransformerDecoder(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, normalize_before), num_decoder_layers, torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })), return_intermediate_dec, d_model);
			register_module("TransformEncoderTransformer", encoder);
			register_module("TransformDecoderTransformer", decoder);
			reset_parameters();
			d_model_ = d_model;
			nhead_ = nhead; 
			dec_layers_ = num_decoder_layers;
			encoder->to(torch::kCUDA);
			decoder->to(torch::kCUDA);
		}
		std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor src, torch::Tensor mask, torch::Tensor query_embed, torch::Tensor pos_embed)
		{
			//flatten NxCxHxW to HWxNxC
			const auto siz = src.sizes();
			auto bs = siz[0];
			auto c = siz[1];
			auto h = siz[2];
			auto w = siz[3];
			src = src.flatten(2).permute({ 2,0,1 });
			pos_embed = pos_embed.flatten(2).permute({ 2,0,1 });
			query_embed = query_embed.unsqueeze(1).repeat({ 1,bs,1 });
			mask = mask.flatten(1);
			auto tgt = torch::zeros_like(query_embed).to(src.device());
			auto memory = encoder->forward(src, {}, mask, pos_embed).to(src.device());
			return decoder->forward(tgt, memory, {}, {}, {}, mask, pos_embed, query_embed);
		}
	};
	TORCH_MODULE(Transformer);
}
#endif