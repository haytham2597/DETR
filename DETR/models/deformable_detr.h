#pragma once

#ifndef MODELS_DEFORMABLE_DETR_H
#define MODELS_DEFORMABLE_DETR_H

#include <torch/torch.h>
#include "backbone.h"

#include "deformable_transformer.h"
#include "../libs/lsa/hungarian_matcher.h"
#include "position_encoding.h"

class DeformableDetr
{
private:
	Backbone backbone_;
	DeformableTransformer transformer_;
	PositionEncoding position_encoding_;
	int num_classes_;
	int num_queries_;
	int num_features_level_;
	bool aux_loss_;
	bool with_box_refine_;
public:
	/**
	 * \brief 
	 * \param backbone Backboen example resnet50, etc
	 * \param transformer Deformable Transformer
	 * \param num_classes Num of class that have your datasets
	 * \param num_queries Number of query slots default = 300
	 * \param num_features_levels number of feature levels, default = 4 
	 * \param aux_loss Disables auxiliary decoding losses (loss at each layer)
	 * \param with_box_refine 
	 * \param two_stage 
	 */
	DeformableDetr(Backbone backbone, DeformableTransformer transformer, PositionEncoding pos_encoding, int num_classes, int num_queries, int num_features_levels, bool aux_loss = true, bool with_box_refine = false, bool two_stage= false)
	{
		
		backbone_ = backbone;
		transformer_ = transformer;
		position_encoding_ = pos_encoding;

		this->num_classes_ = num_classes+1; //Add new class for Empty object as id
		this->num_queries_ = num_queries;
		this->num_features_level_ = num_features_levels;
		this->aux_loss_ = aux_loss;
		this->with_box_refine_ = with_box_refine;
		
		//int hidden_dim = transformer
	}
	/*backbone, transformer, num_classes, num_queries, num_feature_levels,
		aux_loss = True, with_box_refine = False, two_stage = False*/
};

class SetCriterion : public torch::nn::Module
{
private:
	int num_class_;
	HungarianMatcher matcher_;
	torch::OrderedDict<std::string, float> weight_dict_;
	std::vector<double> losses_;
	float focal_alpha_;
	void get_src_permutation_idx(int indices)
	{
		//torch::full_like()
	}
public:
	SetCriterion(int num_class, HungarianMatcher matcher, torch::OrderedDict<std::string, float> weight_dict, std::vector<double> losses, float focal_alpha = 0.25)
	{
		this->num_class_ = num_class;
		this->matcher_ = matcher;
		this->weight_dict_ = weight_dict;
		this->losses_ = losses;
		this->focal_alpha_ = focal_alpha;
	}
	std::tuple<torch::Tensor, torch::Tensor> get_src_permutation_idx(std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices)
	{
		std::vector<torch::Tensor> batch_idx;
		std::vector<torch::Tensor> src_idx;
		for(int i=0;i<indices.size();i++)
		{
			batch_idx.push_back(torch::full_like(std::get<0>(indices[i]), i));
			src_idx.push_back(std::get<0>(indices[i]));
		}
		std::make_tuple(torch::cat(batch_idx), torch::cat(src_idx));
	}
	std::tuple<torch::Tensor, torch::Tensor> get_tgt_permutation_idx(std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices)
	{
		std::vector<torch::Tensor> batch_idx;
		std::vector<torch::Tensor> src_idx;
		for (int i = 0; i < indices.size(); i++)
		{
			batch_idx.push_back(torch::full_like(std::get<1>(indices[i]), i));
			src_idx.push_back(std::get<1>(indices[i]));
		}
		std::make_tuple(torch::cat(batch_idx), torch::cat(src_idx));
	}

	torch::OrderedDict<std::string, float> loss_labels(torch::OrderedDict<std::string, torch::Tensor> output, torch::OrderedDict<std::string, torch::Tensor> targets, std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices, torch::Scalar num_boxes)
	{
		torch::OrderedDict<std::string, float> result_dicts;
		auto idx = get_src_permutation_idx(indices);
		
		auto src_logits = output["pred_logits"];

		std::vector<torch::Tensor> tgt_classes_o_vec;
		for(int i=0;i<indices.size();i++)
		{
			tgt_classes_o_vec.push_back(targets["labels"][i]);
		}
		auto target_classes_o = torch::cat(tgt_classes_o_vec);
		auto target_classes = torch::full(src_logits.index({ torch::indexing::Slice(torch::indexing::None, 2) }).sizes(), num_class_, torch::TensorOptions(torch::kInt64).device(src_logits.device()));
		target_classes = target_classes.index_put({ std::get<0>(idx), std::get<1>(idx)}, target_classes_o);

		auto target_classes_onehot = torch::zeros({ src_logits.size(0), src_logits.size(1), src_logits.size(2) + 1 }, torch::TensorOptions(src_logits.dtype()).layout(src_logits.layout()).device(src_logits.device()));
		target_classes_onehot = target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1);
		target_classes_onehot = target_classes_onehot.index({ sli(),sli(),sli(non, -1) });
		auto loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, focal_alpha_, 2) * src_logits.size(1);
		result_dicts.insert("loss_ce", loss_ce);
		//result_dicts.insert("loss_ce", )
		return result_dicts;

	}
	void loss_boxes(torch::Tensor outputs)
	{
		
	}

	void forward(torch::OrderedDict<std::string, torch::Tensor> outputs, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets)
	{
		/*""" This performs the loss computation.
			Parameters:
				outputs: dict of tensors, see the output specification of the model for the format
				targets : list of dicts, such that len(targets) == batch_size.
		The expected keys in each dict depends on the losses applied, see each loss' doc*/
		std::vector<torch::Tensor> auxs;
		for(auto v : outputs)
		{
			if(v.key() != "aux_outputs" || v.key() != "enc_encoder")
			{
				auxs.push_back(v.value());
			}
		}
		torch::Tensor outputs_without_aux = torch::cat(auxs);
		auto indices = matcher_->forward(outputs, targets);

	}
};
#endif