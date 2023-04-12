#pragma once

#ifndef MODELS_CONDITIONAL_DETR_H
#define MODELS_CONDITIONAL_DETR_H

#include <torch/torch.h>

#include "backbone.h"
#include "hungarian_matcher.h"
#include "layers.h"
#include "position_encoding.h"

class ConditionalDETR : public torch::nn::Module
{
private:
	conditional_detr::Transformer transformer_;
	Joiner joiner_ = Joiner();
	//Backbone backbone_;
	int num_queries_, hidden_dim_;
	torch::nn::Linear class_embed{ nullptr };
	MLP bbox_embed{ nullptr };
	torch::nn::Embedding query_embed{ nullptr };
	torch::nn::Conv2d input_proj{ nullptr };
	//PositionEncoding encoding_ = static_cast<PositionEncoding>(PositionEmbeddingLearned());
public:
	ConditionalDETR()
	{
		
	}
	ConditionalDETR(Backbone backbone, conditional_detr::Transformer transformer, int num_classes, int num_queries, bool aux_loss = false)
	{
		//backbone_ = backbone;
		transformer_ = transformer;
		hidden_dim_ = transformer_->d_model_;
		//encoding_ = static_cast<PositionEncoding>(PositionEmbeddingSine(static_cast<int>(hidden_dim_ / 2)));
		num_queries_ = num_queries;
		joiner_ = Joiner(backbone, static_cast<PositionEncoding>(PositionEmbeddingLearned(static_cast<int>(hidden_dim_ / 2))));
		class_embed = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim_, num_classes));
		bbox_embed = MLP(hidden_dim_, hidden_dim_, 4, 3);
		query_embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_queries, hidden_dim_));
		input_proj = torch::nn::Conv2d(torch::nn::Conv2dOptions(backbone.num_channels[0], hidden_dim_, {1,1}));


		double prior_prob = 0.01;
		auto bias_value = -std::log((1 - prior_prob) / prior_prob);
		class_embed->bias.data() = torch::ones({ num_classes }) * bias_value;
		
		if (bbox_embed->layers[bbox_embed->layers->size() - 1]->as<torch::nn::LinearImpl>()) {
			auto l = bbox_embed->layers[bbox_embed->layers->size() - 1]->as<torch::nn::LinearImpl>();
			torch::nn::init::constant_(l->weight.data(), 0);
			torch::nn::init::constant_(l->bias.data(), 0);
		}
		register_module("class_embed", class_embed);
		register_module("bbox_embed", bbox_embed);
		register_module("query_embed", query_embed);
		register_module("input_proj", input_proj);

		//joiner_(torch::kCUDA);
		transformer_->to(torch::kCUDA);
		//encoding_.to(torch::kCUDA);
	}
	torch::OrderedDict<std::string, torch::Tensor> forward(NestedTensor samples)
	{
		std::cout << "sample tensor size (Supossed to be [bs, 3, H, W]): " << samples.tensors_.sizes() << std::endl;
		auto features_pos = joiner_.forward(samples);
		auto features = std::get<0>(features_pos);
		auto pos = std::get<1>(features_pos);
		/*auto feat_and_mask = backbone_.forward(samples);
		auto pos = encoding_.forward(feat_and_mask);*/
		
		if(!features.masks_.defined())
		{
			//should throw exception???
		}
		auto src = features.tensors_;
		
		auto mask = features.masks_;
		auto tuple_transformer = transformer_->forward(input_proj->forward(src), mask, query_embed->weight, pos);
		auto hs = std::get<0>(tuple_transformer);
		auto reference = std::get<1>(tuple_transformer);
		auto reference_before_sigmoid = inverse_sigmoid(reference);
		std::vector<torch::Tensor> output_coords_vec;
		for(int64_t i=0;i<static_cast<int64_t>(hs.size(0));i++)
		{
			auto tmp = bbox_embed->forward(hs[i]);
			tmp.index({ "...", torch::indexing::Slice(torch::indexing::None, 2) }) += reference_before_sigmoid;
			tmp = tmp.sigmoid();
			output_coords_vec.push_back(tmp);
		}
		auto outputs_coords = torch::stack(output_coords_vec);
		auto output_class = class_embed->forward(hs);
		torch::OrderedDict<std::string, torch::Tensor> outputs;
		outputs.insert("pred_logits", output_class[-1]);
		outputs.insert("pred_boxes", outputs_coords[-1]);
		return outputs;
	}
};

class SetCriterion : public torch::nn::Module
{
private:
	int num_class_;
	HungarianMatcher matcher_ = HungarianMatcher();
	
	//std::vector<double> losses_;
	float focal_alpha_;

	/**
	 * \brief 
	 * \param indices 
	 * \param src set false if you want use mask panoptic
	 * \return 
	 */
	std::tuple<torch::Tensor, torch::Tensor> get_permutation_idx(std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices, bool src = true)
	{
		
		std::vector<torch::Tensor> batch_idx;
		std::vector<torch::Tensor> src_idx;
		for (int i = 0; i < indices.size(); i++)
		{
			batch_idx.push_back(src ? torch::full_like(std::get<0>(indices[i]), i) : torch::full_like(std::get<1>(indices[i]), i));
			src_idx.push_back(src ? std::get<0>(indices[i]) : std::get<1>(indices[i]));
		}
		std::make_tuple(torch::cat(batch_idx), torch::cat(src_idx));
	}
	/*std::tuple<torch::Tensor, torch::Tensor> get_src_permutation_idx(std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices)
	{
		std::vector<torch::Tensor> batch_idx;
		std::vector<torch::Tensor> src_idx;
		for (int i = 0; i < indices.size(); i++)
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
	}*/
	std::unordered_map<std::string, torch::Tensor> losses_;
public:
	std::unordered_map<std::string, float> weight_dict_;
	SetCriterion(){}
	SetCriterion(int num_class, float focal_alpha = 0.25)
	{
		this->num_class_ = num_class;
		//std::unordered_map<std::string, float> weight_dict
		//this->weight_dict_ = weight_dict;
		this->focal_alpha_ = focal_alpha;
		weight_dict_["loss_ce"] = 2;
		weight_dict_["loss_bbox"] = 5;
		weight_dict_["loss_giou"] = 2;
		weight_dict_["loss_dice"] = 1;
	}
	
	void loss_labels(torch::OrderedDict<std::string, torch::Tensor> output, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets, std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices, torch::Scalar num_boxes)
	{
		torch::OrderedDict<std::string, torch::Tensor> result_dicts;
		auto idx = get_permutation_idx(indices);

		auto src_logits = output["pred_logits"];

		std::vector<torch::Tensor> tgt_classes_o_vec;
		for (int i = 0; i < indices.size(); i++)
			tgt_classes_o_vec.push_back(targets[i]["labels"].index({std::get<1>(indices[i])}));
		auto target_classes_o = torch::cat(tgt_classes_o_vec);
		auto target_classes = torch::full(src_logits.index({ torch::indexing::Slice(torch::indexing::None, 2) }).sizes(), num_class_, torch::TensorOptions(torch::kInt64).device(src_logits.device()));
		target_classes = target_classes.index_put({ std::get<0>(idx), std::get<1>(idx) }, target_classes_o);

		auto target_classes_onehot = torch::zeros({ src_logits.size(0), src_logits.size(1), src_logits.size(2) + 1 }, torch::TensorOptions(src_logits.dtype()).layout(src_logits.layout()).device(src_logits.device()));
		target_classes_onehot = target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1);
		target_classes_onehot = target_classes_onehot.index({ sli(),sli(),sli(non, -1) });
		auto loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, focal_alpha_, 2) * src_logits.size(1);
		this->losses_["loss_ce"] = loss_ce;
	}
	torch::OrderedDict<std::string, torch::Tensor> loss_cardinality(torch::OrderedDict<std::string, torch::Tensor> output, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets, std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices, torch::Scalar num_boxes)
	{
		torch::OrderedDict<std::string, torch::Tensor> losses;
		return losses;
	}
	void loss_boxes(torch::OrderedDict<std::string, torch::Tensor> output, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets, std::vector<std::tuple<torch::Tensor, torch::Tensor>> indices, torch::Scalar num_boxes)
	{
		torch::OrderedDict<std::string, torch::Tensor> losses;
		auto idx = get_permutation_idx(indices);
		auto src_boxes = output["pred_boxes"].index({ std::get<0>(idx), std::get<1>(idx) });
		std::vector<torch::Tensor> tgt_boxes_vec;
		for (int i = 0; i < indices.size(); i++)
			tgt_boxes_vec.push_back(targets[i]["boxes"].index({ std::get<1>(indices[i]) }));
		auto target_boxes = torch::cat(tgt_boxes_vec);
		auto loss_bbox = torch::nn::functional::l1_loss(src_boxes, target_boxes, torch::nn::functional::L1LossFuncOptions(torch::enumtype::kNone()));
		auto loss_giou = 1 - torch::diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)));
		
		losses_["loss_bbox"] = loss_bbox.sum() / num_boxes;
		losses_["loss_giou"] = loss_giou.sum() / num_boxes;
	}

	std::unordered_map<std::string, torch::Tensor> forward(torch::OrderedDict<std::string, torch::Tensor> outputs, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets)
	{
		//if(outputs[])
		/*""" This performs the loss computation.
			Parameters:
				outputs: dict of tensors, see the output specification of the model for the format
				targets : list of dicts, such that len(targets) == batch_size.
		The expected keys in each dict depends on the losses applied, see each loss' doc*/
		std::vector<torch::Tensor> auxs;
		torch::OrderedDict<std::string, torch::Tensor> outputs_without_aux; //because aux is segmentation
		for (auto v : outputs)
			if (v.key() != "aux_outputs" || v.key() != "enc_encoder")
				outputs_without_aux.insert(v.key(), v.value());

		auto indices = matcher_->forward(outputs_without_aux, targets);
		int64_t num_boxes = 0;
		for(int64_t i=0;i<static_cast<int64_t>(targets.size());i++)
			num_boxes += targets[i]["labels"].size(0);
		auto num_box = torch::from_blob(std::vector<int64_t>({ num_boxes }).data(), { 1 }, torch::kFloat);
		num_boxes = torch::clamp(num_box, 1).item<int64_t>();

		loss_boxes(outputs, targets, indices, num_boxes);
		loss_labels(outputs, targets, indices, num_boxes);
		return this->losses_;
	}
};

class PostProcess
{
public:
	std::vector<std::unordered_map<std::string, torch::Tensor>> forward(torch::OrderedDict<std::string, torch::Tensor> outputs, torch::Tensor target_sizes)
	{
		auto out_logits = outputs["pred_logits"];
		auto out_bbox = outputs["pred_boxes"];
		if(out_logits.size(0) == target_sizes.size(0))
		{
			std::cout << "should throw exception???" << std::endl;
			//should throw exception???
		}
		if(target_sizes.size(1) == 2)
		{
			std::cout << "should throw exception???" << std::endl;
			//should throw exception???
		}
		auto prob = out_logits.sigmoid();
		auto topk = torch::topk(prob.view({ out_logits.size(0), -1 }), 100, 1);
		auto topk_values = std::get<0>(topk);
		auto topk_indexes = std::get<1>(topk);
		auto scores = topk_values;
		auto topk_boxes = topk_indexes / out_logits.size(2);
		auto labels = topk_indexes % out_logits.size(2);
		auto boxes = box_cxcywh_to_xyxy(out_bbox);
		boxes = torch::gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat({ 1,1,4 }));

		//auto img_h = target_sizes.unbind(1); conditional_detr.py 312
		auto img_ = target_sizes.unbind(1);
		auto img_h = img_[0];
		auto img_w = img_[1];
		auto scale_fct = torch::stack({ img_w, img_h, img_w, img_h }, 1);
		boxes = boxes * scale_fct.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice() });

		std::vector<std::unordered_map<std::string, torch::Tensor>> results;
		if (!(labels.size(0) == boxes.size(0) == scores.size(0)))
			std::cerr << "Some sizes is not equals: " << "Labels: " << labels.size(0) << " Boxes: " << boxes.size(0) << " Scores: " << scores.size(0) << std::endl;
		if(labels.size(0) == boxes.size(0) == scores.size(0))
		{
			for (int i = 0; i < labels.size(0); i++)
			{
				std::unordered_map<std::string, torch::Tensor> ord;
				ord["scores"] = scores[i];
				ord["labels"] = labels[i];
				ord["boxes"] = boxes[i];
				results.push_back(ord);
			}
		}
		return results;
	}
};
#endif