#pragma once

#ifndef LIBS_HUNGARIAN_MATCHER
#define LIBS_HUNGARIAN_MATCHER

#include <torch/torch.h>
#include "hungarian_algorithm.h"
#include "boxes.h"
#include "lsap.h"

struct HungarianMatcherImpl : public torch::nn::Module
{
	float cost_class_, cost_bbox_, cost_giou_;
	HungarianAlgorithm hungarian_ = HungarianAlgorithm();
	HungarianMatcherImpl(float cost_class = 1, float cost_bbox = 1, float cost_giou = 1)
	{
		this->cost_class_ = cost_class;
		this->cost_bbox_ = cost_bbox;
		this->cost_giou_ = cost_giou;
	}
	//TODO: Fix, (matcher.py 49) the outputs is a DICT of tensors:
	//output["pred_logits"]: Tensor of dim[batch_size, num_queries, num_classes] with the classification logits
	//output["pred_boxes"] : Tensor of dim[batch_size, num_queries, 4] with the predicted box coordinates
	//Todo: Fix, (matcher.py 53) the targets is a DICT of tensors:
	//"labels": Tensor of dim[num_target_boxes](where num_target_boxes is the number of ground - truth objects in the target) containing the class labels
	//"boxes" : Tensor of dim[num_target_boxes, 4] containing the target box coordinates
	/*
	 *Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
	 **/
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> forward(torch::OrderedDict<std::string, torch::Tensor> outputs, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets)
	{
		//TODO: Test this torch::OrderedDict
		/*torch::OrderedDict<std::string, torch::Tensor> dict;
		dict["lero"]*/
		torch::NoGradGuard no_grad;
		auto bs = outputs["pred_logits"].size(0);
		auto num_queries = outputs["pred_logits"].size(1);
		auto out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid();
		auto out_bbox = outputs["pred_boxes"].flatten(0, 1);

		std::vector<torch::Tensor> tgt_ids_vec;
		std::vector<torch::Tensor> tgt_bbox_vec;
		for(int i=0;i<targets.size();i++)
		{
			tgt_ids_vec.push_back(targets[i]["labels"]);
			tgt_ids_vec.push_back(targets[i]["boxes"]);
		}
		auto tgt_ids = torch::cat(tgt_ids_vec);
		auto tgt_bbox = torch::cat(tgt_bbox_vec);
		float alpha = 0.25;
		float gamma = 2.0;
		
		auto neg_cost_class = (1 - alpha) * (out_prob.pow(gamma)) * (-(1 - out_prob + 1e-8).log());
		auto pos_cost_class = alpha * ((1 - out_prob).pow(gamma)) * (-(out_prob + 1e-8).log());
		auto cost_class = torch::zeros({ pos_cost_class.size(0), tgt_ids.size(0) }, torch::kFloat32);

		for(int i=0;i<pos_cost_class.size(0);i++)
		for(int j=0;j<tgt_ids.size(0);j++)
			cost_class[i][j] = pos_cost_class[i][j] - neg_cost_class[i][j];

		auto cost_bbox = torch::cdist(out_bbox, tgt_bbox, 1);
		auto cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox));
		auto c = this->cost_bbox_ * cost_bbox + this->cost_class_ * cost_class + this->cost_giou_ * cost_giou;
		c = c.view({ bs, num_queries, -1 }).cpu();

		std::vector<int64_t> sizes;
		for (int i = 0; i < static_cast<int64_t>(targets.size()); i++)
			sizes.push_back(targets[i]["boxes"].size(0));

		auto si = torch::from_blob(sizes.data(), { static_cast<long long>(sizes.size()) }, torch::kFloat32).sizes();
		auto vec_tensor = c.split(si, -1);
		/*std::vector<int64_t> col;
		std::vector<int64_t> row;
		std::vector<std::vector<std::tuple<int64_t, int64_t>>> v;*/
		std::vector<std::tuple<torch::Tensor, torch::Tensor>> ij;
		for(int i =0;i<static_cast<int64_t>(vec_tensor.size());i++)
		{
			std::tuple<torch::Tensor, torch::Tensor> v_tuple;
			if (linear_sum_assignment::solve(vec_tensor[i], false, v_tuple) == 0)
				ij.push_back(v_tuple);
		}

		std::vector<std::vector<double>> hung;
		std::vector<int> assignement;
		for(int i = 0;i<c.size(0);i++)
			hung.push_back({ c[i].data_ptr<double>(), c[i].data_ptr<double>() + c[i].numel() });

		hungarian_.Solve(hung, assignement);

		auto indices = torch::from_blob(assignement.data(), { static_cast<int64_t>(assignement.size()) }, torch::kFloat32);
		auto resh =indices.reshape({static_cast<int64_t>(hung.size()), -1 });
		//return resh;
	}
};

TORCH_MODULE(HungarianMatcher);

#endif