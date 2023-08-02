#pragma once

#ifndef LIBS_HUNGARIAN_MATCHER
#define LIBS_HUNGARIAN_MATCHER

#include <torch/torch.h>
#include "hungarian_optimize_lsap.h"
//#include "hungarian_algorithm.h"
#include "../util/boxes.h"
#include "lsap.h"

using namespace torch::indexing;

struct HungarianMatcherImpl : public torch::nn::Module
{
	double cost_class_, cost_bbox_, cost_giou_;
	//LinearSumAssignment linearSum = LinearSumAssignment();
	//HungarianAlgorithm hungarian_ = HungarianAlgorithm();
	HungarianMatcherImpl(double cost_class = 1, double cost_bbox = 1, double cost_giou = 1)
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
	std::vector<std::pair<torch::Tensor, torch::Tensor>> forward(torch::OrderedDict<std::string, torch::Tensor> outputs, std::vector<torch::OrderedDict<std::string, torch::Tensor>> targets)
	{
		torch::NoGradGuard no_grad;
		//TODO: Test this torch::OrderedDict

		//TORCH_CHECK(outputs["pred_logits"].sizes() == torch::IntArrayRef({2,300,37}), "Expected size: ", at::IntArrayRef({ 2,300,37 }), " but got: ", outputs["pred_logits"].sizes())
		//TORCH_CHECK(outputs["pred_boxes"].sizes() == torch::IntArrayRef({ 2,300,4 }), "Expected size: ", at::IntArrayRef({ 2,300,4 }), " but got: ", outputs["pred_boxes"].sizes())

		int64_t bs = outputs["pred_logits"].size(0);
		int64_t num_queries = outputs["pred_logits"].size(1);

		torch::Tensor out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid();
		torch::Tensor out_bbox = outputs["pred_boxes"].flatten(0, 1);

		//TORCH_CHECK(out_prob.sizes() == torch::IntArrayRef({ static_cast<int64_t>(2 * 300),37 }), "Expected size: ", at::IntArrayRef({ 2*300,37 }), " but got: ", out_prob.sizes())
		//TORCH_CHECK(out_bbox.sizes() == torch::IntArrayRef({ static_cast<int64_t>(2 * 300),4 }), "Expected size: ", at::IntArrayRef({ 2 * 300,4 }), " but got: ", out_bbox.sizes())

		std::vector<torch::Tensor> tgt_ids_vec;
		std::vector<torch::Tensor> tgt_bbox_vec;

		//WARNING: Maybe the problem contigous memory is here??
		for (uint64_t i = 0; i < targets.size(); i++)
		{
			/*std::cout << "Label [" << std::to_string(i) << "]" << targets[i]["labels"].sizes() << " File: " << __FILE__ << "Line: " << __LINE__ << std::endl;
			std::cout << "Boxes [" << std::to_string(i) << "]" << targets[i]["boxes"].sizes() << " File: " << __FILE__ << "Line: " << __LINE__ << std::endl;*/
			tgt_ids_vec.push_back(targets[i]["labels"]);
			tgt_bbox_vec.push_back(targets[i]["boxes"]);
		}
		
		torch::Tensor tgt_ids = torch::cat(tgt_ids_vec);
		torch::Tensor tgt_bbox = torch::cat(tgt_bbox_vec);
		//MESSAGE_LOG_ObJ("TGTIDS:", tgt_ids)
		
		float alpha = 0.25;
		float gamma = 2.0;

		torch::Tensor neg_cost_class = (1 - alpha) * (out_prob.pow(gamma)) * (-(1 - out_prob + 1e-8).log());
		torch::Tensor pos_cost_class = alpha * ((1 - out_prob).pow(gamma)) * (-(out_prob + 1e-8).log());

		torch::Tensor cost_class = pos_cost_class.index({ Slice(), tgt_ids }) - neg_cost_class.index({ Slice(), tgt_ids });

		tgt_bbox = tgt_bbox.to(out_bbox.device());
		out_bbox = out_bbox.to(tgt_bbox.dtype());

		torch::Tensor cost_bbox = torch::cdist(out_bbox, tgt_bbox, 1);
		torch::Tensor cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox));
		cost_class = cost_class.to(cost_bbox.device());
		cost_giou = cost_giou.to(cost_bbox.device());
		
		torch::Tensor c = (this->cost_bbox_ * cost_bbox) + (this->cost_class_ * cost_class) + (this->cost_giou_ * cost_giou);
		//std::cout << "PRINT C SIZES: " << c.sizes() << std::endl;
		//std::cout << "C: " << c << std::endl;
		//c = c.view({ bs, num_queries, -1 }).cpu();
		//c = c.view({ bs, num_queries, -1 }).to(torch::kCPU);
		c = c.reshape({ bs, num_queries, -1 }).to(torch::kCPU);

		std::vector<int64_t> sizes;
		for (int i = 0; i < static_cast<int64_t>(targets.size()); i++)
			sizes.push_back(targets[i]["boxes"].size(0));
		//std::cout << "c size: " << c.sizes() << " device : " << c.get_device() << " dtype : " << c.dtype().name() << std::endl;
		//MESSAGE_LOG_OBJ("IntarrayRef data: " , at::IntArrayRef(sizes.data(), sizes.size()))
		std::vector<torch::Tensor> vec_tensor = c.split(at::IntArrayRef(sizes.data(), sizes.size()), -1);
		//auto vec_tensor = c.split({ sizes.data(), sizes.size() }, -1);

		std::vector<std::pair<torch::Tensor, torch::Tensor>> ij;
		for(int64_t i =0;i<static_cast<int64_t>(vec_tensor.size());i++)
		{
			std::pair<torch::Tensor, torch::Tensor> v_tuple;
			int solve = linear_sum_assignment::solve(vec_tensor[i][i], false, v_tuple); //WARNING: Fix this, because if matrix is all same should return indices sort
			//std::cout << "Print tuple indices: " << std::get<1>(v_tuple) << std::endl;

			/*MESSAGE_LOG_OBJ("Index i sizes: ", v_tuple.first)
			MESSAGE_LOG_OBJ("Index j sizes: ", v_tuple.second)*/
			ij.push_back(v_tuple);
			/*if (linear_sum_assignment::solve(vec_tensor[i][i], false, v_tuple) == 0) {
				ij.push_back(v_tuple);
			}*/
		}
		/*std::vector<std::vector<double>> hung;
		std::vector<int> assignement;
		for (int i = 0; i < c.size(0); i++)
			hung.push_back({ c[i].data_ptr<double>(), c[i].data_ptr<double>() + c[i].numel() });

		hungarian_.Solve(hung, assignement);

		auto indices = torch::from_blob(assignement.data(), { static_cast<int64_t>(assignement.size()) }, torch::kInt);
		auto resh = indices.reshape({ static_cast<int64_t>(hung.size()), -1 });*/
		//std::cout << "IJ size: " << ij.size() << std::endl;
		//std::cout << "HEREEE" << std::endl;

		/*
		 *Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
		 **/

		return ij;
		
		//return resh;
	}
};

TORCH_MODULE(HungarianMatcher);

#endif