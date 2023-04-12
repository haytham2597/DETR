#pragma once

#ifndef UTILS_BOXES_H
#define UTILS_BOXES_H

#include <torch/torch.h>

torch::Tensor box_cxcywh_to_xyxy(torch::Tensor x)
{
	auto unb = x.unbind(-1);
	auto x_c = unb[0].item<double>();
	auto y_c = unb[1].item<double>();
	auto w = unb[2].item<double>();
	auto h = unb[3].item<double>();
	std::vector<double> v({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h });
	return torch::from_blob(v.data(), { 4,1 }, torch::kFloat32);
}

torch::Tensor upcast(torch::Tensor x)
{
	if (x.is_floating_point()) {
		if (x.dtype() == torch::kFloat || x.dtype() == torch::kFloat32)
			return x;
		return x.to(torch::kFloat32);
	}
	if (x.dtype() == torch::kInt32 || x.dtype() == torch::kInt64)
		return x;
	return x.to(torch::kInt32);
}
/**
 * \brief 
 * \param boxes Tensor[N, 4] (x1,x2,y1,y2)
 * \return Tensor[N]
 */
torch::Tensor box_area(torch::Tensor boxes)
{
	auto area = torch::zeros({ boxes.size(0) }, torch::kFloat32);
	for(int i=0;i<boxes.size(0);i++)
		area[i] = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
	return area;
}

/**
 * \brief [STABLE]
 * \param boxes1 
 * \param boxes2 
 * \return 
 */
std::tuple<torch::Tensor, torch::Tensor> box_inter_union(torch::Tensor boxes1, torch::Tensor boxes2)
{
	auto area1 = box_area(boxes1);
	auto area2 = box_area(boxes2);

	auto lt = torch::max(boxes1.narrow_copy(1, 0, 2).unsqueeze(1), boxes2.narrow_copy(1, 0, 2));
	auto rb = torch::min(boxes1.narrow_copy(1, 2, 2).unsqueeze(1), boxes2.narrow_copy(1, 2, 2));
	auto wh = upcast(rb - lt).clamp(0); //debería poner _upcast para prevenir overflow test_box 81

	std::vector<torch::Tensor> tensors;
	for(int i=0;i<wh.size(-1);i++)
	{
		std::vector<torch::Tensor> tensor;
		for (int j = 0; j < wh.size(0); j++)
			tensor.push_back(wh[j].narrow_copy(1, i, 1).reshape({ 1,-1 }));
		tensors.push_back(torch::cat(tensor));
	}
	auto inter = tensors[0] * tensors[1];
	auto uni = area1.unsqueeze(1) + area2 - inter;
	return std::make_tuple(inter, uni);
}

/**
 * \brief [STABLE IS WORK]
 * \param boxes1 
 * \param boxes2 
 * \return 
 */
torch::Tensor generalized_box_iou(torch::Tensor boxes1, torch::Tensor boxes2)
{
	auto tup = box_inter_union(boxes1, boxes2);
	auto iou = std::get<0>(tup) / std::get<1>(tup);
	auto lt = torch::min(boxes1.narrow_copy(1, 0, 2).unsqueeze(1), boxes2.narrow_copy(1, 0, 2));
	auto rb = torch::max(boxes1.narrow_copy(1, 2, 2).unsqueeze(1), boxes2.narrow_copy(1, 2, 2));
	auto wh = upcast(rb - lt).clamp(0); //debería poner _upcast para prevenir overflow test_box 81

	std::vector<torch::Tensor> tensors;
	for (int i = 0; i < wh.size(-1); i++)
	{
		std::vector<torch::Tensor> tensor;
		for (int j = 0; j < wh.size(0); j++)
			tensor.push_back(wh[j].narrow_copy(1, i, 1).reshape({ 1,-1 }));
		tensors.push_back(torch::cat(tensor));
	}
	auto areai = tensors[0] * tensors[1];
	return iou - (areai - std::get<1>(tup)) / areai;
}

#endif