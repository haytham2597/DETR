#pragma once

#ifndef UTILS_BOXES_H
#define UTILS_BOXES_H

#include <torch/torch.h>

torch::Tensor box_cxcywh_to_xyxy(torch::Tensor x)
{
	const auto unb = x.unbind(-1);
	const auto x_c = unb[0];
	const auto y_c = unb[1];
	const auto w = unb[2];
	const auto h = unb[3];
	const std::vector<torch::Tensor> t({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h });
	return torch::stack(t, -1);
	/*torch::from_blob({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h })
	auto b = torch::from_blob(std::vector<double>({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h }).data());*/

	/*x = x.to(torch::kDouble);
	std::vector<torch::Tensor> t;
	for(int i=0;i<x.size(0);i++)
	{
		auto x_c = x[i][0].item<double>();
		auto y_c = x[i][1].item<double>();
		auto w = x[i][2].item<double>();
		auto h = x[i][3].item<double>();
		std::vector<double> v({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h });
		auto te = torch::from_blob(v.data(), { 4 }, torch::kDouble);
		t.push_back(te);
	}*/
	/*std::cout << "Box_CXCY sizes: " << x.sizes() << " " << __FILE__ << " Line: " << __LINE__ << std::endl;
	auto unb = x.unbind(-1);
	std::cout << "Box_CXCY UNBIND sizes: " << unb.size() << " " << __FILE__ << " Line: " << __LINE__ << std::endl;
	//std::cout << "UNB: " << unb << " " << __FILE__ << " " << __LINE__ << std::endl;
	
	for(int i=0;i<static_cast<int64_t>(unb.size());i++)
	{
		auto x_c = unb[i][0].item<double>();
		auto y_c = unb[i][1].item<double>();
		auto w = unb[i][2].item<double>();
		auto h = unb[i][3].item<double>();
		std::vector<double> v({ x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h });
		auto te = torch::from_blob(v.data(), { 4, 1 }, torch::kDouble);
		t.push_back(te);
		//std::cout << "Boxes: " << boxes << " device: " << boxes.get_device() << " dtype: " << boxes.dtype().name() << " size: " << boxes.sizes() << " " << __FILE__ << " Line: " << __LINE__ << std::endl;
	}*/
	return torch::stack(t);
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
	//std::cout << "Boxes size: " << boxes.sizes() << " " << __FILE__  << " Line: " << __LINE__ << std::endl;
	boxes = upcast(boxes);
	return (boxes.index({ torch::indexing::Slice(), 2 }) - boxes.index({ torch::indexing::Slice(), 0 })) * (boxes.index({ torch::indexing::Slice(), 3 }) - boxes.index({ torch::indexing::Slice(), 1 }));
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

	auto lt = torch::max(boxes1.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(torch::indexing::None, 2) }), boxes2.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2) }));
	auto rb = torch::min(boxes1.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2,torch::indexing::None) }), boxes2.index({ torch::indexing::Slice(), torch::indexing::Slice(2,torch::indexing::None) }));

	auto wh = upcast(rb - lt).clamp(0); //debería poner _upcast para prevenir overflow test_box 81
	auto inter = wh.index({ torch::indexing::Slice(),torch::indexing::Slice() , 0 }) * wh.index({ torch::indexing::Slice(),torch::indexing::Slice() , 1 });
	auto uni = area1.index({ torch::indexing::Slice(), torch::indexing::None }) + area2 - inter;
	auto iou = inter / uni;
	return std::make_tuple(iou, uni);
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
	//auto iou = std::get<0>(tup) / std::get<1>(tup);
	auto lt = torch::min(boxes1.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(torch::indexing::None, 2) }), boxes2.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2) }));
	auto rb = torch::max(boxes1.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2,torch::indexing::None) }), boxes2.index({ torch::indexing::Slice(), torch::indexing::Slice(2,torch::indexing::None) }));
	
	auto wh = upcast(rb - lt).clamp(0); //debería poner _upcast para prevenir overflow test_box 81
	auto area = wh.index({ torch::indexing::Slice(),torch::indexing::Slice() , 0 }) * wh.index({ torch::indexing::Slice(),torch::indexing::Slice() , 1 });
	return std::get<0>(tup) - (area - std::get<1>(tup)) / area;
	
}

#endif