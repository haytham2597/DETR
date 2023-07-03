#pragma once

#ifndef LIBS_EXT_H
#define LIBS_EXT_H

#include <opencv2/imgproc.hpp>

bool anyext(torch::Tensor x)
{
	for (int i = 0; i < x.size(0); i++)
		if (x[i].item<bool>())
			return true;
	return false;
}
std::vector<torch::Tensor> unravel_index(torch::Tensor indices, const at::IntArrayRef shape)
{
	std::vector<torch::Tensor> coord;
	for (int64_t i = shape.size(); i --> 0;) //reverse
	{
		coord.push_back(indices % shape[i]);
		//indices = indices / shape[i];
		indices = indices.div(shape[i], "floor");
	}
	std::reverse(coord.begin(), coord.end());
	return coord;
	//return torch::stack(coord, -1);
}

inline bool is_power_of_2(int n)
{
	return ((n & (n - 1)) == 0) && n != 0;
}

inline torch::Tensor inverse_sigmoid(torch::Tensor x, double eps = 1e-5)
{
	x = x.clamp(0, 1);
	return torch::log(x.clamp(eps) / (1 - x).clamp(eps));
}

template<typename Contained>
torch::nn::ModuleHolder<Contained> get_activation(std::string activation = "relu")
{
	if(activation == "relu")
		return torch::nn::ModuleHolder<torch::nn::ReLUImpl>();
	if (activation == "gelu")
		return torch::nn::ModuleHolder<torch::nn::GELUImpl>();
	return torch::nn::ModuleHolder<torch::nn::GLUImpl>();
}

inline torch::Tensor with_pos_embed(torch::Tensor tensor, torch::Tensor pos)
{
	if (!pos.defined() || pos.numel() == 0) //epmpty TODO:TORCH.LUMEL EMPTY
		return tensor;
	return tensor + pos;
}

inline torch::Tensor with_pos_embed(torch::Tensor tensor, torch::optional<torch::Tensor> pos = torch::nullopt)
{
	if(!pos.has_value())
		return tensor;
	return tensor + pos.value();
}

inline torch::Tensor sigmoid_focal_loss(torch::Tensor inputs, torch::Tensor targets, torch::Scalar num_boxes, float alpha = 0.25, float gamma = 2)
{
	auto prob = inputs.sigmoid();
	///torch::nn::BCEWithLogitsLossOptions::reduction_t l(torch::kNone);
	//auto ce_loss = torch::nn::functional::binary_cross_entropy_with_logits(inputs, targets, torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
	auto ce_loss = torch::nn::functional::binary_cross_entropy_with_logits(inputs, targets, torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
	auto p_t = prob * targets + (1 - prob) * (1 - targets);
	auto loss = ce_loss * ((1 - p_t).pow(gamma));
	if(alpha >= 0)
	{
		auto alpha_t = alpha * targets + (1 - alpha) * (1 - targets);
		loss = alpha_t * loss;
	}
	return loss.mean(1).sum() / num_boxes;
}

inline std::vector<torch::Tensor> accuracy(torch::Tensor output, torch::Tensor target, int topk = 1)
{
	if (target.numel() == 0) {
		return { torch::zeros({0}) };
	}
	auto maxk = topk;
	auto bs = target.size(0);
	auto pred = std::get<1>(output.topk(maxk, 1, true, true));
	pred = pred.t();
	auto correct = pred.eq(target.view({ 1,-1 }).expand_as(pred));
	std::vector<torch::Tensor> res;
	for(int i=0;i<topk;i++)
	{
		auto correct_k = correct.index({ torch::indexing::Slice(), i }).view({ -1 }).to(torch::kFloat).sum(0);
		res.push_back(correct_k.mul_(100/bs));
	}
	return res;
}

inline void split_str(std::string const& str, const char delim, std::vector<std::string>& out)
{
	// construct a stream from the string 
	std::stringstream ss(str);

	std::string s;
	while (std::getline(ss, s, delim))
		out.push_back(s);
}
inline torch::Tensor cv8uc3ToTensor(cv::Mat frame, bool use_fp32 = true)
{
/*#ifndef _DEBUG
	frame.convertTo(frame, use_fp32 ? CV_32FC(frame.channels()) : CV_16FC(frame.channels()));
#else
	frame.convertTo(frame, use_fp32 ? CV_32FC(frame.channels()) : CV_16FC(frame.channels()));
	//frame.convertTo(frame, CV_32FC3);
#endif*/
	frame.convertTo(frame, use_fp32 ? CV_32FC(frame.channels()) : CV_16FC(frame.channels()));
	auto input_tensor = torch::from_blob(frame.data, { frame.rows, frame.cols, frame.channels() }, use_fp32 ? at::kFloat : at::kHalf);
	input_tensor = input_tensor.permute({ 2, 1, 0 });
	input_tensor = input_tensor.div(255);
	return input_tensor;
}
#endif