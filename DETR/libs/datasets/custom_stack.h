#pragma once

#ifndef UTILS_CUSTOM_STACK_H
#define UTILS_CUSTOM_STACK_H

#include <torch/torch.h>
#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <vector>

//https://caffe2.ai/doxygen-c/html/torch_2csrc_2api_2include_2torch_2data_2transforms_2stack_8h_source.html

/*template<typename U, typename V, typename T = torch::data::Example<U,V>, typename Output = std::vector<U, std::vector<V>>>
struct CustomStackV2;*/

template<typename T = torch::data::Example<>, typename Output = std::vector<T>>
struct CustomStackV2;

template<typename U, typename V>
struct CustomStackV2<torch::data::Example<U,V>> : public torch::data::transforms::Collation<torch::data::Example<U, std::vector<V>>, std::vector<torch::data::Example<U,V>>>
{
	
	torch::data::Example<U, std::vector<V>> apply_batch(std::vector<torch::data::Example<U, V>> examples) override
	{
		std::vector<torch::Tensor> data;
		std::vector<V> vec;
		data.reserve(examples.size());
		vec.reserve(examples.size());
		for (auto &sample : examples)
		{
			data.push_back(std::move(sample.data));
			vec.push_back(std::move(sample.target));
		}
		return { torch::stack(data), vec };
	}
};

//template<typename U, typename V, typename T = torch::data::Example<U, V>, typename Output = std::vector<U, std::vector<V>>>
/*template<typename U, typename V, typename T = torch::data::Example<U, V>, typename Output = std::vector<U, std::vector<V>>>
struct CustomStackV2<T<U,V>> : public torch::data::transforms::Collation<T<U,V>, Output>
{
	
};*/

/*
template<typename T = torch::data::Example<torch::Tensor, std::vector<int>>>
struct CustomStack;

template<>
struct CustomStack<torch::data::Example<torch::Tensor, int>> : public torch::data::transforms::Collation<torch::data::Example<torch::Tensor, std::vector<int>>, std::vector<torch::data::Example<torch::Tensor, int>>>
{
	torch::data::Example<at::Tensor, std::vector<int>> apply_batch(std::vector<torch::data::Example<at::Tensor, int>> input_batch) override
	{
		std::vector<torch::Tensor> data;
		std::vector<int> vec;
		data.reserve(input_batch.size());
		vec.reserve(input_batch.size());
		for (auto& sample : input_batch)
		{
			data.push_back(sample.data);
			vec.push_back(sample.target);
		}
		return { torch::cat(data), vec };
	}
};*/

/*
 *template <typename T = Example<>>
 struct Stack;
 
 template <>
 struct Stack<Example<>> : public Collation<Example<>> {
   Example<> apply_batch(std::vector<Example<>> examples) override {
     std::vector<torch::Tensor> data, targets;
     data.reserve(examples.size());
     targets.reserve(examples.size());
     for (auto& example : examples) {
       data.push_back(std::move(example.data));
       targets.push_back(std::move(example.target));
     }
     return {torch::stack(data), torch::stack(targets)};
   }
 };
 
 template <>
 struct Stack<TensorExample>
     : public Collation<Example<Tensor, example::NoTarget>> {
   TensorExample apply_batch(std::vector<TensorExample> examples) override {
     std::vector<torch::Tensor> data;
     data.reserve(examples.size());
     for (auto& example : examples) {
       data.push_back(std::move(example.data));
     }
     return torch::stack(data);
   }
 };
 **/
#endif