#pragma once

#ifndef MODULES_DATASET_H
#define MODULES_DATASET_H

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "../util/ext.h"

class dataset_detr : public torch::data::Dataset<dataset_detr, torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>
{
protected:
	//std::vector<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>> examples_;
	std::vector<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>> examples_;
	int siz_data_ =0;
	//std::mutex* mtx = new std::mutex();
public:
	dataset_detr()
	{
		
	}
	void add_data(std::vector<std::string> imgs, std::vector<std::string> labels, bool use_fp32 = true)
	{
		if (imgs.size() != labels.size())
			return;
		for(int i=0;i<static_cast<int>(imgs.size() > 50 ? 50 : imgs.size());i++)
		{
			torch::OrderedDict<std::string, torch::Tensor> target;
			cv::Mat m = cv::imread(imgs[i]); //TODO: augment size width
			auto w = m.cols;
			auto h = m.rows;
			
			const cv::Size2d original = cv::Size2d(m.cols, m.rows);
			cv::resize(m, m, cv::Size(800, 800));
			const cv::Size2d res = cv::Size2d(m.cols, m.rows);
			torch::Tensor img = cv8uc3ToTensor(m, use_fp32);
			//TODO: the img size should be between 800 and 1333px WIDTH
			std::ifstream file(labels[i]);
			std::string str;
			std::vector<torch::Tensor> boxes_vec;
			std::vector<float> boxescomple;
			std::vector<int64_t> classes_vec;
			int count = 0;
			while(std::getline(file, str))
			{
				std::vector<std::string> spl;
				split_str(str, ' ', spl);
				int pos = 0; 
				int id = std::stoi(spl[pos++]);
				double x_c =  std::stod(spl[pos++])* static_cast<double>(res.width) / static_cast<double>(original.width);
				double y_c = std::stod(spl[pos++])* static_cast<double>(res.height) / static_cast<double>(original.height);
				double w = std::stod(spl[pos++])* static_cast<double>(res.width) / static_cast<double>(original.width);
				double h = std::stod(spl[pos++])* static_cast<double>(res.height) / static_cast<double>(original.height);
				boxescomple.push_back(x_c);
				boxescomple.push_back(y_c);
				boxescomple.push_back(w);
				boxescomple.push_back(h);
				
				classes_vec.push_back(id);
				count++;
			}
			auto label = torch::from_blob(classes_vec.data(), { static_cast<int64_t>(classes_vec.size()) }, torch::kInt64).clone();
			auto size = torch::from_blob(std::vector<int>({ m.cols, m.rows }).data(), { 2,1 }, torch::kInt).clone();
			auto boxesco = torch::from_blob(boxescomple.data(), { count, 4 }, torch::kFloat).clone();
			
			target.insert("boxes", boxesco);
			target.insert("labels", label);
			target.insert("orig_size", torch::from_blob(std::vector<int>({ w, h }).data(), { 2,1 }, at::kInt).clone());
			target.insert("size", size);
			this->examples_.push_back({ img, target });

			//std::cout << "Get labels: " << this->examples_[this->examples_.size()-1].target["labels"] << std::endl;
			//this->examples_.emplace_back(img, target);
			siz_data_++;
			if (siz_data_ % 50 == 0)
				std::cout << "Count data: " << examples_.size() << std::endl;
			file.close();
			boxes_vec.clear();
			classes_vec.clear();
			target.clear();
		}
		std::cout << "Total data size: " << examples_.size() << std::endl;
	}

	torch::optional<size_t> size()const override {
		std::cout << "Size dataset: " << examples_.size();
		return examples_.size();
	}
	ExampleType get(size_t index) override
	{
		return examples_[index];
	}

	/*~dataset_detr() override
	{
		examples_.clear();
	}*/
};
#endif