#pragma once

#ifndef MODULES_DATASET_H
#define MODULES_DATASET_H

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "ext.h"

class dataset_detr : public torch::data::Dataset<dataset_detr, torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>>
{
protected:
	std::vector<torch::data::Example<torch::Tensor, torch::OrderedDict<std::string, torch::Tensor>>> examples_;
	int siz_data_;
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
			cv::Mat m = cv::imread(imgs[i]); //TODO: augment size width
			cv::resize(m, m, cv::Size(800, 800));
			torch::Tensor img = cv8uc3ToTensor(m, use_fp32);
			//TODO: the img size should be between 800 and 1333px WIDTH
			torch::OrderedDict<std::string, torch::Tensor> target;

			std::ifstream file(labels[i]);
			std::string str;
			std::vector<torch::Tensor> boxes_vec;
			std::vector<int> classes_vec;
			while(std::getline(file, str))
			{
				std::vector<std::string> spl;
				split_str(str, ' ', spl);
				int pos = 0;
				int id= std::stoi(spl[pos]);
				double x_c = std::stod(spl[pos++]);
				double y_c = std::stod(spl[pos++]);
				double w = std::stod(spl[pos++]);
				double h = std::stod(spl[pos]);
				boxes_vec.push_back(torch::from_blob(std::vector<double>({ x_c,y_c,w,h }).data(), { 1,4 }, torch::kFloat32));
				classes_vec.push_back(id);
			}
			torch::Tensor boxes = torch::cat(boxes_vec);
			torch::Tensor classes = torch::from_blob(classes_vec.data(), {static_cast<int64_t>(classes_vec.size()), 1});

			target.insert("boxes", boxes);
			target.insert("labels", classes);
			target.insert("orig_size", torch::from_blob(std::vector<int>({m.cols, m.rows}).data(), {1,2}, torch::kInt));
			target.insert("size", torch::from_blob(std::vector<int>({ m.cols, m.rows }).data(), { 1,2 }, torch::kInt)); //this is with augmentation

			this->examples_.push_back({ img, target });
			siz_data_++;
			if (siz_data_ % 50 == 0)
				std::cout << "Count data: " << examples_.size() << std::endl;
		}
		std::cout << "Total data size: " << examples_.size();
	}

	torch::optional<size_t> size()const override {
		return examples_.size();
	}
	ExampleType get(size_t index) override
	{
		return examples_[index];
	}
	~dataset_detr()
	{
		examples_.clear();
	}
};
#endif