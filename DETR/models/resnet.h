#pragma once

#ifndef MODELS_RESNET_H
#define MODELS_RESNET_H

#include <torch/torch.h>
#include <iostream>
#include "../libs/layers.h"

//TODO: replace_stride_dilation 
template<typename Contained>
struct BasicBlock : torch::nn::Module {

    static const int expansion;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::ModuleHolder<Contained> bn1 = nullptr;
    torch::nn::Conv2d conv2;
    torch::nn::ModuleHolder<Contained> bn2 = nullptr;
    torch::nn::Sequential downsample;
    BasicBlock(
        int64_t inplanes, 
        int64_t planes, 
        int64_t stride_ = 1, 
        torch::nn::Sequential downsample_ = torch::nn::Sequential()
        //Contained c = nullptr
    )
        : conv1(torch::nn::Conv2dOptions(inplanes, planes, { 3,3}).stride({stride_,stride_}).padding({1,1})),
        //bn1(planes),
        conv2(torch::nn::Conv2dOptions(planes, planes, { 3,3 }).stride({1,1}).padding({1,1})),
        //bn2(planes),
        downsample(downsample_)
    {
        /*std::string class_name(typeid(FrozenBatchNorm2dImpl()).name());
        std::string this_name(typeid(c).name());
        if((class_name.find(this_name)) != std::string::npos) //FrozenBatchNorm2dImpl is finded
        {
	        
        }*/
        bn1 = torch::nn::ModuleHolder<Contained>(planes);
        bn2 = torch::nn::ModuleHolder<Contained>(planes);
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        stride = stride_;
        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->is_empty()) {
            residual = downsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

template<typename Contained>
const int BasicBlock<Contained>::expansion = 1;

template<typename Contained>
struct BottleNeck : torch::nn::Module {

    static const int expansion;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::ModuleHolder<Contained> bn1 = nullptr;
    torch::nn::Conv2d conv2;
    torch::nn::ModuleHolder<Contained> bn2 = nullptr;
    torch::nn::Conv2d conv3;
    torch::nn::ModuleHolder<Contained> bn3 = nullptr;
    torch::nn::Sequential downsample;
    //torch::nn::Sequential bn;

    BottleNeck(
        int64_t inplanes, 
        int64_t planes, 
        int64_t stride_ = 1, 
        torch::nn::Sequential downsample_ = torch::nn::Sequential() 
        )
        : conv1(torch::nn::Conv2dOptions(inplanes, planes, {1,1})),
        //bn1(planes),
        conv2(torch::nn::Conv2dOptions(planes, planes, { 3,3 }).stride({stride_,stride_}).padding({1,1})),
        //bn2(planes),
        conv3(torch::nn::Conv2dOptions(planes, planes*expansion, { 1,1 })),
        //bn3(planes* expansion),
        downsample(downsample_)
    {
        bn1 = torch::nn::ModuleHolder<Contained>(planes);
        bn2 = torch::nn::ModuleHolder<Contained>(planes);
        bn3 = torch::nn::ModuleHolder<Contained>(planes*expansion);
        /*if(norm_layer.is_empty())
        {
            //torch::nn::ModuleHolder<torch::nn::BatchNorm2dImpl> b = FrozenBatchNorm2d(planes).get()->get();
            bn1 = FrozenBatchNorm2d(planes);
            register_module("bn1", );
            register_module("bn1", norm_layer(torch::nn::BatchNorm2dOptions(planes));
        }*/
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        stride = stride_;
        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);
        x = torch::relu(x);

        x = conv3->forward(x);
        x = bn3->forward(x);

        if (!downsample->is_empty()) {
            residual = downsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

template<typename Contained>
const int BottleNeck<Contained>::expansion = 4;


template <class Block, typename Contained> struct ResNet : torch::nn::Module {
private:
    static void init_weights(torch::nn::Module& module)
    {
	    if(auto* conv = module.as<torch::nn::Conv2d>())
	    {
            torch::nn::init::xavier_normal_(conv->weight);
            /*for(auto* p : conv->parameters())
            {
                torch::nn::init::xavier_normal_(p.value());
            }*/
            //torch::nn::init::xavier_normal_(conv->parameters().value);
	    }
        if(auto* batch_norm = module.as<torch::nn::BatchNorm2d>())
        {
            torch::nn::init::constant_(batch_norm->weight, 1);
            torch::nn::init::constant_(batch_norm->bias, 0);
        }
        /*if (auto* frozen_Bn = module.as<FrozenBatchNorm2d>())
        {
            torch::nn::init::constant_(frozen_Bn->weight, 1);
            torch::nn::init::constant_(frozen_Bn->bias, 0);
        }*/
        /*
         * else if (m.value.name() == "torch::nn::BatchNormImpl") {
                for (auto p : m.value.parameters()) {
                    if (p.key == "weight") {
                        torch::nn::init::constant_(p.value, 1);
                    }
                    else if (p.key == "bias") {
                        torch::nn::init::constant_(p.value, 0);
                    }
                }
            }
         **/
    }

	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false) {
        torch::nn::Sequential downsample;
        /*if (norm_layer.is_empty())
            norm_layer = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * Block::expansion));*/
        if (stride != 1 or inplanes != planes * Block::expansion) {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes* Block::expansion, {1,1}).stride({stride,stride})),
                torch::nn::ModuleHolder<Contained>(planes * Block::expansion)
                //torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * Block::expansion))
            );
        }
        torch::nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int64_t i = 0; i < blocks; i++) {
            layers->push_back(Block(inplanes, planes));
        }

        return layers;
    }
public:
    int64_t inplanes = 64;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::ModuleHolder<Contained> bn1 = nullptr;
    //torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc;

    ResNet(torch::IntArrayRef layers, int64_t num_classes = 1000)
        : conv1(torch::nn::Conv2dOptions(3, 64, { 7,7 }).stride({ 2,2 }).padding({ 3,3 })),
        //bn1(64),
        layer1(_make_layer(64, layers[0])),
        layer2(_make_layer(128, layers[1], 2)),
        layer3(_make_layer(256, layers[2], 2)),
        layer4(_make_layer(512, layers[3], 2)),
        fc(512 * Block::expansion, num_classes)
    {
        bn1 = torch::nn::ModuleHolder<Contained>(64);
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);
        this->apply(init_weights);
        // Initializing weights
        /*for (auto m : this->modules()) {
            if (m.value.name() == "torch::nn::Conv2dImpl") {
                for (auto p : m.value.parameters()) {
                    torch::nn::init::xavier_normal_(p.value);
                }
            }
            else if (m.value.name() == "torch::nn::BatchNormImpl") {
                for (auto p : m.value.parameters()) {
                    if (p.key == "weight") {
                        torch::nn::init::constant_(p.value, 1);
                    }
                    else if (p.key == "bias") {
                        torch::nn::init::constant_(p.value, 0);
                    }
                }
            }
        }*/
    }

    torch::Tensor forward(torch::Tensor x) {

        //std::cout << "Size x (from Resnet.h 265): " << x.sizes() << std::endl;
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 2, 1);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        if (!fc.is_empty()) {
            x = torch::avg_pool2d(x, 7, 1);
            x = x.view({ x.sizes()[0], -1 });
            x = fc->forward(x);
        }
      
        return x;
    }
};

template<typename Contained>
ResNet<BasicBlock<Contained>, Contained> resnet18() {
    ResNet<BasicBlock<Contained>, Contained> model({ 2, 2, 2, 2 });
    return model;
}

template<typename Contained>
ResNet<BasicBlock<Contained>, Contained> resnet34() {
    ResNet<BasicBlock<Contained>, Contained> model({ 3, 4, 6, 3 });
    return model;
}

template<typename Contained>
ResNet<BottleNeck<Contained>, Contained> resnet50() {
    ResNet<BottleNeck<Contained>,Contained> model({ 3, 4, 6, 3 });
    return model;
}

template<typename Contained>
ResNet<BottleNeck<Contained>, Contained> resnet101() {
    ResNet<BottleNeck<Contained>,Contained> model({ 3, 4, 23, 3 });
    return model;
}

template<typename Contained>
ResNet<BottleNeck<Contained>, Contained> resnet152() {
    ResNet<BottleNeck<Contained>, Contained> model({ 3, 8, 36, 3 });
    return model;
}

#endif