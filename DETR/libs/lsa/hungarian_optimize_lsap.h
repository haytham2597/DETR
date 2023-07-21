#pragma once

#ifndef LIBS_HUNGARIAN_OPTIMIZE_LSAP
#define LIBS_HUNGARIAN_OPTIMIZE_LSAP

#ifndef _IOSTREAM_
#include <iostream>
#endif

#include <torch/torch.h>

/**
 * \brief STATE of the Hungarian
 */
class Hungary
{
public:
	torch::Tensor c_, row_uncovered_, col_uncovered_, path_, marked_;
	int64_t z0_r_, z0_c_;
	bool maximize_;
	Hungary(torch::Tensor cost_matrix, bool maximize)
	{
		this->c_ = cost_matrix.clone();
		int64_t n = c_.size(0);
		int64_t m = c_.size(1);

		this->row_uncovered_ = torch::ones({ n }, torch::kBool);
		this->col_uncovered_ = torch::ones({ m }, torch::kBool);

		this->z0_r_ = 0;
		this->z0_c_ = 0;

		this->path_ = torch::zeros({ n + m, 2 }, torch::kInt);
		this->marked_ = torch::zeros({ n,m }, torch::kInt);
		this->maximize_ = maximize;
	}

	void clear_covers()
	{
		this->row_uncovered_.index_put_({ torch::indexing::Slice() }, true);
		this->col_uncovered_.index_put_({ torch::indexing::Slice() }, true);
		/*this->row_uncovered_ = this->row_uncovered_.fill_(true);
		this->col_uncovered_ = this->col_uncovered_.fill_(true);*/
	}
};

class LinearSumAssignment
{
private:
	torch::Tensor Row, Col;
	Hungary* state_ = nullptr;
	///std::map<std::string, std::function<bool>> maps_of_steps;
public:
	LinearSumAssignment()
	{
		
	}
	LinearSumAssignment(torch::Tensor costMatrix, bool maximize = false)
	{
		//maps_of_steps.emplace(std::make_pair("step1", &LinearSumAssignment::step1));
		//maps_of_steps.emplace(std::make_pair("step2", &LinearSumAssignment::step2));
		if (costMatrix.sizes().size() != 2)
			throw std::exception("The tensor should be 2D");

		bool transposed = false;
		if(costMatrix.size(1) > costMatrix.size(0))
		{
			costMatrix = costMatrix.t(); //Transpose
			transposed = true;
		}

		state_ = new Hungary(costMatrix, maximize);
		//auto state = Hungary(costMatrix, maximize);

		bool step = true;
		for (uint64_t i = 0; i < costMatrix.sizes().size(); i++)
			if (costMatrix.sizes()[i] == 0)
				step = false;

		step1();
		if(!step)
		{
			step1();
			//Only need call step1
		}
		else {
			step = step1();
		}
		torch::Tensor marked;
		if(transposed)
		{
			marked = state_->marked_.t();
		}else
		{
			marked = state_->marked_;
		}
		auto ple = torch::where(marked == 1);
		Row = ple[0];
		Col = ple[1];
		//MESSAGE_LOG("RESULT: ")
		/*for(int i=0;i<ple.size();i++)
		{
			std::cout << "Ple[" << std::to_string(i) << "]: " << ple[i] << std::endl;
		}*/
	}
	std::tuple<torch::Tensor, torch::Tensor> Solve(torch::Tensor costMatrix, bool maximize = false)
	{
		//maps_of_steps.emplace(std::make_pair("step1", &LinearSumAssignment::step1));
		//maps_of_steps.emplace(std::make_pair("step2", &LinearSumAssignment::step2));
		if (costMatrix.sizes().size() != 2)
			throw std::exception("The tensor should be 2D");

		bool transposed = false;
		if (costMatrix.size(1) > costMatrix.size(0))
		{
			costMatrix = costMatrix.t(); //Transpose
			transposed = true;
		}

		state_ = new Hungary(costMatrix, maximize);
		//auto state = Hungary(costMatrix, maximize);

		bool step = true;
		for (uint64_t i = 0; i < costMatrix.sizes().size(); i++)
			if (costMatrix.sizes()[i] == 0)
				step = false;

		step1();
		/*if (!step)
		{
			step1();
			//Only need call step1
		}
		else {
			step = step1();
		}*/
		torch::Tensor marked;
		if (transposed)
		{
			marked = state_->marked_.t();
		}
		else
		{
			marked = state_->marked_;
		}
		auto ple = torch::where(marked == 1);
		Row = ple[0];
		Col = ple[1];
		return std::make_tuple(this->Row, this->Col);
	}
	/*std::tuple<torch::Tensor, torch::Tensor> GetResult()
	{
		
	}*/

	//[SOLVED]
	bool step1()
	{
		//MESSAGE_LOG("Step1")
		if (this->state_->maximize_) {
			state_->c_ -= std::get<0>(state_->c_.max(1)).index({ torch::indexing::Slice(), torch::indexing::None });
		}
		else
		{
			state_->c_ -= std::get<0>(state_->c_.min(1)).index({ torch::indexing::Slice(), torch::indexing::None });
		}
		auto where_vec = torch::where(state_->c_ == 0);
		int64_t to = (where_vec[0].size(0) > where_vec[1].size(0) ? where_vec[0].size(0) : where_vec[1].size(0));
		for(int64_t ij=0;ij<to;ij++)
		{
			auto i = where_vec[0][ij].item<int>();
			auto j = where_vec[1][ij].item<int>();
			if (state_->col_uncovered_[j].item<bool>() && state_->row_uncovered_[i].item<bool>())
			{
				state_->marked_.index_put_({ i, j }, 1);
				state_->col_uncovered_.index_put_({ j }, false);
				state_->row_uncovered_.index_put_({ i }, false);
				/*state_->col_uncovered_[j] = false;
				state_->row_uncovered_[i] = false;*/
			}
		}
		state_->clear_covers();
		return step3();
	}


	bool step3()
	{
		//MESSAGE_LOG("Step3")
		auto marked = state_->marked_ == 1;
		auto anyTensor = torch::any(marked, 0);
		state_->col_uncovered_.index_put_({ anyTensor }, false);
		if(marked.sum().item<int>() < state_->c_.size(0))
			return step4();
		return false;
	}

	bool step4()
	{
		//MESSAGE_LOG("Step4")
		const auto c = (state_->c_ == 0).to(torch::kInt);
		auto covered_c = c * state_->row_uncovered_.index({ torch::indexing::Slice(), torch::indexing::None });
		covered_c *= state_->col_uncovered_.to(torch::kInt);
		auto n = state_->c_.size(0);
		auto m = state_->c_.size(1);
		for(;;)
		{
			auto vec_rowcol = unravel_index(covered_c.argmax(), { n,m });
			auto row = vec_rowcol[0].item<int>();
			auto col = vec_rowcol[1].item<int>();
			if(covered_c.index({row, col}).item<int>() == 0)
			{
				return step6();
			}
			else
			{
				state_->marked_.index_put_({ row, col }, 2);
				auto star_col = (state_->marked_.index({row}) == 1).to(torch::kInt).argmax().item<int>();
				if(state_->marked_.index({row, star_col}).item<int>() != 1)
				{
					state_->z0_r_ = row;
					state_->z0_c_ = col;
					return step5();
				}
				else
				{
					col = star_col;
					state_->row_uncovered_.index_put_({ row }, false);
					state_->col_uncovered_.index_put_({ col }, true);
					covered_c.index({ torch::indexing::Slice(), col }) = c.index({ torch::indexing::Slice(), col }) * state_->row_uncovered_.to(torch::kInt);
					covered_c.index_put_({ row }, 0);
				}
			}
		}
		return false;
	}

	bool step5()
	{
		//MESSAGE_LOG("Step5");
		int count = 0;
		auto path = state_->path_;
		path.index_put_({ count, 0 }, state_->z0_r_);
		path.index_put_({ count, 1 }, state_->z0_c_);
		while(true)
		{
			auto row = (state_->marked_.index({ torch::indexing::Slice(), path.index({count, 1}) }) == 1).to(torch::kInt).argmax().item<int>();
			if (state_->marked_.index({ row, path.index({count, 1}) }).item<int>() != 1) {
				break;
			}
			else
			{
				count += 1;
				path.index_put_({ count, 0 }, row);
				path.index_put_({ count, 1 }, path.index({count-1, 1}));
			}
			auto col = (state_->marked_.index({ path.index({count, 0}) }) == 2).to(torch::kInt).argmax().item<int>();
			if (state_->marked_.index({ row, col }).item<int>() != 2)
				col = -1;
			count += 1;
			path.index_put_({ count, 0 }, path.index({ count - 1, 0 }));
			path.index_put_({ count, 1 }, col);
		}

		for(int i=0;i<count+1;i++)
		{
			if (state_->marked_.index({ path.index({i,0}),path.index({i,1}) }).item<int>() == 1)
			{
				state_->marked_.index_put_({ path.index({i,0}),path.index({i,1}) }, 0);
			}
			else
			{
				state_->marked_.index_put_({ path.index({i,0}),path.index({i,1}) }, 1);
			}
		}
		state_->clear_covers();
		state_->marked_.index_put_({ state_->marked_ == 2 }, 0);
		//state_->marked_.index_put_({ torch::cat(torch::where(state_->marked_ == 2)) }, 0);
		return step3();
	}

	bool step6()
	{
		//MESSAGE_LOG("Step6")
		if(torch::any(state_->row_uncovered_).item<bool>() && torch::any(state_->col_uncovered_).item<bool>())
		{
			torch::Tensor minVal;
			if(state_->maximize_)
			{
				minVal = std::get<0>(torch::max(state_->c_.index({ state_->row_uncovered_ }), 0));
				minVal = torch::max(minVal.index({ state_->col_uncovered_ }));
			}else
			{
				minVal = std::get<0>(torch::min(state_->c_.index({ state_->row_uncovered_ }), 0));
				minVal = torch::min(minVal.index({ state_->col_uncovered_ }));
			}
			state_->c_.index({ ~state_->row_uncovered_ }) += minVal;
			state_->c_.index({ { torch::indexing::Slice(), state_->col_uncovered_} }) -= minVal;
			/*state_->c_.index_put_({ ~state_->row_uncovered_ }, state_->c_.index({ ~state_->row_uncovered_ }) += minVal);
			state_->c_.index_put_({ torch::indexing::Slice(), state_->col_uncovered_}, state_->c_.index({ torch::indexing::Slice(), state_->col_uncovered_ }) -= minVal);*/
		}
		return step4();
	}
};

#endif