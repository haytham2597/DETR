#pragma once

#ifndef LINEAR_SUM_ASSIGNMENT_H
#define LINEAR_SUM_ASSIGNMENT_H

#ifndef _IOSTREAM_
#include <iostream>
#endif

//#include <algorithm>

#include <torch/torch.h>


template<typename T>
void fill(T* arr, int64_t n, T value)
{
	for(int64_t i=0;i<n;i++)
		arr[i] = value;
}

void flip_dual_signs(double* u, double* v, int64_t nr_, int64_t nc_)
{
	for(int i=0;i<nr_;i++)
		u[i] = -u[i];
	for(int j=0;j<nc_;j++)
		v[j] = -v[j];
}

/*template<
	typename T,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class assignment
{
public:
	virtual ~assignment() = default;
	assignment(T* columnAssignment, T* rowAssignment)
	{
		ColumnAssignment = columnAssignment;
		RowAssignment = rowAssignment;
	}
	T* ColumnAssignment = nullptr, RowAssignment = nullptr;
	virtual assignment transpose()
	{
		return assignment(RowAssignment, ColumnAssignment);
	}
};
template<
	typename T,
	typename U,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class assignment_with_duals<T> : public assignment<U>
{
public:
	T* DualU, DualV;
	assignment_with_duals(T* columnAssignment, T* rowAssignment, T* dualU, T* dualV) : assignment<T>(columnAssignment, rowAssignment)
	{
		this->DualU = dualU;
		this->DualV = dualV;
	}
};*/

class assignment_with_duals
{
public:
	int* ColumnAssignment = nullptr;
	int* RowAssignment = nullptr;
	double* DualU;
	double* DualV;
	assignment_with_duals(int* columnAssignment, int* rowAssignment, double* dualU, double* dualV)
	{
		ColumnAssignment = columnAssignment;
		RowAssignment = rowAssignment;
		this->DualU = dualU;
		this->DualV = dualV;
	}
	assignment_with_duals transpose()
	{
		return assignment_with_duals(RowAssignment, ColumnAssignment, DualV, DualU);
	}
};

class ISolver
{
protected:
	int64_t nr_ = 0, nc_ = 0;
public:
	virtual assignment_with_duals solve(torch::Tensor cost) = 0;
};

class ShortestPathSolver : public ISolver
{
public:
	ShortestPathSolver()
	{
		
	}

	assignment_with_duals solve(torch::Tensor cost) override
	{
		this->nr_ = cost.size(0);
		this->nc_ = cost.size(1);

		auto u = new double[nr_];
		auto v = new double[nc_];
		auto shortestPathCosts = new double[nc_];
		auto path = new int[nc_];
		auto x = new int[nr_];
		auto y = new int[nc_];
		auto sr = new bool[nr_];
		auto sc = new bool[nc_];
		fill(path, nc_, -1);
		fill(x, nr_, -1);
		fill(y, nc_, -1);

		for(int64_t curRow = 0;curRow < nr_;curRow++)
		{
			double minVal = 0;
			auto i = curRow;
			auto remaining = new int[nc_];
			auto numRemaining = nc_;
			for(int64_t jp = 0; jp < nc_;jp++)
			{
				remaining[jp] = jp;
				shortestPathCosts[jp] = std::numeric_limits<double>::infinity();
				fill(sr, nr_, false);
				fill(sc, nc_, false);

				auto sink = -1;
				while(sink == -1)
				{
					sr[i] = true;
					auto indexLowest = -1;
					auto lowest = std::numeric_limits<double>::infinity();
					for(int64_t jk = 0;jk<numRemaining;jk++)
					{
						auto jl = remaining[jk];
						auto r = minVal + cost[i][jl].item<double>() - u[i] - v[jl];
						if(r < shortestPathCosts[jl])
						{
							path[jl] = i;
							shortestPathCosts[jl] = r;
						}
						if(shortestPathCosts[jl] < lowest || shortestPathCosts[jl] == lowest && y[jl] == -1)
						{
							lowest = shortestPathCosts[jl];
							indexLowest = jk;
						}
					}

					minVal = lowest;
					auto jp = remaining[indexLowest];
					if(std::isinf(minVal))
						throw std::exception("No feasible solution");
					if (y[jp] == -1) {
						sink = jp;
					}
					else
					{
						i = y[jp];
					}
					sc[jp] = true;
					remaining[indexLowest] = remaining[--numRemaining];
					for(int rem = numRemaining;rem < nc_;++rem)
						remaining[rem] = remaining[rem + 1];
				}

				if (sink < 0)
					throw std::exception("No feasible solution");

				u[curRow] += minVal;
				for (auto ip = 0; ip < nr_; ip++)
					if (sr[ip] && ip != curRow)
						u[ip] += minVal - shortestPathCosts[x[ip]];
				for (auto jp = 0; jp < nc_; jp++)
					if (sc[jp])
						v[jp] -= minVal - shortestPathCosts[jp];

				auto j = sink;
				while(true)
				{
					auto ip = path[j];
					y[j] = ip;
					auto temp = x[ip];
					x[ip] = j;
					j = temp;
					if (ip == curRow)
						break;
				}
			}
			return assignment_with_duals(x, y, u, v);

		}
	}
};

class PseudoflowSolver : public ISolver
{
private:
	double epsilon_;
	double alpha_;
public:
	PseudoflowSolver(double alpha = 10, double epsilon = -1)
	{
		this->alpha_ = alpha;
		this->epsilon_ = epsilon;
	}
	assignment_with_duals solve(torch::Tensor cost) override
	{
		std::vector<int> ia, a, ca;
		this->nr_ = cost.size(0);
		this->nc_ = cost.size(1);
		if (this->nr_ != this->nc_)
			throw std::exception("Pseudoflow is only implmented for square matrix");

		auto n = nr_;
		if(epsilon_ == -1) //no value
		{
			epsilon_ = -std::numeric_limits<double>::infinity();
			for(int i=0;i<n;i++)
			for(int j=0;j<n;j++)
				if (cost[i][j].item<double>() > epsilon_ && cost[i][j].item<double>() != std::numeric_limits<int>::max())
					epsilon_ = cost[i][j].item<double>();
		}

		auto v = new double[n];
		auto col = new int[n];
		auto row = new int[n];
		while (epsilon_ >= (double)1 / n)
		{
			epsilon_ /= alpha_;
			for (int i = 0; i < n; i++)
				col[i] = -1;
			for (int j = 0; j < n; j++)
				row[j] = -1;
			std::vector<int> unassigned;
			for (int i = 1; i < n; i++)
				unassigned.push_back(i);
			/*for(int i=n-1;i --> 0;) //reverse
				unassigned.push_back(i);*/
			auto k = 0;
			while(true)
			{
				auto smallest = std::numeric_limits<double>::infinity();
				auto j = -1;
				auto secondSmallest = std::numeric_limits<double>::infinity();
				for(auto jp=0;jp<n;jp++)
				{
					auto partialReducedCost = cost[k][jp].item<double>() - v[jp];
					if(partialReducedCost <= secondSmallest)
					{
						if(partialReducedCost <= smallest)
						{
							secondSmallest = smallest;
							smallest = partialReducedCost;
							j = jp;
						}
						else
						{
							secondSmallest = partialReducedCost;
						}
					}
				}
				col[k] = j;
				if(row[j] != -1)
				{
					auto i = row[j];
					row[j] = k;

					v[j] += smallest - secondSmallest - epsilon_;
					col[i] = -1;
					k = i;
				}
				else
				{
					row[j] = k;
					if (unassigned.empty())
						break;
					k = unassigned.front();
					unassigned.erase(unassigned.begin());
				}
			}
		}
		return assignment_with_duals(col, row, nullptr, nullptr);
	}
};

class solver
{
private:
	/// <summary>
	/// Number Row and Number Column
	/// </summary>
	int64_t nr_, nc_;
	torch::Tensor cost_;
	bool maximize_;
	bool transpose()
	{
		if (nr_ <= nc_)
			return false;
		cost_ = cost_.t();
		return true;
	}
public:
	solver(torch::Tensor cost, bool maximize, bool allowOverwrite)
	{
		nr_ = cost.size(0);
		nc_ = cost.size(1);
		cost_ = cost;
		this->maximize_ = maximize;
		if (nr_ == 0 || nc_ == 0)
			return;
	}

	assignment_with_duals solve()
	{
		auto trans = transpose();
		if (maximize_) {
			cost_ = cost_.neg();
		}
		auto min = std::numeric_limits<double>::infinity();
		for(int i=0;i<nr_;i++)
		{
			for(int j=0;j<nc_;j++)
			{
				if(cost_[i][j].item<double>() < min)
				{
					min = cost_[i][j].item<double>();
				}
			}
		}
		if (min < 0)
		{
			for (int i = 0; i < nr_; i++)
			{
				for (int j = 0; j < nc_; j++)
				{
					cost_[i][j] -= min;
				}
			}
		}else
		{
			min = 0;
		}
		ISolver* solver = nullptr;
		
		if(nr_ != nc_)
		{
			solver = new ShortestPathSolver();
			
		}
		else
		{
			solver = new PseudoflowSolver();

		}
		assignment_with_duals solution = solver->solve(cost_);
		if (min != 0)
			for (auto ip = 0; ip < nr_; ip++)
				solution.DualU[ip] += min;
		if (maximize_)
			flip_dual_signs(solution.DualU, solution.DualV, nr_, nc_);

		if (trans)
			solution = solution.transpose();
		return solution;
		//return assignment_with_duals(nullptr, nullptr, nullptr, nullptr);


	}
};

#endif