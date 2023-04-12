#pragma once

#ifndef UTILS_LSAP_H
#define UTILS_LSAP_H


#define RECTANGULAR_LSAP_INFEASIBLE -1
#define RECTANGULAR_LSAP_INVALID -2

#include <cstdint>
#include <vector>

namespace linear_sum_assignment {

    template <typename T> std::vector<intptr_t> argsort_iter(const std::vector<T>& v)
    {
        std::vector<intptr_t> index(v.size());
        std::iota(index.begin(), index.end(), 0);
        std::sort(index.begin(), index.end(), [&v](intptr_t i, intptr_t j)
            {return v[i] < v[j]; });
        return index;
    }

    static intptr_t augmenting_path(
        intptr_t nc,
        std::vector<double> cost,
        std::vector<double>& u,
        std::vector<double>& v,
        std::vector<intptr_t>& path,
        std::vector<intptr_t>& row4col,
        std::vector<double>& shortestPathCosts,
        intptr_t i,
        std::vector<bool>& SR, std::vector<bool>& SC,
        std::vector<intptr_t>& remaining, double* p_minVal)
    {
        double minVal = 0;

        // Crouse's pseudocode uses set complements to keep track of remaining
        // nodes.  Here we use a vector, as it is more efficient in C++.
        intptr_t num_remaining = nc;
        for (intptr_t it = 0; it < nc; it++) {
            // Filling this up in reverse order ensures that the solution of a
            // constant cost matrix is the identity matrix (c.f. #11602).
            remaining[it] = nc - it - 1;
        }

        std::fill(SR.begin(), SR.end(), false);
        std::fill(SC.begin(), SC.end(), false);
        std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), INFINITY);

        // find shortest augmenting path
        intptr_t sink = -1;
        while (sink == -1) {

            intptr_t index = -1;
            double lowest = INFINITY;
            SR[i] = true;

            for (intptr_t it = 0; it < num_remaining; it++) {
                intptr_t j = remaining[it];
                double r = minVal + cost[i * nc + j] - u[i] - v[j];
                if (r < shortestPathCosts[j]) {
                    path[j] = i;
                    shortestPathCosts[j] = r;
                }

                // When multiple nodes have the minimum cost, we select one which
                // gives us a new sink node. This is particularly important for
                // integer cost matrices with small co-efficients.
                if (shortestPathCosts[j] < lowest ||
                    (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                    lowest = shortestPathCosts[j];
                    index = it;
                }
            }

            minVal = lowest;
            if (minVal == INFINITY)// infeasible cost matrix
                return -1;

            intptr_t j = remaining[index];
            if (row4col[j] == -1) {
                sink = j;
            }
            else {
                i = row4col[j];
            }

            SC[j] = true;
            remaining[index] = remaining[--num_remaining];
        }

        *p_minVal = minVal;
        return sink;
    }

    static int solve(torch::Tensor te, bool maximize, std::tuple<torch::Tensor, torch::Tensor>& indices)
    {
        //INFO Assuming "te" is a Tensor rectangular matrix [M,N] OR [N,N]
        int nr = static_cast<int>(te.size(0));
        int nc = static_cast<int>(te.size(1));
        if (nr == 0 || nc == 0)
            return 0;
        const bool transpose = nc < nr;
        if (transpose)
            te = te.transpose(0, 1);
        te = te.flatten(); //Flatten in one dimension
        std::vector<double> cost = std::vector<double>(nr * nc);
        if (transpose) {
            for (intptr_t i = 0; i < nr; i++)
                for (intptr_t j = 0; j < nc; j++)
                    cost[j * nr + i] = te[i * nc + j].item<double>();

            std::swap(nr, nc);
        }
        else {
            for (intptr_t i = 0; i < nr; i++)
                for (intptr_t j = 0; j < nc; j++)
                    cost[i * nr + j] = te[i * nc + j].item<double>();
        }
        if (maximize) {
            for (int i = 0; i < cost.size(); i++) {
                cost[i] = -cost[i];
                //test for NaN and -inf entries
                if (cost[i] == -INFINITY) //prevent re-loop
                    return RECTANGULAR_LSAP_INVALID;
            }
        }

        // initialize variables
        std::vector<double> u(nr, 0);
        std::vector<double> v(nc, 0);
        std::vector<double> shortestPathCosts(nc);
        std::vector<intptr_t> path(nc, -1);
        std::vector<intptr_t> col4row(nr, -1);
        std::vector<intptr_t> row4col(nc, -1);
        std::vector<bool> SR(nr);
        std::vector<bool> SC(nc);
        std::vector<intptr_t> remaining(nc);

        // iteratively build the solution
        for (intptr_t curRow = 0; curRow < nr; curRow++) {

            double minVal;
            intptr_t sink = augmenting_path(nc, cost, u, v, path, row4col,
                shortestPathCosts, curRow, SR, SC,
                remaining, &minVal);
            if (sink < 0)
                return RECTANGULAR_LSAP_INFEASIBLE;

            // update dual variables
            u[curRow] += minVal;
            for (intptr_t i = 0; i < nr; i++)
                if (SR[i] && i != curRow)
                    u[i] += minVal - shortestPathCosts[col4row[i]];
            for (intptr_t j = 0; j < nc; j++)
                if (SC[j])
                    v[j] -= minVal - shortestPathCosts[j];

            // augment previous solution
            intptr_t j = sink;
            for (;;)
            {
                intptr_t i = path[j];
                row4col[j] = i;
                std::swap(col4row[i], j);
                if (i == curRow)
                    break;
            }
        }

        /*col = std::vector <int64_t>(nr);
        row = std::vector<int64_t>(nr);*/

        auto tens_col = torch::zeros({ 1,nr }, torch::kInt64);
        auto tens_row = torch::zeros({ 1,nr }, torch::kInt64);
        if (transpose) {
            intptr_t i = 0;
            for (auto v : argsort_iter(col4row)) {
                tens_col[0][i] = col4row[v];
                tens_row[0][i] = v;
                /*col[i] = col4row[v];
                row[i] = v;*/
                i++;
            }
        }
        else {
            for (intptr_t i = 0; i < nr; i++) {
                tens_col[0][i] = i;
                tens_row[0][i] = col4row[i];
                /*col[i] = i;
                row[i] = col4row[i];*/
            }
        }
        indices = std::make_tuple(tens_col, tens_row);
        return 0;
    }
}
#endif