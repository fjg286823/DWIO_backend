#ifndef AA_H
#define AA_H

#include <fstream>
#include <vector>
#include "data.h"

class AA
{

public:
    AA()
        : m_(-1), dim_(-1), iter_(-1), col_idx_(-1) {}

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compute(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> g)
    {
        assert(iter_ >= 0);
        assert(g.rows() == dim_);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G = g;
        current_F_ = G - current_u_;//当前值与上一次迭代的差

        if (iter_ == 0)
        {
            prev_dF_.col(0) = -current_F_;
            prev_dG_.col(0) = -G;
            current_u_ = G;//其实存放上一个的状态
        }
        else
        {
            prev_dF_.col(col_idx_) += current_F_;
            prev_dG_.col(col_idx_) += G;

            double eps = 1e-14;
            double scale = std::max(eps, prev_dF_.col(col_idx_).norm());
            dF_scale_(col_idx_) = scale;
            prev_dF_.col(col_idx_) /= scale;

            int m_k = std::min(m_, iter_);

            if (m_k == 1)
            {
                theta_(0) = 0;
                double dF_sqrnorm = prev_dF_.col(col_idx_).squaredNorm();
                M_(0, 0) = dF_sqrnorm;
                double dF_norm = std::sqrt(dF_sqrnorm);

                if (dF_norm > eps)
                {
                    theta_(0) = (prev_dF_.col(col_idx_) / dF_norm).dot(current_F_ / dF_norm);
                }
            }
            else
            {
                // Update the normal equation matrix, for the column and row corresponding to the new dF column
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_inner_prod = (prev_dF_.col(col_idx_).transpose() * prev_dF_.block(0, 0, dim_, m_k)).transpose();
                M_.block(col_idx_, 0, 1, m_k) = new_inner_prod.transpose();
                M_.block(0, col_idx_, m_k, 1) = new_inner_prod;

                // Solve normal equation
                cod_.compute(M_.block(0, 0, m_k, m_k));
                theta_.head(m_k) = cod_.solve(prev_dF_.block(0, 0, dim_, m_k).transpose() * current_F_);
            }

            // Use rescaled theata to compute new u
            current_u_ = G - prev_dG_.block(0, 0, dim_, m_k) * ((theta_.head(m_k).array() / dF_scale_.head(m_k).array()).matrix());
            col_idx_ = (col_idx_ + 1) % m_;
            prev_dF_.col(col_idx_) = -current_F_;
            prev_dG_.col(col_idx_) = -G;
        }
        iter_++;
        return current_u_;
    }

    void init(int m, int d, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> u0)
	{
		assert(m > 0);
		m_ = m;//迭代20次
		dim_ = d;//6个变量
		current_u_.resize(d);
		current_F_.resize(d);
		prev_dG_.resize(d, m);
		prev_dF_.resize(d, m);
		M_.resize(m, m);
		theta_.resize(m);
		dF_scale_.resize(m);
		current_u_ = u0;
		iter_ = 0;
		col_idx_ = 0;
	}

private:
    Eigen::VectorXd current_u_;
    Eigen::VectorXd current_F_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> prev_dG_;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> prev_dF_;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M_;    
    Eigen::VectorXd theta_;
    Eigen::VectorXd dF_scale_;

    Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> cod_;

    int m_;		// Number of previous iterates used for Andreson Acceleration
	int dim_;	// Dimension of variables
	int iter_;	// Iteration count since initialization
	int col_idx_;	// Index for history matrix column to store the next value
	int m_k_;

};

#endif 