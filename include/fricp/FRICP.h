#ifndef FRICP_H
#define FRICP_H
#include "ICP.h"
#include "AndersonAcceleration.h"
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "median.h"
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>


namespace DWIO{

template<int N>
class FRICP
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N+1, N+1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
    double test_total_construct_time=.0;
    double test_total_solve_time=.0;
    int test_total_iters=0;

    FRICP(){};
    ~FRICP(){};

private:
    AffineMatrixN LogMatrix(const AffineMatrixN& T)
    {
        Eigen::RealSchur<AffineMatrixN> schur(T);
        AffineMatrixN U = schur.matrixU();
        AffineMatrixN R = schur.matrixT();
        std::vector<bool> selected(N, true);
        MatrixNN mat_B = MatrixNN::Zero(N, N);
        MatrixNN mat_V = MatrixNN::Identity(N, N);

        for (int i = 0; i < N; i++)
        {
            if (selected[i] && fabs(R(i, i) - 1)> SAME_THRESHOLD)
            {
                int pair_second = -1;
                for (int j = i + 1; j <N; j++)
                {
                    if (fabs(R(j, j) - R(i, i)) < SAME_THRESHOLD)
                    {
                        pair_second = j;
                        selected[j] = false;
                        break;
                    }
                }
                if (pair_second > 0)
                {
                    selected[i] = false;
                    R(i, i) = R(i, i) < -1 ? -1 : R(i, i);
                    double theta = acos(R(i, i));
                    if (R(i, pair_second) < 0)
                    {
                        theta = -theta;
                    }
                    mat_B(i, pair_second) += theta;
                    mat_B(pair_second, i) += -theta;
                    mat_V(i, pair_second) += -theta / 2;
                    mat_V(pair_second, i) += theta / 2;
                    double coeff = 1 - (theta * R(i, pair_second)) / (2 * (1 - R(i, i)));
                    mat_V(i, i) += -coeff;
                    mat_V(pair_second, pair_second) += -coeff;
                }
            }
        }

        AffineMatrixN LogTrim = AffineMatrixN::Zero();
        LogTrim.block(0, 0, N, N) = mat_B;
        LogTrim.block(0, N, N, 1) = mat_V * R.block(0, N, N, 1);
        AffineMatrixN res = U * LogTrim * U.transpose();
        return res;
    }


    double FindKnearestMed(const KDtree& kdtree,
                           const MatrixNX& X, int nk)
    {
        Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
        for(int i = 0; i<X.cols(); i++)
        {
            int* id = new int[nk];
            double *dist = new double[nk];
            kdtree.query(X.col(i).data(), nk, id, dist);
            Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
            igl::median(k_dist.tail(nk-1), X_nearest[i]);
            delete[]id;
            delete[]dist;
        }
        double med;
        igl::median(X_nearest, med);
        return sqrt(med);
    }

    //X 源点云，Y搜索出的最近点，W每个点对应的权重！
    template <typename Derived1, typename Derived2, typename Derived3>
    AffineNd point_to_point(Eigen::MatrixBase<Derived1>& X,
                            Eigen::MatrixBase<Derived2>& Y,
                            const Eigen::MatrixBase<Derived3>& w) {
        int dim = X.rows();
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// De-mean
        Eigen::VectorXd X_mean(dim), Y_mean(dim);
        for (int i = 0; i<dim; ++i) {
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array()*w_normalized.transpose().array()).sum();
        }
        X.colwise() -= X_mean;
        Y.colwise() -= Y_mean;

        /// Compute transformation。这里完全就是SVD分解求R和T,但是R求解跟slam十四讲不太一样，R = U*V^T,这里是V*U^T，因为公式推导的不同！
        AffineNd transformation;
        MatrixXX sigma = X * w_normalized.asDiagonal() * Y.transpose();
        Eigen::JacobiSVD<MatrixXX> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            VectorN S = VectorN::Ones(dim); S(dim-1) = -1.0;
            transformation.linear() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
        }
        else {
            transformation.linear() = svd.matrixV()*svd.matrixU().transpose();
        }
        transformation.translation() = Y_mean - transformation.linear()*X_mean;
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += Y_mean;
        /// Return transformation
        return transformation;
    }



public:
        //X为源点云，Y为目标点云！
    double point_to_point(MatrixNX& X, MatrixNX& Y, VectorN& source_mean,
                        VectorN& target_mean, ICP::Parameters& par){
        /// Build kd-tree
        KDtree kdtree(Y);
        /// Buffers
        MatrixNX Q = MatrixNX::Zero(N, X.cols());//3行点数目列
        VectorX W = VectorX::Zero(X.cols());
        AffineNd T; //要求解的位姿！
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        MatrixXX To1 = T.matrix();
        MatrixXX To2 = T.matrix(); //迭代的中间值！
        int nPoints = X.cols();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        AffineNd SVD_T = T;
        double energy = .0, last_energy = std::numeric_limits<double>::max();

        //ground truth point clouds
        MatrixNX X_gt = X;
        if(par.has_groundtruth)
        {
            VectorN temp_trans = par.gt_trans.col(N).head(N);
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, N, N) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        //output para
        std::string file_out = par.out_path;
        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse = 0.0;

        // dynamic welsch paras
        double nu1 = 1, nu2 = 1;
        double begin_init = omp_get_wtime();

        //Find initial closest point
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) { //找到最近点计算残差
            VectorN cur_p = T * X.col(i);
            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
            W[i] = (cur_p - Q.col(i)).norm();
        }
        if(par.f == ICP::WELSCH)
        {
            //dynamic welsch, calc k-nearest points with itself;计算7个近邻的平均距离平方根，再乘上一个0.19245系数
            nu2 = par.nu_end_k * FindKnearestMed(kdtree, Y, 7);
            double med1;
            igl::median(W, med1); //得到距离的中位数！
            nu1 = par.nu_begin_k * med1; //3乘上这个值
            nu1 = nu1>nu2? nu1:nu2; //得到两者中较大的残差
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        //AA init
        accelerator_.init(par.anderson_m, (N + 1) * (N + 1), LogMatrix(T.matrix()).data());

        begin_time = omp_get_wtime();
        bool stop1 = false;
        while(!stop1)
        {
            /// run ICP
            int icp = 0;
            for (; icp<par.max_icp; ++icp) //默认迭代100次
            {
                bool accept_aa = false;
                energy = get_energy(par.f, W, nu1); //计算一个WELSCH残差！
                if (par.use_AA)
                {
                    if (energy < last_energy) {
                        last_energy = energy;
                        accept_aa = true;
                    }
                    else{
                        accelerator_.replace(LogMatrix(SVD_T.matrix()).data());
                        //Re-find the closest point
#pragma omp parallel for
                        for (int i = 0; i<nPoints; ++i) {
                            VectorN cur_p = SVD_T * X.col(i);
                            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                            W[i] = (cur_p - Q.col(i)).norm();
                        }
                        last_energy = get_energy(par.f, W, nu1);
                    }
                }
                else
                    last_energy = energy;

                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                if(par.has_groundtruth)
                {
                    gt_mse = (T*X - X_gt).squaredNorm()/nPoints;
                }

                // save results
                energys.push_back(last_energy);
                times.push_back(run_time);
                gt_mses.push_back(gt_mse);

                if (par.print_energy)
                    std::cout << "icp iter = " << icp << ", Energy = " << last_energy
                             << ", time = " << run_time << std::endl;

                robust_weight(par.f, W, nu1); //为每个点计算一个权重，存放在W中
                // Rotation and translation update，使用SVD分解得到变换矩阵！
                T = point_to_point(X, Q, W);

                //Anderson Acc
                SVD_T = T;
                if (par.use_AA)
                {
                    AffineMatrixN Trans = (Eigen::Map<const AffineMatrixN>(accelerator_.compute(LogMatrix(T.matrix()).data()).data(), N+1, N+1)).exp();
                    T.linear() = Trans.block(0,0,N,N);
                    T.translation() = Trans.block(0,N,N,1);
                }

                // Find closest point
#pragma omp parallel for
                for (int i = 0; i<nPoints; ++i) { //重新为每个点计算对应距离
                    VectorN cur_p = T * X.col(i) ;
                    Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                    W[i] = (cur_p - Q.col(i)).norm();
                }
                /// Stopping criteria
                double stop2 = (T.matrix() - To2).norm(); //优化后的更新量大小！
                To2 = T.matrix();
                if(stop2 < par.stop)//1e-5
                {
                    break;
                }
            }
            if(par.f!= ICP::WELSCH)
                stop1 = true;
            else
            {
                stop1 = fabs(nu1 - nu2)<SAME_THRESHOLD? true: false;
                nu1 = nu1*par.nu_alpha > nu2? nu1*par.nu_alpha : nu2; //nu_alpha =0.5
                if(par.use_AA)
                {
                    accelerator_.reset(LogMatrix(T.matrix()).data());
                    last_energy = std::numeric_limits<double>::max();
                }
            }
        }

        ///calc convergence energy
        last_energy = get_energy(par.f, W, nu1);
        X = T * X;
        gt_mse = (X-X_gt).squaredNorm()/nPoints;
        T.translation() += - T.rotation() * source_mean + target_mean;
        X.colwise() += target_mean;

        ///save convergence result
        par.convergence_energy = last_energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();
        //W拿过来作为参数判断是否匹配成功！
        int counts = 0;
        double fitness = 0.0;
        
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) { //重新为每个点计算对应距离
            VectorN cur_p = T * X.col(i) ;
            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
            W[i] = (cur_p - Q.col(i)).norm();
            if(W[i]<0.1){
                counts++;
                fitness += W[i];
            }
        }
        fitness = fitness / counts;

        return fitness;
        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            //output time and energy
            out_res.precision(16);
            for (int i = 0; i<times.size(); i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }

};

}

#endif
