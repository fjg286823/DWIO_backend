//
// Created by baominjie on 23-7-6.
//

#ifndef DWIO_UTILITY_H
#define DWIO_UTILITY_H

#include "data.h"

namespace DWIO {
    template<typename T>
    void MyCV2Eigen(const cv::Mat_<float> &src, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &dst) {
        int dst_row = (int) dst.rows();
        for (int row = 0; row < src.rows; ++row) {
            int i = row / dst_row;
            int j = row - i * dst_row;
            dst(i, j) = src.ptr<float>(row)[0];
        }
    }

    template<typename T>
    void PrintMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m) {
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                printf("%e ", m(i, j));
            }
            std::cout << std::endl;
        }
    }

    template<typename T>
    void PrintRotationMatrix(const Eigen::Matrix<T, 3, 3> &m) {
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                printf("%-.16f ", m(i, j));
            }
            std::cout << std::endl;
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> NormalizeRotation(const Eigen::Matrix<T, 3, 3> &R) {
        Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> Skew(const Eigen::Matrix<T, 3, 1> &w) {
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0;
        return W;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> Rodrigues(const Eigen::Matrix<T, 3, 1> &rotation_axis, const T rotation_angle) {
        return cos(rotation_angle) * Eigen::Matrix<T, 3, 3>::Identity() +
               (1 - cos(rotation_angle)) * rotation_axis * rotation_axis.transpose() +
               sin(rotation_angle) * Skew(rotation_axis);
    }

    template<typename T>
    Eigen::Matrix<T, 6, 6> AdSkew(const Eigen::Matrix<T, 6, 1> &xi) {
        const Eigen::Matrix<T, 3, 1> rou = xi.block(0, 0, 3, 1);
        const Eigen::Matrix<T, 3, 1> phi = xi.block(3, 0, 3, 1);
        Eigen::Matrix<T, 6, 6> ad_skew = Eigen::Matrix<T, 6, 6>::Zero();
        ad_skew.block(0, 0, 3, 3) = Skew(phi);
        ad_skew.block(3, 3, 3, 3) = Skew(phi);
        ad_skew.block(0, 3, 3, 3) = Skew(rou);
        return ad_skew;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSE3QLeft(const Eigen::Matrix<T, 6, 1> &xi) {
        const T d = xi.block(3, 0, 3, 1).norm();
        const T d_2 = d * d;
        const T d_3 = d_2 * d;
        const T d_4 = d_3 * d;
        const T d_5 = d_4 * d;
        const T c_d = cos(d);
        const T s_d = sin(d);

        const Eigen::Matrix<T, 3, 1> rou = xi.block(0, 0, 3, 1);
        const Eigen::Matrix<T, 3, 1> phi = xi.block(3, 0, 3, 1);

        Eigen::Matrix<T, 3, 3> rou_skew = Skew(rou);
        Eigen::Matrix<T, 3, 3> phi_skew = Skew(phi);

        Eigen::Matrix<T, 3, 3> se3_q_left = Eigen::Matrix<T, 3, 3>::Zero();
        se3_q_left = 0.5 * rou_skew + (d - s_d) / d_3 * (phi_skew * rou_skew + rou_skew * phi_skew + phi_skew * rou_skew * phi_skew) +
                     (d_2 + 2 * c_d - 2) / (2 * d_4) * (phi_skew * phi_skew * rou_skew + rou_skew * phi_skew * phi_skew -
                                                        3 * phi_skew * rou_skew * phi_skew) +
                     (2 * d - 3 * s_d + d * c_d) / (2 * d_5) * (phi_skew * rou_skew * phi_skew * phi_skew +
                                                                phi_skew * phi_skew * rou_skew * phi_skew);
        return se3_q_left;
    }

    template<typename T>
    Eigen::Matrix<T, 6, 6> ComputeSE3JLeft(const Eigen::Matrix<T, 6, 1> &xi) {
        Eigen::Matrix<T, 6, 6> ad_skew_1 = AdSkew(xi);
        Eigen::Matrix<T, 6, 6> ad_skew_2 = ad_skew_1 * ad_skew_1;
        Eigen::Matrix<T, 6, 6> ad_skew_3 = ad_skew_2 * ad_skew_1;
        Eigen::Matrix<T, 6, 6> ad_skew_4 = ad_skew_3 * ad_skew_1;

        const T d = xi.block(3, 0, 3, 1).norm();
        const T d_2 = d * d;
        const T d_3 = d_2 * d;
        const T d_4 = d_3 * d;
        const T d_5 = d_4 * d;
        const T c_d = cos(d);
        const T s_d = sin(d);

        Eigen::Matrix<T, 6, 6> se3_j_left = Eigen::Matrix<T, 6, 6>::Identity() +
                                            ((4 - d * s_d - 4 * c_d) / (2 * d_2)) * ad_skew_1 +
                                            ((4 * d - 5 * s_d + d * c_d) / (2 * d_3)) * ad_skew_2 +
                                            ((2 - d * s_d - 2 * c_d) / (2 * d_4)) * ad_skew_3 +
                                            ((2 * d - 3 * s_d + d * c_d) / (2 * d_5)) * ad_skew_4;
        return se3_j_left;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ExpSO3(const T x, const T y, const T z) {
        const T d2 = x * x + y * y + z * z;
        const T d = sqrt(d2);
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5) {
            Eigen::Matrix<T, 3, 3> res = Eigen::Matrix<T, 3, 3>::Identity() + W + 0.5 * W * W;
            return NormalizeRotation(res);
        } else {
            Eigen::Matrix<T, 3, 3> res = Eigen::Matrix<T, 3, 3>::Identity() + W * sin(d) / d + W * W * (1.0 - cos(d)) / d2;
            return NormalizeRotation(res);
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ExpSO3(const Eigen::Matrix<T, 3, 1> &w) {
        return ExpSO3(w[0], w[1], w[2]);
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3InverseJRight(const T x, const T y, const T z) {
        const T d2 = x * x + y * y + z * z;
        const T d = sqrt(d2);
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        } else {
            return Eigen::Matrix<T, 3, 3>::Identity() + 0.5 * W + (1 / d2 + (1 + cos(d)) / sin(d) / (2 * d)) * W * W;
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3InverseJRight(const Eigen::Matrix<T, 3, 1> &v) {
        return ComputeSO3InverseJRight(v(0), v(1), v(2));
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3InverseJLeft(const T x, const T y, const T z) {
        const T d2 = x * x + y * y + z * z;
        const T d = sqrt(d2);
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        } else {
            return Eigen::Matrix<T, 3, 3>::Identity() - 0.5 * W + (1 / d2 - cos(d / 2) / sin(d / 2) / (2 * d)) * W * W;
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3InverseJLeft(const Eigen::Matrix<T, 3, 1> &v) {
        return ComputeSO3InverseJLeft(v(0), v(1), v(2));
    }

    template<typename T>
    Eigen::Matrix<T, 3, 1> LogSO3(const Eigen::Matrix<T, 3, 3> &R) {
        Eigen::Matrix<T, 3, 1> phi = Eigen::Matrix<T, 3, 1>::Zero();
        const T tr = R(0, 0) + R(1, 1) + R(2, 2);
        Eigen::Matrix<T, 3, 1> w;
        w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2, (R(1, 0) - R(0, 1)) / 2;
        const T cos_theta = (tr - 1.0) * 0.5;
        if (cos_theta > 1 || cos_theta < -1)
            phi = w;
        const T theta = acos(cos_theta);
        const T s = sin(theta);
        if (fabs(s) < 1e-5)
            phi = w;
        else
            phi = theta * w / s;
        return phi;
    }

    template<typename T>
    Eigen::Matrix<T, 6, 1> LogSE3(const Eigen::Matrix<T, 3, 3> &R, const Eigen::Matrix<T, 3, 1> &t) {
        Eigen::Matrix<T, 6, 1> xi = Eigen::Matrix<T, 6, 1>::Zero();
        Eigen::Matrix<T, 3, 1> rou = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> phi = Eigen::Matrix<T, 3, 1>::Zero();
        const T tr = R(0, 0) + R(1, 1) + R(2, 2);
        Eigen::Matrix<T, 3, 1> w;
        w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2, (R(1, 0) - R(0, 1)) / 2;
        const T cos_theta = (tr - 1.0) * 0.5;
        if (cos_theta > 1 || cos_theta < -1)
            phi = w;
        const T theta = acos(cos_theta);
        const T s = sin(theta);
        if (fabs(s) < 1e-5)
            phi = w;
        else
            phi = theta * w / s;
        rou = ComputeSO3InverseJLeft(phi) * t;
        xi.block(0, 0, 3, 1) = rou;
        xi.block(3, 0, 3, 1) = phi;
        return xi;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3JRight(const T x, const T y, const T z) {
        const T d2 = x * x + y * y + z * z;
        const T d = sqrt(d2);
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        } else {
            return Eigen::Matrix<T, 3, 3>::Identity() - W * (1.0 - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3JRight(const Eigen::Matrix<T, 3, 1> &v) {
        return ComputeSO3JRight(v(0), v(1), v(2));
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3JLeft(const T x, const T y, const T z) {
        const T d2 = x * x + y * y + z * z;
        const T d = sqrt(d2);
        Eigen::Matrix<T, 3, 3> W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        if (d < 1e-5) {
            return Eigen::Matrix<T, 3, 3>::Identity();
        } else {
            return Eigen::Matrix<T, 3, 3>::Identity() + W * (1.0 - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
        }
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> ComputeSO3JLeft(const Eigen::Matrix<T, 3, 1> &v) {
        return ComputeSO3JLeft(v(0), v(1), v(2));
    }

    template<typename T>
    Eigen::Matrix<T, 4, 4> InterpolationMainifold(const Eigen::Matrix<T, 4, 4> &pose_last,
                                                  const Eigen::Matrix<T, 4, 4> &pose_now,
                                                  const T &ratio) {
        Eigen::Matrix<T, 4, 4> pose_interpolation = Eigen::Matrix4d::Identity();
        Eigen::Matrix<T, 3, 1> translation_last = pose_last.block(0, 3, 3, 1);
        Eigen::Matrix<T, 3, 1> translation_now = pose_now.block(0, 3, 3, 1);
        pose_interpolation.block(0, 3, 3, 1) = (1 - ratio) * translation_last + ratio * translation_now;
        Eigen::Matrix<T, 3, 3> R_last = pose_last.block(0, 0, 3, 3);
        Eigen::Matrix<T, 3, 3> R_now = pose_now.block(0, 0, 3, 3);
        Eigen::Matrix<T, 3, 3> delta_R = R_now * R_last.transpose();
        Eigen::Matrix<T, 3, 1> delta_phi = ratio * LogSO3(delta_R);
        Eigen::Matrix<T, 3, 3> R_interpolation = ExpSO3(delta_phi) * R_last;
        pose_interpolation.block(0, 0, 3, 3) = R_interpolation;
        return pose_interpolation;
    }

}

#endif //DWIO_UTILITY_H
