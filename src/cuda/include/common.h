
#pragma once

#include "data.h"

#define DIVSHORTMAX 0.0000305185f
#define SHORTMAX 32767

typedef pcl::PointXYZI PointType;

//#define SUBMAP
struct Vertex
{
    float3 p;
    float3 c;
};

#define MAX_WEIGHT 128
#define IS_EQUAL3(a,b) (((a).x == (b).x()) && ((a).y == (b).y()) && ((a).z == (b).z()))
using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;
using cv::cuda::GpuMat;

using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Vector2i = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;
using Vector3i = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;
using Vector4i = Eigen::Matrix<int, 4, 1, Eigen::DontAlign>;
using Vector2f = Eigen::Matrix<float, 2, 1, Eigen::DontAlign>;
using Vector4f = Eigen::Matrix<float, 4, 1, Eigen::DontAlign>;
using Vector3f = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Vector3s = Eigen::Matrix<short, 3, 1, Eigen::DontAlign>;
using Vector4s = Eigen::Matrix<short, 4, 1, Eigen::DontAlign>;


#ifndef ROUND
#define ROUND(x) ((x < 0) ? (x - 0.5f) : (x + 0.5f))
#endif


