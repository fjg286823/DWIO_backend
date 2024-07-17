#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

/** \brief
    Stores the information of a single voxel in the volume定义了体素存放什么数据类型的结构体
*/
struct ITMVoxel_d {
    __host__ __device__ static int16_t TSDF_initialValue() { return 32767; }

    __host__ __device__ static float valueToFloat(int16_t x) { return (float) (x) / 32767.0f; };//这样是为了归一化
    __host__ __device__ static int16_t floatToValue(float x) { return (int16_t) (x * 32767.0f); };//这样是为了归一化
    /** Value of the truncated signed distance transformation. */
    int16_t tsdf;
    int16_t w_depth;
    uchar3 clr;
    int16_t w_color;

    __host__ __device__ ITMVoxel_d() {
        tsdf = TSDF_initialValue();
        w_depth = 0;
        clr.x = 0;
        clr.y = 0;
        clr.z = 0;
        w_color = 0;
    }
};