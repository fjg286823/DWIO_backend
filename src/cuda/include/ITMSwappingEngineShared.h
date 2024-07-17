// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#define CONSTPTR(x) x
#define DEVICEPTR(x) x

#include "common.h"
#include "../../hash/ITMRepresentationAccess.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include <opencv2/core/cuda.hpp>

template<class TVoxel>
__device__ inline void combineVoxelDepthInformation(const CONSTPTR(TVoxel) &src, DEVICEPTR(TVoxel) &dst, int maxW) {
    int newW = dst.w_depth;
    int oldW = src.w_depth;
    float newF = static_cast<float>(dst.tsdf);
    float oldF = static_cast<float>(src.tsdf);

    if (oldW == 0) return;

    newF = oldW * oldF + newW * newF;
    newW = oldW + newW;
    newF /= newW;
    newW = MIN(newW, maxW);

    dst.w_depth = newW;
    dst.tsdf = static_cast<int16_t>(newF);
}

template<class TVoxel>
__device__ inline void combineVoxelColorInformation(const CONSTPTR(TVoxel) &src, DEVICEPTR(TVoxel) &dst, int maxW) {
    int newW = dst.w_color;
    int oldW = src.w_color;
    Vector3f newC;
    newC(0, 0) = static_cast<float>(dst.clr.x) / 255.0f;
    newC(1, 0) = static_cast<float>(dst.clr.y) / 255.0f;
    newC(2, 0) = static_cast<float>(dst.clr.z) / 255.0f;
    Vector3f oldC;
    oldC(0, 0) = static_cast<float>(src.clr.x) / 255.0f;
    oldC(1, 0) = static_cast<float>(src.clr.y) / 255.0f;
    oldC(2, 0) = static_cast<float>(src.clr.z) / 255.0f;

    if (oldW == 0) return;

    newC = oldC * (float) oldW + newC * (float) newW;
    newW = oldW + newW;
    newC /= (float) newW;
    newW = MIN(newW, maxW);

    dst.clr.x = static_cast<uchar>(newC(0, 0) * 255.0f);
    dst.clr.y = static_cast<uchar>(newC(1, 0) * 255.0f);
    dst.clr.z = static_cast<uchar>(newC(2, 0) * 255.0f);
    dst.w_color = newW;

    // uchar3 oldC = src.clr;
	// uchar3 newC = dst.clr;
	// float3 res;
	// //这里的要改
	// res.x =  0.2 * static_cast<float>(newC.x) + 0.8 * static_cast<float>(oldC.x);
	// res.y =  0.2 * static_cast<float>(newC.y) + 0.8 * static_cast<float>(oldC.y);
	// res.z =  0.2 * static_cast<float>(newC.z) + 0.8 * static_cast<float>(oldC.z);

	// dst.clr.x = static_cast<uchar>(res.x);
	// dst.clr.y = static_cast<uchar>(res.y);
	// dst.clr.z = static_cast<uchar>(res.z);
}


template<bool hasColor, class TVoxel>
struct CombineVoxelInformation;

template<class TVoxel>
struct CombineVoxelInformation<false, TVoxel> {
    __device__ static void compute(const CONSTPTR(TVoxel) &src, DEVICEPTR(TVoxel) &dst, int maxW) {
        combineVoxelDepthInformation(src, dst, maxW);
    }
};

template<class TVoxel>
struct CombineVoxelInformation<true, TVoxel> {
    __device__ static void compute(const CONSTPTR(TVoxel) &src, DEVICEPTR(TVoxel) &dst, int maxW) {
        combineVoxelDepthInformation(src, dst, maxW);
        combineVoxelColorInformation(src, dst, maxW);
    }
};