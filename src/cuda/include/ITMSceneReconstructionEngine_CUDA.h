// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "common.h"
#include "../../hash/ITMVoxelTypes.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMRenderState_VH.h"
#include "../../hash/ITMScene.h"
#include "../../hash/ITMGlobalCache.h"

namespace DWIO {

    template<class TVoxel>
    class ITMSceneReconstructionEngine_CUDA {
    private:
        void *allocationTempData_device;
        void *allocationTempData_host;
        unsigned char *entriesAllocType_device;
        Vector4s *blockCoords_device;

    public:
        void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

        void AllocateScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const cv::cuda::GpuMat &depth_map, const Eigen::Matrix4d &pose,
                           ITMRenderState_VH *renderState_vh, Vector4f camera_intrinsic, float truncation_distance, int *csm_size,
                           bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

        void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const cv::cuda::GpuMat &depth_map, const Eigen::Matrix4d &pose,
                                    ITMRenderState_VH *renderState_vh, Vector4f camera_intrinsic, float truncation_distance,int* csm_size);

        void IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const cv::cuda::GpuMat &depth_map,
                                const cv::cuda::GpuMat &rgb, const Eigen::Matrix4d &pose_inv,
                                ITMRenderState_VH *renderState_vh, Vector4f camera_intrinsic, float truncation_distance);

        void computeMapBlock(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, Vector3s *blockPos_device);

        void SwapAllBlocks(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState_vh, int *NeedToSwapIn);

        bool showHashTableAndVoxelAllocCondition(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,ITMRenderState_VH *renderState_vh);
        
        ITMSceneReconstructionEngine_CUDA(void);

        ~ITMSceneReconstructionEngine_CUDA(void);
    };

    template
    class ITMSceneReconstructionEngine_CUDA<ITMVoxel_d>;

}
