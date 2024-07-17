// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../hash/ITMRenderState_VH.h"
#include "../../hash/ITMScene.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include "common.h"

namespace DWIO {
    template<class TVoxel>
    class ITMSwappingEngine_CUDA {
    private:
        int *noNeededEntries_device, *noAllocatedVoxelEntries_device;
        int *entriesToClean_device;
        int *moveInEntries_device;
        uchar *blockEmptyVerify;

        int LoadFromGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

    public:
        void IntegrateGlobalIntoLocal(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState, bool updateFlag);

        void SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState);

        void TransferGlobalMap(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);


        void MoveVoxelToGlobalMemorey(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState);

        ITMSwappingEngine_CUDA(void);

        ~ITMSwappingEngine_CUDA(void);
    };

    template
    class ITMSwappingEngine_CUDA<ITMVoxel_d>;
}

