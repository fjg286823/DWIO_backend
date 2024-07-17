// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include "../../hash/ITMMesh.h"
#include "../../hash/ITMScene.h"

#include "common.h"

namespace DWIO {

    template<class TVoxel>
    class ITMMeshingEngine_CUDA {
    private:
        unsigned int *noTriangles_device;
        Vector4s *visibleBlockGlobalPos_device;

    public:
        void MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMVoxelBlockHash> *scene,int& TotalTriangles);

        ITMMeshingEngine_CUDA(void);

        ~ITMMeshingEngine_CUDA(void);
    };

    template
    class ITMMeshingEngine_CUDA<ITMVoxel_d>;

}


