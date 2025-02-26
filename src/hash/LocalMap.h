#pragma once

#include <map>
#include <Eigen/Dense>
#include "ITMRenderState_VH.h"
#include "ITMScene.h"
#include "ITMVoxelTypes.h"
#include "ITMVoxelBlockHash.h"





namespace DWIO {
	

	class LocalMap//局部子图结构，以及相应的约束！
	{
	public:
		ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene;//存放hash表和对应的体素
		ITMRenderState_VH *renderState;//用来渲染用的
		Eigen::Matrix4d estimatedGlobalPose;//子图的全局位置，这些位姿都是世界到子图的估计，
        int initial_nums;
        bool generate_newmap;

        LocalMap(Eigen::Matrix4d pose,float factor)
		{
            scene = new ITMScene<ITMVoxel_d, ITMVoxelBlockHash>(true, MEMORYDEVICE_CUDA,factor);
            renderState = new ITMRenderState_VH(scene->index.noTotalEntries, MEMORYDEVICE_CUDA);
            initial_nums = 1;
            generate_newmap = false;
            estimatedGlobalPose = pose;
        }
        ~LocalMap(void)
		{
			delete scene;
			delete renderState;
		}
	};
}

