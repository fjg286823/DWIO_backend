#pragma once

#include "../src/hash/ITMVoxelBlockHash.h"
#include "../src/hash/ITMVoxelTypes.h"
#include "../src/hash/ITMMesh.h"
#include "../src/hash/ITMScene.h"
#include "../src/cuda/include/common.h"
#include "../src/hash/Submap.h"
namespace DWIO
{
	template<class TVoxel>
	class ITMMeshingEngine_CPU
	{
	public:
		void MeshScene(ITMMesh *mesh, ITMScene<TVoxel, ITMVoxelBlockHash> *scene);
		void MeshScene(ITMMesh *mesh, TVoxel* globalVBA , ITMHashEntry* hashTable,int noTotalEntries,float factor);
		void MeshScene(ITMMesh *mesh, std::map<int,DWIO::BlockData*>& blocks , ITMHashEntry* hashTable,int noTotalEntries,float factor,Eigen::Matrix4f Trans);
		void MeshScene_global(ITMMesh *mesh, std::map<uint32_t,DWIO::submap*>&submaps_,int noTotalEntries,float factor);

		ITMMeshingEngine_CPU(void) { }
		~ITMMeshingEngine_CPU(void) { }
	};

    template
    class ITMMeshingEngine_CPU<ITMVoxel_d>;
}
