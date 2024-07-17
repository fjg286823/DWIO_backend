#pragma once

#include "../src/hash/ITMVoxelBlockHash.h"
#include "../src/hash/ITMVoxelTypes.h"
#include "../src/hash/ITMMesh.h"
#include "../src/hash/ITMScene.h"
#include "../src/cuda/include/common.h"
namespace DWIO
{
	template<class TVoxel>
	class ITMMeshingEngine_CPU
	{
	public:
		void MeshScene(ITMMesh *mesh, ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

		ITMMeshingEngine_CPU(void) { }
		~ITMMeshingEngine_CPU(void) { }
	};

    template
    class ITMMeshingEngine_CPU<ITMVoxel_d>;
}
