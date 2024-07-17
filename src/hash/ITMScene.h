
#pragma once

#include "ITMLocalVBA.h"
#include "ITMGlobalCache.h"
#include <Eigen/Dense>

namespace DWIO
{
	/** \brief
	Represents the 3D world model as a hash of small voxel
	blocks
	*/
	template<class TVoxel, class TIndex>
	class ITMScene
	{
	public:

		/** Hash table to reference the 8x8x8 blocks */
		TIndex index;//hash表

		/** Current local content of the 8x8x8 voxel blocks -- stored host or device */
		ITMLocalVBA<TVoxel> localVBA;

		/** Global content of the 8x8x8 voxel blocks -- stored on host only */
		ITMGlobalCache<TVoxel> *globalCache;


		int maxW = 128;
		float voxel_resolution = 10.0f;
		float viewFrustum_min, viewFrustum_max;

		Eigen::Matrix4d initial_pose;

		ITMScene( bool _useSwapping, MemoryDeviceType _memoryType)
			: index(_memoryType), localVBA(_memoryType, index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
		{
			if (_useSwapping) globalCache = new ITMGlobalCache<TVoxel>();
			else globalCache = NULL;
			viewFrustum_min = 600.0f;
			viewFrustum_max = 3000.0f;
			initial_pose.setIdentity();
		}

		~ITMScene(void)
		{
			if (globalCache != NULL) delete globalCache;
		}

		// Suppress the default copy constructor and assignment operator
		ITMScene(const ITMScene&);
		ITMScene& operator=(const ITMScene&);
	};
}
