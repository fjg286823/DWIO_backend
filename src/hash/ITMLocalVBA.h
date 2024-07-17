#pragma once

#include "MemoryBlock.h"

//如果不交换这个一整个都能去掉
namespace DWIO {
    /** \brief
    Stores the actual voxel content that is referred to by a
    ITMLib::ITMHashTable.
    */
    template<class TVoxel>
    class ITMLocalVBA {
    private:
        DWIO::MemoryBlock<TVoxel> *voxelBlocks;
        DWIO::MemoryBlock<int> *allocationList;

        MemoryDeviceType memoryType;

    public:
        inline TVoxel *GetVoxelBlocks(void) { return voxelBlocks->GetData(memoryType); }

        inline const TVoxel *GetVoxelBlocks(void) const { return voxelBlocks->GetData(memoryType); }

        int *GetAllocationList(void) { return allocationList->GetData(memoryType); }

        int lastFreeBlockId;

        int allocatedSize;


        ITMLocalVBA(MemoryDeviceType memoryType, int noBlocks, int blockSize) {
            this->memoryType = memoryType;

            allocatedSize = noBlocks * blockSize;

            voxelBlocks = new DWIO::MemoryBlock<TVoxel>(allocatedSize, memoryType);
            allocationList = new DWIO::MemoryBlock<int>(noBlocks, memoryType);
        }

        ~ITMLocalVBA(void) {
            delete voxelBlocks;//这岂不是表明这就是在cpu上的内存中
            delete allocationList;
        }

        // Suppress the default copy constructor and assignment operator
        ITMLocalVBA(const ITMLocalVBA &);

        ITMLocalVBA &operator=(const ITMLocalVBA &);
    };
}
