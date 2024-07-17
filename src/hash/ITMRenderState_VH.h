// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdlib.h>


#include "ITMVoxelBlockHash.h"
#include "MemoryBlock.h"

namespace DWIO {
    /** \brief
        Stores the render state used by the SceneReconstruction
        and visualisation engines, as used by voxel hashing.
    */
    class ITMRenderState_VH {
    private:
        MemoryDeviceType memoryType;

        /** A list of "visible entries", that are currently
        being processed by the tracker.
        */
        DWIO::MemoryBlock<int> *visibleEntryIDs;

        /** A list of "visible entries", that are
        currently being processed by integration
        and tracker.
        */
        // 0：表示该体素块是空闲的，没有被分配，也没有被使用
        // 1：表示该体素块是活跃的，已经被分配，且在当前帧中被观察到
        // 2：应该是可见但在cpu上
        // 3：表示该体素块是删除的，已经被分配，但在之前的帧中被判定为不可见，需要被释放
        DWIO::MemoryBlock<uchar> *entriesVisibleType;
        DWIO::MemoryBlock<uchar> *emptyBlockEntries;//表明哪些是由射线投射得到的真正有值的block

    public:
        /** Number of entries in the live list. */
        int noVisibleEntries;

        ITMRenderState_VH(int noTotalEntries, MemoryDeviceType memoryType = MEMORYDEVICE_CPU) {
            this->memoryType = memoryType;

            visibleEntryIDs = new DWIO::MemoryBlock<int>(SDF_LOCAL_BLOCK_NUM, memoryType);
            entriesVisibleType = new DWIO::MemoryBlock<uchar>(noTotalEntries, memoryType);
            emptyBlockEntries = new DWIO::MemoryBlock<uchar>(noTotalEntries, memoryType);
            noVisibleEntries = 0;
        }

        ~ITMRenderState_VH() {
            delete visibleEntryIDs;
            delete entriesVisibleType;
            delete emptyBlockEntries;
        }

        int *GetVisibleEntryIDs(void) { return visibleEntryIDs->GetData(memoryType); }

        uchar *GetEntriesVisibleType(void) { return entriesVisibleType->GetData(memoryType); }

        uchar *GetEmptyBlockEntries(void) { return emptyBlockEntries->GetData(memoryType); }

    };
} 
