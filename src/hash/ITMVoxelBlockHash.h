
#pragma once

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>


#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "MemoryBlock.h"

#define SDF_BLOCK_SIZE 8                // SDF block size
#define SDF_BLOCK_SIZE3 512                // SDF_BLOCK_SIZE3 = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE

#define SDF_LOCAL_BLOCK_NUM 0x20000        // Number of locally stored blocks, 0x60000

#ifdef SUBMAP
    #define SDF_BUCKET_NUM 0x2000           
    #define SDF_HASH_MASK 0x1fff            
    #define SDF_EXCESS_LIST_SIZE 0x2000    
#else
    #define SDF_BUCKET_NUM 0x40000           // Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM, SDF_HASH_MASK = SDF_BUCKET_NUM - 1
    #define SDF_HASH_MASK 0x3ffff             // Used for get hashing value of the bucket index,  SDF_HASH_MASK = SDF_BUCKET_NUM - 1
    #define SDF_EXCESS_LIST_SIZE 0x8000      // Size of excess list, used to handle collisions. Also max offset (unsigned short) value.
#endif
//for global map
#define MAP_BUCKET_NUM 0x100000           
#define MAP_HASH_MASK 0xfffff           
#define MAP_EXCESS_LIST_SIZE 0x10000  


#define SDF_TRANSFER_BLOCK_NUM 0x2000    // Maximum number of blocks transfered in one swap operation

/** \brief
	A single entry in the hash table.
*/
struct ITMHashEntry
{
    /** Position of the corner of the 8x8x8 volume, that identifies the entry. */
    int3 pos;
    /** Offset in the excess list. */
    int offset;
    /** Pointer to the voxel block array.
        - >= 0 identifies an actual allocated entry in the voxel block array
        - -1 identifies an entry that has been removed (swapped out)
        - <-1 identifies an unallocated block
    */
    int ptr;
};

namespace DWIO
{
    /** \brief
    This is the central class for the voxel block hash
    implementation. It contains all the data needed on the CPU
    and a pointer to the data structure on the GPU.
    */
    class ITMVoxelBlockHash {
    public:
        typedef ITMHashEntry IndexData;

        struct IndexCache {
            int3 blockPos;
            int blockPtr;
            __device__ IndexCache(void) {
                blockPos.x = 0x7fffffff;
                blockPos.y = 0x7fffffff;
                blockPos.z = 0x7fffffff;
                blockPtr = -1;
            }
        };

        /** Maximum number of total entries. */
        static const int noTotalEntries = SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE;
        static const int voxelBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

    private:
        int lastFreeExcessListId;

        /** The actual data in the hash table. */
        DWIO::MemoryBlock<ITMHashEntry> *hashEntries;
        DWIO::MemoryBlock<ITMHashEntry> *hashEntries_cpu;

        /** Identifies which entries of the overflow
        list are allocated. This is used if too
        many hash collisions caused the buckets to
        overflow.
        */
        DWIO::MemoryBlock<int> *excessAllocationList;

        MemoryDeviceType memoryType;

    public:
        ITMVoxelBlockHash(MemoryDeviceType memoryType) {
            this->memoryType = memoryType;

            hashEntries = new DWIO::MemoryBlock<ITMHashEntry>(noTotalEntries, memoryType);
            hashEntries_cpu = new DWIO::MemoryBlock<ITMHashEntry>(noTotalEntries, MEMORYDEVICE_CPU);
            excessAllocationList = new DWIO::MemoryBlock<int>(SDF_EXCESS_LIST_SIZE, memoryType);
        }

        ~ITMVoxelBlockHash(void) {
            delete hashEntries;
            delete excessAllocationList;
        }

        /** Get the list of actual entries in the hash table. */
        const ITMHashEntry *GetEntries(void) const { return hashEntries->GetData(memoryType); }

        ITMHashEntry *GetEntries(void) { return hashEntries->GetData(memoryType); }

        ITMHashEntry *GetEntriesCpu(void) { 
            return hashEntries_cpu->GetData(MEMORYDEVICE_CPU); 
        }

        void GpuToCpuHashData(void)
        {
            cudaMemcpy(hashEntries_cpu->GetData(MEMORYDEVICE_CPU),hashEntries->GetData(memoryType),sizeof(ITMHashEntry)*noTotalEntries,cudaMemcpyDeviceToHost);
        }
        
        void CopyHashEntries(DWIO::MemoryBlock<ITMHashEntry>* submap_hashEntries)
        {
            cudaMemcpy(submap_hashEntries->GetData(MEMORYDEVICE_CPU),hashEntries->GetData(memoryType),sizeof(ITMHashEntry)*noTotalEntries,cudaMemcpyDeviceToHost);
        }

        const IndexData *getIndexData(void) const { return hashEntries->GetData(memoryType); }

        IndexData *getIndexData(void) { return hashEntries->GetData(memoryType); }

        /** Get the list that identifies which entries of the
        overflow list are allocated. This is used if too
        many hash collisions caused the buckets to overflow.
        */
        const int *GetExcessAllocationList(void) const { return excessAllocationList->GetData(memoryType); }

        int *GetExcessAllocationList(void) { return excessAllocationList->GetData(memoryType); }

        int GetLastFreeExcessListId(void) { return lastFreeExcessListId; }

        void SetLastFreeExcessListId(int lastFreeExcessListId) { this->lastFreeExcessListId = lastFreeExcessListId; }


        /** Maximum number of total entries. */
        int getNumAllocatedVoxelBlocks(void) { return SDF_LOCAL_BLOCK_NUM; }

        int getVoxelBlockSize(void) { return SDF_BLOCK_SIZE3; }

        // Suppress the default copy constructor and assignment operator
        ITMVoxelBlockHash(const ITMVoxelBlockHash &);

        ITMVoxelBlockHash &operator=(const ITMVoxelBlockHash &);
    };
}