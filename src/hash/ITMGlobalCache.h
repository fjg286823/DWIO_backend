#pragma once

#include <stdlib.h>
#include <stdio.h>

#include "ITMVoxelBlockHash.h"
#include "CUDADefines.h"//里面包含一个函数用于检查cuda函数调用是否正确

namespace DWIO {
    struct ITMHashSwapState {
        /** 0 - most recent data is on host, data not currently in active memory */
        /** 1 - data both on host and in active memory, information has not yet been combined */
        /** 2 - most recent data is in active memory, should save this data back to host at some point */
        uchar state;
    };


    //里面定义所有体素数据存放的数组，GPU和CPU的数据交换缓冲区和一些标志数组表明是否完成数据同步。
    template<class TVoxel>//TVoxel是体素数据
    class ITMGlobalCache {
    private:
        bool *hasStoredData;//表明哪些block被存贮了？
        TVoxel *storedVoxelBlocks;//也是一个缓冲区，存放在cpu上的所有体素的数据，这里的TVoxel应该是体素数据类型结构体
        ITMHashSwapState *swapStates_host, *swapStates_device;
        //表明数据是否放入对应设备（cpu or gpu）的缓冲区里去了，大小为缓冲区大小4096
        bool *hasSyncedData_host, *hasSyncedData_device;
        //cpu和gpu数据交换的缓冲区。前面是cpu上的缓冲区，后面是GPU的缓冲区
        TVoxel *syncedVoxelBlocks_host, *syncedVoxelBlocks_device;//需要加载全局缓冲区的数据，就将数据写入到其中

        int *neededEntryIDs_host, *neededEntryIDs_device;//可见的且swapstate=1的block索引
        int *moveInEntryIDs_host, *moveInEntryIDs_device;//大小为缓冲区大小，存放的是需要交换数据在hash表数组中的索引

        ITMHashSwapState *States_host, *States_device;// 1 represent block in cpu

    public:
        inline void SetStoredData(int address, TVoxel *data) {
            hasStoredData[address] = true;
            memcpy(storedVoxelBlocks + address * SDF_BLOCK_SIZE3, data, sizeof(TVoxel) * SDF_BLOCK_SIZE3);
        }

        inline bool HasStoredData(int address) const { return hasStoredData[address]; }

        inline TVoxel *GetStoredVoxelBlock(int address) { return storedVoxelBlocks + address * SDF_BLOCK_SIZE3; }
        
        TVoxel * GetVoxelData(){return storedVoxelBlocks;}

        bool *GetHasSyncedData(bool useGPU) const { return useGPU ? hasSyncedData_device : hasSyncedData_host; }

        TVoxel *GetSyncedVoxelBlocks(bool useGPU) const { return useGPU ? syncedVoxelBlocks_device : syncedVoxelBlocks_host; }

        ITMHashSwapState *GetSwapStates(bool useGPU) { return useGPU ? swapStates_device : swapStates_host; }

        ITMHashSwapState *GetStates(bool useGPU) { return useGPU ? States_device : States_host; }

        int *GetNeededEntryIDs(bool useGPU) { return useGPU ? neededEntryIDs_device : neededEntryIDs_host; }

        int *GetMoveInEntryIDs(bool useGPU) { return useGPU ? moveInEntryIDs_device : moveInEntryIDs_host; }

        void CopyVoxelBlocks(TVoxel* submap,size_t size_){
            memcpy(submap ,storedVoxelBlocks ,size_ * sizeof(TVoxel) * SDF_BLOCK_SIZE3);
        }

        void ResetVoxelBlocks()
        {
            std::fill(storedVoxelBlocks, storedVoxelBlocks + (noTotalEntries * SDF_BLOCK_SIZE3), TVoxel());
        }

        int noTotalEntries;

        ITMGlobalCache() : noTotalEntries(SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) {
            hasStoredData = (bool *) malloc(noTotalEntries * sizeof(bool));
            storedVoxelBlocks = (TVoxel *) malloc(noTotalEntries * sizeof(TVoxel) * SDF_BLOCK_SIZE3);
            memset(hasStoredData, 0, noTotalEntries);

            swapStates_host = (ITMHashSwapState *) malloc(noTotalEntries * sizeof(ITMHashSwapState));
            memset(swapStates_host, 0, sizeof(ITMHashSwapState) * noTotalEntries);

            // New
            States_host = (ITMHashSwapState *) malloc(noTotalEntries * sizeof(ITMHashSwapState));
            memset(States_host, 0, sizeof(ITMHashSwapState) * noTotalEntries);

            DWIOcudaSafeCall(cudaMallocHost((void **) &syncedVoxelBlocks_host, SDF_TRANSFER_BLOCK_NUM * sizeof(TVoxel) * SDF_BLOCK_SIZE3));
            DWIOcudaSafeCall(cudaMallocHost((void **) &hasSyncedData_host, SDF_TRANSFER_BLOCK_NUM * sizeof(bool)));
            DWIOcudaSafeCall(cudaMallocHost((void **) &neededEntryIDs_host, noTotalEntries * sizeof(int)));

            // New
            DWIOcudaSafeCall(cudaMallocHost((void **) &moveInEntryIDs_host, SDF_TRANSFER_BLOCK_NUM * sizeof(int)));

            DWIOcudaSafeCall(cudaMalloc((void **) &swapStates_device, noTotalEntries * sizeof(ITMHashSwapState)));
            DWIOcudaSafeCall(cudaMemset(swapStates_device, 0, noTotalEntries * sizeof(ITMHashSwapState)));

            // New
            DWIOcudaSafeCall(cudaMalloc((void **) &States_device, noTotalEntries * sizeof(ITMHashSwapState)));
            DWIOcudaSafeCall(cudaMemset(States_device, 0, noTotalEntries * sizeof(ITMHashSwapState)));

            DWIOcudaSafeCall(cudaMalloc((void **) &syncedVoxelBlocks_device, SDF_TRANSFER_BLOCK_NUM * sizeof(TVoxel) * SDF_BLOCK_SIZE3));
            DWIOcudaSafeCall(cudaMalloc((void **) &hasSyncedData_device, SDF_TRANSFER_BLOCK_NUM * sizeof(bool)));

            DWIOcudaSafeCall(cudaMalloc((void **) &neededEntryIDs_device, noTotalEntries * sizeof(int)));

            // New
            DWIOcudaSafeCall(cudaMalloc((void **) &moveInEntryIDs_device, SDF_TRANSFER_BLOCK_NUM * sizeof(int)));

        }

        ~ITMGlobalCache(void) {
            free(hasStoredData);
            free(storedVoxelBlocks);

            free(swapStates_host);


            DWIOcudaSafeCall(cudaFreeHost(hasSyncedData_host));
            DWIOcudaSafeCall(cudaFreeHost(syncedVoxelBlocks_host));
            DWIOcudaSafeCall(cudaFreeHost(neededEntryIDs_host));


            DWIOcudaSafeCall(cudaFree(swapStates_device));
            DWIOcudaSafeCall(cudaFree(syncedVoxelBlocks_device));
            DWIOcudaSafeCall(cudaFree(hasSyncedData_device));
            DWIOcudaSafeCall(cudaFree(neededEntryIDs_device));

            free(States_host);
            DWIOcudaSafeCall(cudaFree(States_device));
            DWIOcudaSafeCall(cudaFree(moveInEntryIDs_device));
            DWIOcudaSafeCall(cudaFreeHost(moveInEntryIDs_host));

        }
    };
}
