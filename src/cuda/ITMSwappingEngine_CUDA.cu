#include "../cuda/include/ITMSwappingEngine_CUDA.h"
#include "../cuda/include/ITMSwappingEngineShared.h"

using namespace DWIO;

template<typename T>
__device__ int computePrefixSum_device(uint element, T *sum, int localSize, int localId) {
    __shared__ uint prefixBuffer[16 * 16];
    __shared__ uint groupOffset;

    prefixBuffer[localId] = element;
    __syncthreads();

    int s1, s2;
    for (s1 = 1, s2 = 1; s1 < localSize; s1 <<= 1) {
        s2 |= s1;
        if ((localId & s2) == s2) prefixBuffer[localId] += prefixBuffer[localId - s1];
        __syncthreads();
    }

    for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1) {
        if (localId != localSize - 1 && (localId & s2) == s2) prefixBuffer[localId + s1] += prefixBuffer[localId];
        __syncthreads();
    }

    if (localId == 0 && prefixBuffer[localSize - 1] > 0) groupOffset = atomicAdd(sum, prefixBuffer[localSize - 1]);
    __syncthreads();

    int offset;
    if (localId == 0) {
        if (prefixBuffer[localId] == 0) offset = -1;
        else offset = groupOffset;
    } else {
        if (prefixBuffer[localId] == prefixBuffer[localId - 1]) offset = -1;
        else offset = groupOffset + prefixBuffer[localId - 1];
    }

    return offset;
}

namespace {
    __global__ void buildListToSwapIn_device(int *neededEntryIDs,
                                             int *noNeededEntries,
                                             ITMHashSwapState *swapStates,
                                             int noTotalEntries,
                                             ITMHashSwapState *States,
                                             int *moveInEntryIDs,
                                             int *moveInEntries);

    template<class TVoxel>
    __global__ void integrateOldIntoActiveData_device(TVoxel *localVBA,
                                                      ITMHashSwapState *swapStates,
                                                      ITMHashSwapState *States,
                                                      TVoxel *syncedVoxelBlocks_local,
                                                      int *neededEntryIDs_local,
                                                      ITMHashEntry *hashTable,
                                                      int maxW);

    template<class TVoxel>
    __global__ void integrateDataCpuAndDevice(TVoxel *localVBA,
                                              ITMHashSwapState *swapStates,
                                              ITMHashSwapState *States,
                                              TVoxel *syncedVoxelBlocks_local,
                                              int *neededEntryIDs_local,
                                              ITMHashEntry *hashTable,
                                              int maxW);

    __global__ void buildRealListToSwapOut_device(int *neededEntryIDs,
                                                  int *noNeededEntries,
                                                  ITMHashSwapState *swapStates,
                                                  ITMHashEntry *hashTable,
                                                  uchar *entriesVisibleType,
                                                  int noTotalEntries,
                                                  const uchar *emptyBlockEntries,
                                                  int *sum);

    __global__ void cleanMemory_device(int *voxelAllocationList,
                                       int *noAllocatedVoxelEntries,
                                       ITMHashSwapState *swapStates,
                                       ITMHashEntry *hashTable,
                                       int *neededEntryIDs_local,
                                       int noNeededEntries,
                                       int *swap_num);

    template<class TVoxel>
    __global__ void moveActiveDataToTransferBuffer_device(TVoxel *syncedVoxelBlocks_local,
                                                          bool *hasSyncedData_local,
                                                          int *neededEntryIDs_local,
                                                          ITMHashEntry *hashTable,
                                                          TVoxel *localVBA);

    __global__ void SetBlockSwapstate(ITMHashSwapState *swapStates,
                                      int noNeededEntries_device,
                                      int *neededEntryIDs_local);

    __global__ void MoveVoxelToGlobalMemorey_device( int *neededEntryIDs, int *noNeededEntries, ITMHashEntry *hashTable, int noTotalEntries);

}


template<class TVoxel>
ITMSwappingEngine_CUDA<TVoxel>::ITMSwappingEngine_CUDA(void) {
    DWIOcudaSafeCall(cudaMalloc((void **) &noAllocatedVoxelEntries_device, sizeof(int)));
    DWIOcudaSafeCall(cudaMalloc((void **) &noNeededEntries_device, sizeof(int)));
    DWIOcudaSafeCall(cudaMalloc((void **) &entriesToClean_device, SDF_LOCAL_BLOCK_NUM * sizeof(int)));
    DWIOcudaSafeCall(cudaMalloc((void **) &moveInEntries_device, sizeof(int)));
    DWIOcudaSafeCall(cudaMalloc((void **) &blockEmptyVerify, SDF_LOCAL_BLOCK_NUM * sizeof(uchar)));
}

template<class TVoxel>
ITMSwappingEngine_CUDA<TVoxel>::~ITMSwappingEngine_CUDA(void) {
    DWIOcudaSafeCall(cudaFree(noAllocatedVoxelEntries_device));
    DWIOcudaSafeCall(cudaFree(noNeededEntries_device));
    DWIOcudaSafeCall(cudaFree(entriesToClean_device));
    DWIOcudaSafeCall(cudaFree(moveInEntries_device));
    DWIOcudaSafeCall(cudaFree(blockEmptyVerify));
}

template<class TVoxel>
int ITMSwappingEngine_CUDA<TVoxel>::LoadFromGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene) {
    ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

    ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);
    ITMHashSwapState *States = globalCache->GetStates(true);

    TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(true);
    bool *hasSyncedData_local = globalCache->GetHasSyncedData(true);
    int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(true);
    int *moveInEntryIDs_local = globalCache->GetMoveInEntryIDs(true);
    TVoxel *syncedVoxelBlocks_global = globalCache->GetSyncedVoxelBlocks(false);
    bool *hasSyncedData_global = globalCache->GetHasSyncedData(false);
    int *moveInEntryIDs_global = globalCache->GetMoveInEntryIDs(false);

    dim3 blockSize(256);
    dim3 gridSize((int) ceil((float) scene->index.noTotalEntries / (float) blockSize.x));

    DWIOcudaSafeCall(cudaMemset(noNeededEntries_device, 0, sizeof(int)));
    DWIOcudaSafeCall(cudaMemset(moveInEntries_device, 0, sizeof(int)));
    buildListToSwapIn_device <<<gridSize, blockSize >>>
            (neededEntryIDs_local, noNeededEntries_device, swapStates,
             scene->globalCache->noTotalEntries, States, moveInEntryIDs_local, moveInEntries_device);
    cudaDeviceSynchronize();
    int noNeededEntries, noNeededEntries_2;
    DWIOcudaSafeCall(cudaMemcpy(&noNeededEntries_2, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
    DWIOcudaSafeCall(cudaMemcpy(&noNeededEntries, moveInEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
    int noNeededEntries_origin = noNeededEntries;
    int sum = 0;
    if (noNeededEntries > 0) {
        noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
        DWIOcudaSafeCall(cudaMemcpy(moveInEntryIDs_global, moveInEntryIDs_local,
                                    sizeof(int) * noNeededEntries, cudaMemcpyDeviceToHost));
        memset(syncedVoxelBlocks_global, 0, noNeededEntries * SDF_BLOCK_SIZE3 * sizeof(TVoxel));
        memset(hasSyncedData_global, 0, noNeededEntries * sizeof(bool));
        for (int i = 0; i < noNeededEntries; i++) {
            int entryId = moveInEntryIDs_global[i];

            if (globalCache->HasStoredData(entryId)) {
                sum++;
                hasSyncedData_global[i] = true;
                memcpy(syncedVoxelBlocks_global + i * SDF_BLOCK_SIZE3, globalCache->GetStoredVoxelBlock(entryId),
                       SDF_BLOCK_SIZE3 * sizeof(TVoxel));
            }
        }

        DWIOcudaSafeCall(cudaMemcpy(hasSyncedData_local, hasSyncedData_global,
                                    sizeof(bool) * noNeededEntries, cudaMemcpyHostToDevice));
        DWIOcudaSafeCall(cudaMemcpy(syncedVoxelBlocks_local, syncedVoxelBlocks_global,
                                    sizeof(TVoxel) * SDF_BLOCK_SIZE3 * noNeededEntries, cudaMemcpyHostToDevice));
    }
    return noNeededEntries_origin;
}

template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel>::IntegrateGlobalIntoLocal(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
                                                              ITMRenderState_VH *renderState,
                                                              bool updateFlag) {
    ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

    ITMHashEntry *hashTable = scene->index.GetEntries();

    ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);
    ITMHashSwapState *States = globalCache->GetStates(true);

    TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(true);
    int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(true);
    int *moveInEntryIDs_local = globalCache->GetMoveInEntryIDs(true);


    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
    int noNeededEntries_orgin = this->LoadFromGlobalMemory(scene);
    int noNeededEntries;
    int maxW = scene->maxW;
    if (noNeededEntries_orgin > SDF_TRANSFER_BLOCK_NUM)
        noNeededEntries = SDF_TRANSFER_BLOCK_NUM;
    else
        noNeededEntries = noNeededEntries_orgin;
    if (noNeededEntries > 0) {
        dim3 blockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
        dim3 gridSize(noNeededEntries);
        if (!updateFlag) {
            integrateOldIntoActiveData_device <<< gridSize, blockSize >>>
                    (localVBA, swapStates, States, syncedVoxelBlocks_local, moveInEntryIDs_local, hashTable, maxW);
            cudaDeviceSynchronize();
        } else {
            integrateDataCpuAndDevice <<< gridSize, blockSize >>>
                    (localVBA, swapStates, States, syncedVoxelBlocks_local, moveInEntryIDs_local, hashTable, maxW);
            cudaDeviceSynchronize();
        }
    }
    int moveInEntries;
    if (noNeededEntries_orgin <= SDF_TRANSFER_BLOCK_NUM) {
        DWIOcudaSafeCall(cudaMemcpy(&moveInEntries, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        moveInEntries = SDF_TRANSFER_BLOCK_NUM;
    }
    if (moveInEntries > 0) {
        dim3 cudaBlockSizeState(256, 1);
        dim3 gridSizeState((int) ceil((float) moveInEntries / (float) cudaBlockSizeState.x));
        SetBlockSwapstate <<< gridSizeState, cudaBlockSizeState >>>
                (swapStates, moveInEntries, neededEntryIDs_local);
        cudaDeviceSynchronize();
    }
    if (noNeededEntries_orgin > SDF_TRANSFER_BLOCK_NUM) {
        IntegrateGlobalIntoLocal(scene, renderState, updateFlag);
    }
}

template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel>::TransferGlobalMap(ITMScene<TVoxel, ITMVoxelBlockHash> *scene) {
    ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

    ITMHashEntry *hashTable = scene->index.GetEntries();

    ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);
    ITMHashSwapState *States = globalCache->GetStates(true);

    TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(true);
    int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(true);
    int *moveInEntryIDs_local = globalCache->GetMoveInEntryIDs(true);

    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
    int noNeededEntries = this->LoadFromGlobalMemory(scene);
    int maxW = scene->maxW;
    noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
    if (noNeededEntries > 0) {
        dim3 blockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
        dim3 gridSize(noNeededEntries);

        integrateOldIntoActiveData_device <<< gridSize, blockSize >>>
                (localVBA, swapStates, States, syncedVoxelBlocks_local, moveInEntryIDs_local, hashTable, maxW);
        cudaDeviceSynchronize();
    }

    if (noNeededEntries > 0) {
        dim3 cudaBlockSizeState(256, 1);
        dim3 gridSizeState((int) ceil((float) noNeededEntries / (float) cudaBlockSizeState.x));
        SetBlockSwapstate <<< gridSizeState, cudaBlockSizeState >>>
                (swapStates, noNeededEntries, neededEntryIDs_local);
        cudaDeviceSynchronize();
    }
}

template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel>::SaveToGlobalMemory(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState) {
    ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

    ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);

    ITMHashEntry *hashTable = scene->index.GetEntries();
    uchar *entriesVisibleType = renderState->GetEntriesVisibleType();
    uchar *emptyBlockEntries = renderState->GetEmptyBlockEntries();

    TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(true);
    bool *hasSyncedData_local = globalCache->GetHasSyncedData(true);
    int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(true);
    TVoxel *syncedVoxelBlocks_global = globalCache->GetSyncedVoxelBlocks(false);
    bool *hasSyncedData_global = globalCache->GetHasSyncedData(false);
    int *neededEntryIDs_global = globalCache->GetNeededEntryIDs(false);

    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
    int *voxelAllocationList = scene->localVBA.GetAllocationList();

    int noTotalEntries = scene->index.noTotalEntries;

    dim3 blockSize, gridSize;
    int noNeededEntries;

    blockSize = dim3(256);
    gridSize = dim3((int) ceil((float) scene->index.noTotalEntries / (float) blockSize.x));

    int *noEmpty_device;
    int *noEmpty = (int *) malloc(sizeof(int));
    cudaMalloc((void **) &noEmpty_device, sizeof(int));
    cudaMemset(noEmpty_device, 0, sizeof(int));
    DWIOcudaSafeCall(cudaMemset(noNeededEntries_device, 0, sizeof(int)));
    buildRealListToSwapOut_device <<< gridSize, blockSize >>>
            (neededEntryIDs_local, noNeededEntries_device, swapStates,
             hashTable, entriesVisibleType, noTotalEntries, emptyBlockEntries, noEmpty_device);
    cudaMemcpy(noEmpty, noEmpty_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    DWIOcudaSafeCall(cudaMemcpy(&noNeededEntries, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout<<"need to move out block :"<<noEmpty[0]<<" noNeededEntries:"<<noNeededEntries<<std::endl;
    cudaFree(noEmpty_device);
    free(noEmpty);

    if (noNeededEntries > 0) {
        noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
        {
            blockSize = dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
            gridSize = dim3(noNeededEntries);
            moveActiveDataToTransferBuffer_device <<< gridSize, blockSize >>>
                    (syncedVoxelBlocks_local, hasSyncedData_local, neededEntryIDs_local, hashTable, localVBA);
            cudaDeviceSynchronize();
        }

        DWIOcudaSafeCall(cudaMemcpy(neededEntryIDs_global, neededEntryIDs_local,
                                    sizeof(int) * noNeededEntries, cudaMemcpyDeviceToHost));
        DWIOcudaSafeCall(cudaMemcpy(hasSyncedData_global, hasSyncedData_local,
                                    sizeof(bool) * noNeededEntries, cudaMemcpyDeviceToHost));
        DWIOcudaSafeCall(cudaMemcpy(syncedVoxelBlocks_global, syncedVoxelBlocks_local,
                                    sizeof(TVoxel) * SDF_BLOCK_SIZE3 * noNeededEntries, cudaMemcpyDeviceToHost));

        int swapout = 0;
        for (int entryId = 0; entryId < noNeededEntries; entryId++) {
            if (hasSyncedData_global[entryId]) {
                globalCache->SetStoredData(neededEntryIDs_global[entryId], syncedVoxelBlocks_global + entryId * SDF_BLOCK_SIZE3);
                swapout++;
            }
        }
    }

    if (noNeededEntries > 0) {
        blockSize = dim3(256);
        gridSize = dim3((noNeededEntries + blockSize.x - 1) / blockSize.x);
        DWIOcudaSafeCall(cudaMemcpy(noAllocatedVoxelEntries_device, &scene->localVBA.lastFreeBlockId,
                                    sizeof(int), cudaMemcpyHostToDevice));
        int *swap_device;
        int *swap = (int *) malloc(sizeof(int));
        cudaMalloc((void **) &swap_device, sizeof(int));
        cudaMemset(swap_device, 0, sizeof(int));
        cleanMemory_device <<<gridSize, blockSize >>>
                (voxelAllocationList, noAllocatedVoxelEntries_device, swapStates, hashTable,
                 neededEntryIDs_local, noNeededEntries, swap_device);
        cudaMemcpy(swap, swap_device, sizeof(int), cudaMemcpyDeviceToHost);
        DWIOcudaSafeCall(cudaMemcpy(&scene->localVBA.lastFreeBlockId, noAllocatedVoxelEntries_device,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        scene->localVBA.lastFreeBlockId = MAX(scene->localVBA.lastFreeBlockId, 0);
        scene->localVBA.lastFreeBlockId = MIN(scene->localVBA.lastFreeBlockId, SDF_LOCAL_BLOCK_NUM);
        cudaFree(swap_device);
        free(swap);
    }
}

template<class TVoxel>
void ITMSwappingEngine_CUDA<TVoxel>::MoveVoxelToGlobalMemorey(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState) {
    ITMGlobalCache<TVoxel> *globalCache = scene->globalCache;

    ITMHashSwapState *swapStates = globalCache->GetSwapStates(true);

    ITMHashEntry *hashTable = scene->index.GetEntries();


    TVoxel *syncedVoxelBlocks_local = globalCache->GetSyncedVoxelBlocks(true);
    bool *hasSyncedData_local = globalCache->GetHasSyncedData(true);
    int *neededEntryIDs_local = globalCache->GetNeededEntryIDs(true);
    TVoxel *syncedVoxelBlocks_global = globalCache->GetSyncedVoxelBlocks(false);
    bool *hasSyncedData_global = globalCache->GetHasSyncedData(false);
    int *neededEntryIDs_global = globalCache->GetNeededEntryIDs(false);

    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
    int *voxelAllocationList = scene->localVBA.GetAllocationList();

    int noTotalEntries = scene->index.noTotalEntries;

    dim3 blockSize, gridSize;
    int noNeededEntries;

    blockSize = dim3(256);
    gridSize = dim3((int) ceil((float) scene->index.noTotalEntries / (float) blockSize.x));


    DWIOcudaSafeCall(cudaMemset(noNeededEntries_device, 0, sizeof(int)));
    //统计所有hash表ptr >0的索引
    MoveVoxelToGlobalMemorey_device <<< gridSize, blockSize >>>(neededEntryIDs_local, noNeededEntries_device, hashTable, noTotalEntries);


    DWIOcudaSafeCall(cudaMemcpy(&noNeededEntries, noNeededEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout<<"need to move out noNeededEntries:"<<noNeededEntries<<std::endl;

    int noNeededEntries_orin = noNeededEntries;

    if (noNeededEntries > 0) {
        noNeededEntries = MIN(noNeededEntries, SDF_TRANSFER_BLOCK_NUM);
        {
            blockSize = dim3(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
            gridSize = dim3(noNeededEntries);
            moveActiveDataToTransferBuffer_device <<< gridSize, blockSize >>>
                    (syncedVoxelBlocks_local, hasSyncedData_local, neededEntryIDs_local, hashTable, localVBA);
            cudaDeviceSynchronize();
        }

        DWIOcudaSafeCall(cudaMemcpy(neededEntryIDs_global, neededEntryIDs_local,
                                    sizeof(int) * noNeededEntries, cudaMemcpyDeviceToHost));
        DWIOcudaSafeCall(cudaMemcpy(hasSyncedData_global, hasSyncedData_local,
                                    sizeof(bool) * noNeededEntries, cudaMemcpyDeviceToHost));
        DWIOcudaSafeCall(cudaMemcpy(syncedVoxelBlocks_global, syncedVoxelBlocks_local,
                                    sizeof(TVoxel) * SDF_BLOCK_SIZE3 * noNeededEntries, cudaMemcpyDeviceToHost));

        int swapout = 0;
        for (int entryId = 0; entryId < noNeededEntries; entryId++) {
            if (hasSyncedData_global[entryId]) {
                globalCache->SetStoredData(neededEntryIDs_global[entryId], syncedVoxelBlocks_global + entryId * SDF_BLOCK_SIZE3);
                swapout++;
            }
        }
    }

    if (noNeededEntries > 0) {
        blockSize = dim3(256);
        gridSize = dim3((noNeededEntries + blockSize.x - 1) / blockSize.x);
        DWIOcudaSafeCall(cudaMemcpy(noAllocatedVoxelEntries_device, &scene->localVBA.lastFreeBlockId,
                                    sizeof(int), cudaMemcpyHostToDevice));
        int *swap_device;
        int *swap = (int *) malloc(sizeof(int));
        cudaMalloc((void **) &swap_device, sizeof(int));
        cudaMemset(swap_device, 0, sizeof(int));
        cleanMemory_device <<<gridSize, blockSize >>>
                (voxelAllocationList, noAllocatedVoxelEntries_device, swapStates, hashTable,
                 neededEntryIDs_local, noNeededEntries, swap_device);
        cudaMemcpy(swap, swap_device, sizeof(int), cudaMemcpyDeviceToHost);
        DWIOcudaSafeCall(cudaMemcpy(&scene->localVBA.lastFreeBlockId, noAllocatedVoxelEntries_device,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        scene->localVBA.lastFreeBlockId = MAX(scene->localVBA.lastFreeBlockId, 0);
        scene->localVBA.lastFreeBlockId = MIN(scene->localVBA.lastFreeBlockId, SDF_LOCAL_BLOCK_NUM);
        cudaFree(swap_device);
        free(swap);
    }

    if(noNeededEntries_orin > SDF_TRANSFER_BLOCK_NUM)//递归调用
    {
        MoveVoxelToGlobalMemorey(scene,renderState);
    }
}





namespace {
    __global__ void buildListToSwapIn_device(int *neededEntryIDs,
                                             int *noNeededEntries,
                                             ITMHashSwapState *swapStates,
                                             int noTotalEntries,
                                             ITMHashSwapState *States,
                                             int *moveInEntryIDs,
                                             int *moveInEntries) {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1) return;

        __shared__ bool shouldPrefix;
        __shared__ bool shouldMoveIn;

        shouldPrefix = false;
        shouldMoveIn = false;
        __syncthreads();

        bool isNeededId = (swapStates[targetIdx].state == 1);
        bool isNeedMoveIn = (swapStates[targetIdx].state == 1 && States[targetIdx].state == 1);
        if (isNeededId) shouldPrefix = true;
        if (isNeedMoveIn) shouldMoveIn = true;
        __syncthreads();

        if (shouldPrefix) {
            int offset = computePrefixSum_device<int>(isNeededId, noNeededEntries, blockDim.x * blockDim.y, threadIdx.x);
            if (offset != -1 && offset < noTotalEntries) {
                neededEntryIDs[offset] = targetIdx;
            }
            if (shouldMoveIn) {
                int set = computePrefixSum_device<int>(isNeedMoveIn, moveInEntries, blockDim.x * blockDim.y, threadIdx.x);
                if (set != -1 && set < SDF_TRANSFER_BLOCK_NUM) {
                    moveInEntryIDs[set] = targetIdx;
                }
            }
        }
    }

    __global__ void buildRealListToSwapOut_device(int *neededEntryIDs,
                                                  int *noNeededEntries,
                                                  ITMHashSwapState *swapStates,
                                                  ITMHashEntry *hashTable,
                                                  uchar *entriesVisibleType,
                                                  int noTotalEntries,
                                                  const uchar *emptyBlockEntries,
                                                  int *sum) {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1) return;

        __shared__ bool shouldPrefix;

        shouldPrefix = false;
        __syncthreads();

        ITMHashSwapState &swapState = swapStates[targetIdx];
        bool isNeededId = swapState.state == 2 && hashTable[targetIdx].ptr >= 0
                          && entriesVisibleType[targetIdx] == 0 && emptyBlockEntries[targetIdx] == 1;

        if (isNeededId) {
            shouldPrefix = true;
            atomicAdd(sum, 1);
        }
        __syncthreads();

        if (shouldPrefix) {
            int offset = computePrefixSum_device<int>(isNeededId, noNeededEntries, blockDim.x * blockDim.y, threadIdx.x);
            if (offset != -1 && offset < SDF_TRANSFER_BLOCK_NUM) {
                neededEntryIDs[offset] = targetIdx;
            }
        }
    }

    __global__ void cleanMemory_device(int *voxelAllocationList,
                                       int *noAllocatedVoxelEntries,
                                       ITMHashSwapState *swapStates,
                                       ITMHashEntry *hashTable,
                                       int *neededEntryIDs_local,
                                       int noNeededEntries,
                                       int *swap_num) {
        int locId = threadIdx.x + blockIdx.x * blockDim.x;

        if (locId > noNeededEntries - 1) return;

        int entryDestId = neededEntryIDs_local[locId];

        swapStates[entryDestId].state = 0;

        int vbaIdx = atomicAdd(&noAllocatedVoxelEntries[0], 1);
        if (vbaIdx < SDF_LOCAL_BLOCK_NUM - 1) {
            voxelAllocationList[vbaIdx + 1] = hashTable[entryDestId].ptr;
            atomicAdd(swap_num, 1);
            hashTable[entryDestId].ptr = -1;
        }
    }

    template<class TVoxel>
    __global__ void moveActiveDataToTransferBuffer_device(TVoxel *syncedVoxelBlocks_local,
                                                          bool *hasSyncedData_local,
                                                          int *neededEntryIDs_local,
                                                          ITMHashEntry *hashTable,
                                                          TVoxel *localVBA) {
        int entryDestId = neededEntryIDs_local[blockIdx.x];

        ITMHashEntry &hashEntry = hashTable[entryDestId];

        TVoxel *dstVB = syncedVoxelBlocks_local + blockIdx.x * SDF_BLOCK_SIZE3;
        TVoxel *srcVB = localVBA + hashEntry.ptr * SDF_BLOCK_SIZE3;

        int vIdx = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
        dstVB[vIdx] = srcVB[vIdx];
        srcVB[vIdx] = TVoxel();

        if (vIdx == 0) hasSyncedData_local[blockIdx.x] = true;
    }

    template<class TVoxel>
    __global__ void integrateOldIntoActiveData_device(TVoxel *localVBA,
                                                      ITMHashSwapState *swapStates,
                                                      ITMHashSwapState *States,
                                                      TVoxel *syncedVoxelBlocks_local,
                                                      int *neededEntryIDs_local,
                                                      ITMHashEntry *hashTable,
                                                      int maxW) {
        int entryDestId = neededEntryIDs_local[blockIdx.x];

        TVoxel *srcVB = syncedVoxelBlocks_local + blockIdx.x * SDF_BLOCK_SIZE3;
        TVoxel *dstVB = localVBA + hashTable[entryDestId].ptr * SDF_BLOCK_SIZE3;

        int vIdx = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
        dstVB[vIdx] = srcVB[vIdx];

        if (vIdx == 0) States[entryDestId].state = 0;
    }

    template<class TVoxel>
    __global__ void integrateDataCpuAndDevice(TVoxel *localVBA,
                                              ITMHashSwapState *swapStates,
                                              ITMHashSwapState *States,
                                              TVoxel *syncedVoxelBlocks_local,
                                              int *neededEntryIDs_local,
                                              ITMHashEntry *hashTable,
                                              int maxW) {
        int entryDestId = neededEntryIDs_local[blockIdx.x];

        TVoxel *srcVB = syncedVoxelBlocks_local + blockIdx.x * SDF_BLOCK_SIZE3;
        TVoxel *dstVB = localVBA + hashTable[entryDestId].ptr * SDF_BLOCK_SIZE3;

        int vIdx = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

        CombineVoxelInformation<true, TVoxel>::compute(srcVB[vIdx], dstVB[vIdx], maxW);

        if (vIdx == 0) States[entryDestId].state = 0;
    }

    __global__ void SetBlockSwapstate(ITMHashSwapState *swapStates, int noNeededEntries_device, int *neededEntryIDs_local) {
        int entryId = threadIdx.x + blockIdx.x * blockDim.x;
        if (entryId > noNeededEntries_device - 1)
            return;
        int entryDestId = neededEntryIDs_local[entryId];
        swapStates[entryDestId].state = 2;
    }

    __global__ void MoveVoxelToGlobalMemorey_device( int *neededEntryIDs, int *noNeededEntries, ITMHashEntry *hashTable, int noTotalEntries)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1) return;

        __shared__ bool shouldPrefix;

        shouldPrefix = false;
        __syncthreads();

        bool isNeededId = hashTable[targetIdx].ptr >= 0;

        if (isNeededId) {
            shouldPrefix = true;
        }
        __syncthreads();

        if (shouldPrefix) {
            int offset = computePrefixSum_device<int>(isNeededId, noNeededEntries, blockDim.x * blockDim.y, threadIdx.x);
            if (offset != -1 && offset < SDF_TRANSFER_BLOCK_NUM) {
                neededEntryIDs[offset] = targetIdx;
            }
        }        
    }

}

