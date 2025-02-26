#include <utility>

#include "../cuda/include/ITMSceneReconstructionEngine_CUDA.h"
#include "../cuda/include/ITMSceneReconstructionEngineShared.h"

using namespace DWIO;
namespace
{

    template <class TVoxel, bool stopMaxW>
    __global__ void integrateIntoScene_device(TVoxel *localVBA,
                                              const ITMHashEntry *hashTable,
                                              int *noVisibleEntryIDs,
                                              const PtrStepSz<float> depth_map,
                                              const PtrStepSz<uchar3> color_map,
                                              Eigen::Matrix4f M_d,
                                              Eigen::Matrix4f M_rgb,
                                              Vector4f projParams_d,
                                              Vector4f projParams_rgb,
                                              float _voxelSize,
                                              float mu,
                                              int maxW);

    __global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType,
                                                        uchar *entriesVisibleType,
                                                        Vector4s *blockCoords,
                                                        const PtrStepSz<float> depth,
                                                        Eigen::Matrix4f invM_d,
                                                        Vector4f projParams_d,
                                                        float mu,
                                                        float _voxelSize,
                                                        ITMHashEntry *hashTable,
                                                        float viewFrustum_min,
                                                        float viewFrustum_max,
                                                        uchar *emptyBlockEntries);

    __global__ void allocateVoxelBlocksList_device(int *voxelAllocationList,
                                                   int *excessAllocationList,
                                                   ITMHashEntry *hashTable,
                                                   int noTotalEntries,
                                                   AllocationTempData *allocData,
                                                   uchar *entriesAllocType,
                                                   uchar *entriesVisibleType,
                                                   Vector4s *blockCoords,
                                                   int *alloc_device,
                                                   int *alloc_extra_device,
                                                   uchar *emptyBlockEntries);

    __global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList,
                                                           ITMHashEntry *hashTable,
                                                           int noTotalEntries,
                                                           AllocationTempData *allocData,
                                                           uchar *entriesVisibleType,
                                                           ITMHashSwapState *States,
                                                           int *swapout_num_device);

    __global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries);

    template <bool useSwapping>
    __global__ void buildVisibleList_device(ITMHashEntry *hashTable,
                                            ITMHashSwapState *swapStates,
                                            int noTotalEntries,
                                            int *visibleEntryIDs,
                                            AllocationTempData *allocData,
                                            uchar *entriesVisibleType,
                                            Eigen::Matrix4f M_d,
                                            Vector4f projParams_d,
                                            Vector2i depthImgSize,
                                            float voxelSize, int *csm_size);

    __global__ void buildCsmAllocAndVisibleType_device(uchar *entriesAllocType,
                                                       uchar *entriesVisibleType,
                                                       Vector4s *blockCoords,
                                                       float _voxelSize,
                                                       ITMHashEntry *hashTable,
                                                       int *csm_size,
                                                       int *voxelAllocationList,
                                                       int *excessAllocationList,
                                                       AllocationTempData *allocData,
                                                       int *found_num_device,
                                                       int *found_extra_num_device,
                                                       int *unfound_num_device,
                                                       int *alloc_device,
                                                       int *alloc_extra_device);

    template <bool useSwapping>
    __global__ void buildCsmVisibleList_device(ITMHashEntry *hashTable,
                                               ITMHashSwapState *swapStates,
                                               int noTotalEntries,
                                               int *visibleEntryIDs,
                                               AllocationTempData *allocData,
                                               uchar *entriesVisibleType,
                                               Eigen::Matrix4f M_d,
                                               Vector4f projParams_d,
                                               Vector2i depthImgSize,
                                               float voxelSize,
                                               int *csm_size,
                                               int *sum);

    __global__ void findMapBlock(ITMHashEntry *hashTable,
                                 int *Triangles_device,
                                 int noTotalEntries,
                                 Vector3s *blockPos_device);

    __global__ void findNeedtoSwapInBlocks_device(ITMHashEntry *hashTable,
                                                  int noTotalEntries,
                                                  uchar *entriesVisibleType,
                                                  ITMHashSwapState *States,
                                                  ITMHashSwapState *swapStates,
                                                  int *NeedToSwapIn_device);

    __global__ void CheckHashTableCondition(ITMHashEntry *hashTable, int noTotalEntries, int *allHash_device, int *extraHash_device,
                                            int *block_cpu_device, int *block_gpu_device);
}

template <typename T>
__global__ void memsetKernel_device(T *devPtr, const T val, size_t nwords)
{
    size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset >= nwords)
        return;
    devPtr[offset] = val;
}

template <typename T>
__global__ void memsetKernelLarge_device(T *devPtr, const T val, size_t nwords)
{
    size_t offset = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
    if (offset >= nwords)
        return;
    devPtr[offset] = val;
}

template <typename T>
inline void memsetKernel(T *devPtr, const T val, size_t nwords)
{
    dim3 blockSize(256);
    dim3 gridSize((int)ceil((float)nwords / (float)blockSize.x));
    if (gridSize.x <= 65535)
    {
        memsetKernel_device<T><<<gridSize, blockSize>>>(devPtr, val, nwords);
    }
    else
    {
        gridSize.x = (int)ceil(sqrt((float)gridSize.x));
        gridSize.y = (int)ceil((float)nwords / (float)(blockSize.x * gridSize.x));
        memsetKernelLarge_device<T><<<gridSize, blockSize>>>(devPtr, val, nwords);
    }
}

template <typename T>
__global__ void fillArrayKernel_device(T *devPtr, size_t nwords)
{
    size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset >= nwords)
        return;
    devPtr[offset] = offset;
}

template <typename T>
inline void fillArrayKernel(T *devPtr, size_t nwords)
{
    dim3 blockSize(256);
    dim3 gridSize((int)ceil((float)nwords / (float)blockSize.x));
    fillArrayKernel_device<T><<<gridSize, blockSize>>>(devPtr, nwords);
}

template <class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel>::ITMSceneReconstructionEngine_CUDA(void)
{
    DWIOcudaSafeCall(cudaMalloc((void **)&allocationTempData_device, sizeof(AllocationTempData)));
    DWIOcudaSafeCall(cudaMallocHost((void **)&allocationTempData_host, sizeof(AllocationTempData)));

    int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
    DWIOcudaSafeCall(cudaMalloc((void **)&entriesAllocType_device, noTotalEntries));
    DWIOcudaSafeCall(cudaMalloc((void **)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));
}

template <class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel>::~ITMSceneReconstructionEngine_CUDA(void)
{
    DWIOcudaSafeCall(cudaFreeHost(allocationTempData_host));
    DWIOcudaSafeCall(cudaFree(allocationTempData_device));
    DWIOcudaSafeCall(cudaFree(entriesAllocType_device));
    DWIOcudaSafeCall(cudaFree(blockCoords_device));
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::computeMapBlock(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, Vector3s *blockPos_device)
{
    ITMHashEntry *hashTable = scene->index.GetEntries();
    int noTotalEntries = scene->index.noTotalEntries;
    dim3 cudaBlockSizeAL(256, 1);
    dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));
    int *NeedToMap_device;
    cudaMalloc((void **)&NeedToMap_device, sizeof(int));
    cudaMemset(NeedToMap_device, 0, sizeof(int));

    findMapBlock<<<gridSizeAL, cudaBlockSizeAL>>>(hashTable, NeedToMap_device, noTotalEntries, blockPos_device);
    int block_number;
    cudaMemcpy(&block_number, NeedToMap_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(NeedToMap_device);
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
    int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
    int blockSize = scene->index.getVoxelBlockSize();

    TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
    memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);

    int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
    fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);

    scene->localVBA.lastFreeBlockId = numBlocks - 1;

    ITMHashEntry tmpEntry{};
    memset(&tmpEntry, 0, sizeof(ITMHashEntry));
    tmpEntry.ptr = -2;
    ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
    memsetKernel<ITMHashEntry>(hashEntry_ptr, tmpEntry, scene->index.noTotalEntries);

    int *excessList_ptr = scene->index.GetExcessAllocationList();
    fillArrayKernel<int>(excessList_ptr, SDF_EXCESS_LIST_SIZE);
    scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
                                                                       const cv::cuda::GpuMat &depth_map,
                                                                       const Eigen::Matrix4d &pose,
                                                                       ITMRenderState_VH *renderState_vh,
                                                                       Vector4f camera_intrinsic,
                                                                       float truncation_distance, int *csm_size)
{
    float voxelSize = scene->voxel_resolution;

    Eigen::Matrix4f M_d, invM_d;
    M_d = (pose.inverse()).cast<float>();
    invM_d = pose.cast<float>();

    Vector4f projParams_d, invProjParams_d;
    projParams_d = std::move(camera_intrinsic);
    invProjParams_d = projParams_d;
    invProjParams_d(0, 0) = 1.0f / invProjParams_d(0, 0);
    invProjParams_d(1, 0) = 1.0f / invProjParams_d(1, 0);

    float mu = truncation_distance;

    int *voxelAllocationList = scene->localVBA.GetAllocationList();
    int *excessAllocationList = scene->index.GetExcessAllocationList();
    ITMHashEntry *hashTable = scene->index.GetEntries();
    ITMHashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(true) : 0;
    ITMHashSwapState *States = scene->globalCache != NULL ? scene->globalCache->GetStates(true) : 0;

    int noTotalEntries = scene->index.noTotalEntries;

    int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
    uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();
    uchar *emptyBlockEntries = renderState_vh->GetEmptyBlockEntries();

    dim3 cudaBlockSizeAL(256, 1);
    dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

    dim3 cudaBlockSizeHV(16, 16);
    dim3 gridSizeHV((int)ceil((float)depth_map.cols / (float)cudaBlockSizeHV.x),
                    ceil((float)depth_map.rows / (float)cudaBlockSizeHV.y));

    dim3 cudaBlockSizeVS(256, 1);
    dim3 gridSizeVS((int)ceil((float)renderState_vh->noVisibleEntries / (float)cudaBlockSizeVS.x));

    float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

    Vector2i depthImgSize;
    depthImgSize.x() = depth_map.cols;
    depthImgSize.y() = depth_map.rows;

    auto *tempData = (AllocationTempData *)allocationTempData_host;
    tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
    tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
    tempData->noVisibleEntries = 0;
    DWIOcudaSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

    DWIOcudaSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char) * noTotalEntries));

    if (gridSizeVS.x > 0)
    {
        setToType3<<<gridSizeVS, cudaBlockSizeVS>>>(entriesVisibleType, visibleEntryIDs, renderState_vh->noVisibleEntries);
    }
    // 为点云对应的block分配空间和标注可见性
    buildHashAllocAndVisibleType_device<<<gridSizeHV, cudaBlockSizeHV>>>(entriesAllocType_device, entriesVisibleType, blockCoords_device, depth_map,
                                                                         invM_d, invProjParams_d, mu, oneOverVoxelSize, hashTable,
                                                                         scene->viewFrustum_min, scene->viewFrustum_max, emptyBlockEntries);
    cudaCheckError();
    cudaDeviceSynchronize();

    int *alloc_device;
    int *alloc_extra_device;
    int *alloc = (int *)malloc(sizeof(int));
    int *alloc_extra = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&alloc_device, sizeof(int));
    cudaMemset(alloc_device, 0, sizeof(int));
    cudaMalloc((void **)&alloc_extra_device, sizeof(int));
    cudaMemset(alloc_extra_device, 0, sizeof(int));
    // 为上面对应的bloc填充hashTable上数据结构
    allocateVoxelBlocksList_device<<<gridSizeAL, cudaBlockSizeAL>>>(voxelAllocationList, excessAllocationList, hashTable, noTotalEntries,
                                                                    (AllocationTempData *)allocationTempData_device, entriesAllocType_device,
                                                                    entriesVisibleType, blockCoords_device, alloc_device, alloc_extra_device, emptyBlockEntries);
    cudaCheckError();
    cudaMemcpy(alloc, alloc_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(alloc_extra, alloc_extra_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "深度图分配：" << "hash : " << *alloc << "extra list: " << *alloc_extra << std::endl;
    int *csm_size_gpu;
    cudaMalloc((void **)&csm_size_gpu, sizeof(int) * 6);
    cudaMemcpy(csm_size_gpu, csm_size, sizeof(int) * 6, cudaMemcpyHostToDevice);
    // 计算可见性
    buildVisibleList_device<true><<<gridSizeAL, cudaBlockSizeAL>>>(hashTable, swapStates, noTotalEntries, visibleEntryIDs, (AllocationTempData *)allocationTempData_device,
                                                                   entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize, csm_size_gpu);
    cudaCheckError();
    cudaDeviceSynchronize();

    int *swapout_num_device;
    cudaMalloc((void **)&swapout_num_device, sizeof(int));
    cudaMemset(swapout_num_device, 0, sizeof(int));
    // 为block分配gpu内存
    reAllocateSwappedOutVoxelBlocks_device<<<gridSizeAL, cudaBlockSizeAL>>>(voxelAllocationList, hashTable, noTotalEntries, (AllocationTempData *)allocationTempData_device,
                                                                            entriesVisibleType, States, swapout_num_device);
    cudaCheckError();
    cudaDeviceSynchronize();
    // 更新相关参数
    DWIOcudaSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
    renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
    scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
    scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);
    cudaFree(csm_size_gpu);
    cudaFree(alloc_device);
    cudaFree(alloc_extra_device);
    cudaFree(swapout_num_device);
    free(alloc);
    free(alloc_extra);
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::AllocateScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
                                                              const cv::cuda::GpuMat &depth_map,
                                                              const Eigen::Matrix4d &pose,
                                                              ITMRenderState_VH *renderState_vh,
                                                              Vector4f camera_intrinsic,
                                                              float truncation_distance,
                                                              int *csm_size,
                                                              bool onlyUpdateVisibleList,
                                                              bool resetVisibleList)
{
    float voxelSize = scene->voxel_resolution;

    Eigen::Matrix4f M_d = (pose.inverse()).cast<float>();

    if (resetVisibleList)
        renderState_vh->noVisibleEntries = 0;

    Vector4f projParams_d, invProjParams_d;
    projParams_d = std::move(camera_intrinsic);
    invProjParams_d = projParams_d;
    invProjParams_d(0, 0) = 1.0f / invProjParams_d(0, 0);
    invProjParams_d(1, 0) = 1.0f / invProjParams_d(1, 0);

    int *voxelAllocationList = scene->localVBA.GetAllocationList();
    int *excessAllocationList = scene->index.GetExcessAllocationList();
    ITMHashEntry *hashTable = scene->index.GetEntries();
    ITMHashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(true) : 0;
    ITMHashSwapState *States = scene->globalCache != NULL ? scene->globalCache->GetStates(true) : 0; // 在cpu上可见的block

    int noTotalEntries = scene->index.noTotalEntries;

    int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
    uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

    dim3 cudaBlockSizeAL(256, 1);
    dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

    dim3 cudaBlockSizeCsm(8, 8, 16);
    dim3 gridSizeCSM((csm_size[3] - csm_size[0] + cudaBlockSizeCsm.x - 1) / cudaBlockSizeCsm.x,
                     (csm_size[4] - csm_size[1] + cudaBlockSizeCsm.y - 1) / cudaBlockSizeCsm.y,
                     (csm_size[5] - csm_size[2] + cudaBlockSizeCsm.z - 1) / cudaBlockSizeCsm.z);

    Vector2i depthImgSize;
    depthImgSize.x() = depth_map.cols;
    depthImgSize.y() = depth_map.rows;

    auto *tempData = (AllocationTempData *)allocationTempData_host;
    tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
    tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
    tempData->noVisibleEntries = 0;
    DWIOcudaSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

    DWIOcudaSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char) * noTotalEntries));

    int *csm_size_gpu;
    cudaMalloc((void **)&csm_size_gpu, sizeof(int) * 6);
    cudaMemcpy(csm_size_gpu, csm_size, sizeof(int) * 6, cudaMemcpyHostToDevice);
    float blockSize = 1.0f / SDF_BLOCK_SIZE;
    int *found_num_device;
    int *found_extra_num_device;
    int *unfound_num_device;
    cudaMalloc((void **)&found_num_device, sizeof(int));
    cudaMemset(found_num_device, 0, sizeof(int));
    cudaMalloc((void **)&found_extra_num_device, sizeof(int));
    cudaMemset(found_extra_num_device, 0, sizeof(int));
    cudaMalloc((void **)&unfound_num_device, sizeof(int));
    cudaMemset(unfound_num_device, 0, sizeof(int));
    int *found_num = (int *)malloc(sizeof(int));
    int *found_extra_num = (int *)malloc(sizeof(int));
    int *unfound_num = (int *)malloc(sizeof(int));

    int *alloc_device;
    int *alloc_extra_device;
    int *alloc = (int *)malloc(sizeof(int));
    int *alloc_extra = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&alloc_device, sizeof(int));
    cudaMemset(alloc_device, 0, sizeof(int));
    cudaMalloc((void **)&alloc_extra_device, sizeof(int));
    cudaMemset(alloc_extra_device, 0, sizeof(int));

    buildCsmAllocAndVisibleType_device<<<gridSizeCSM, cudaBlockSizeCsm>>>(entriesAllocType_device, entriesVisibleType, blockCoords_device, blockSize, hashTable,
                                                                          csm_size_gpu, voxelAllocationList, excessAllocationList, (AllocationTempData *)allocationTempData_device,
                                                                          found_num_device, found_extra_num_device, unfound_num_device, alloc_device, alloc_extra_device);

    cudaCheckError();
    cudaDeviceSynchronize();

    cudaMemcpy(found_num, found_num_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_extra_num, found_extra_num_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(unfound_num, unfound_num_device, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(alloc, alloc_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(alloc_extra, alloc_extra_device, sizeof(int), cudaMemcpyDeviceToHost);

    int *sum_device;
    int *sum = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&sum_device, sizeof(int));
    cudaMemset(sum_device, 0, sizeof(int));

    buildCsmVisibleList_device<true><<<gridSizeAL, cudaBlockSizeAL>>>(hashTable, swapStates, noTotalEntries, visibleEntryIDs, (AllocationTempData *)allocationTempData_device,
                                                                      entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize, csm_size_gpu, sum_device);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaMemcpy(sum, sum_device, sizeof(int), cudaMemcpyDeviceToHost);

    int *swapout_num_device;
    cudaMalloc((void **)&swapout_num_device, sizeof(int));
    cudaMemset(swapout_num_device, 0, sizeof(int));
    int *swapout_num = (int *)malloc(sizeof(int));
    reAllocateSwappedOutVoxelBlocks_device<<<gridSizeAL, cudaBlockSizeAL>>>(voxelAllocationList, hashTable, noTotalEntries, (AllocationTempData *)allocationTempData_device,
                                                                            entriesVisibleType, States, swapout_num_device);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaMemcpy(swapout_num, swapout_num_device, sizeof(int), cudaMemcpyDeviceToHost);

    DWIOcudaSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
    renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
    scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
    scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);
    cudaFree(csm_size_gpu);
    cudaFree(found_num_device);
    cudaFree(found_extra_num_device);
    cudaFree(unfound_num_device);
    cudaFree(alloc_device);
    cudaFree(alloc_extra_device);
    cudaFree(sum_device);
    cudaFree(swapout_num_device);
    free(found_num);
    free(found_extra_num);
    free(unfound_num);
    free(sum);
    free(swapout_num);
    free(alloc);
    free(alloc_extra);
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::SwapAllBlocks(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
                                                              ITMRenderState_VH *renderState_vh,
                                                              int *NeedToSwapIn)
{
    ITMHashEntry *hashTable = scene->index.GetEntries();
    ITMHashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(true) : 0;
    ITMHashSwapState *States = scene->globalCache != NULL ? scene->globalCache->GetStates(true) : 0;
    int *voxelAllocationList = scene->localVBA.GetAllocationList();

    int noTotalEntries = scene->index.noTotalEntries;
    uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();
    dim3 cudaBlockSizeAL(256, 1);
    dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));
    int *NeedToSwapIn_device;
    cudaMalloc((void **)&NeedToSwapIn_device, sizeof(int));
    cudaMemset(NeedToSwapIn_device, 0, sizeof(int));

    findNeedtoSwapInBlocks_device<<<gridSizeAL, cudaBlockSizeAL>>>(hashTable, noTotalEntries, entriesVisibleType, States, swapStates, NeedToSwapIn_device);
    DWIOcudaSafeCall(cudaMemcpy(NeedToSwapIn, NeedToSwapIn_device, sizeof(int), cudaMemcpyDeviceToHost));

    AllocationTempData *tempData = (AllocationTempData *)allocationTempData_host;
    tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
    tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
    tempData->noVisibleEntries = 0;
    DWIOcudaSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

    int *swapout_num_device;
    int *swapout_num = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&swapout_num_device, sizeof(int));
    cudaMemset(swapout_num_device, 0, sizeof(int));
    reAllocateSwappedOutVoxelBlocks_device<<<gridSizeAL, cudaBlockSizeAL>>>(voxelAllocationList, hashTable, noTotalEntries, (AllocationTempData *)allocationTempData_device,
                                                                            entriesVisibleType, States, swapout_num_device);
    cudaMemcpy(swapout_num, swapout_num_device, sizeof(int), cudaMemcpyDeviceToHost);
    DWIOcudaSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
    renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
    scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
    scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);
    cudaFree(NeedToSwapIn_device);
    cudaFree(swapout_num_device);
    free(swapout_num);
}

template <class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
                                                                   const cv::cuda::GpuMat &depth_map,
                                                                   const cv::cuda::GpuMat &rgb,
                                                                   const Eigen::Matrix4d &pose_inv,
                                                                   ITMRenderState_VH *renderState_vh,
                                                                   Vector4f camera_intrinsic,
                                                                   float truncation_distance)
{
    float voxelSize = scene->voxel_resolution;

    Eigen::Matrix4f M_d, M_rgb;
    Vector4f projParams_d, projParams_rgb;

    if (renderState_vh->noVisibleEntries == 0) // 多少block是可见的
        return;

    M_d = pose_inv.cast<float>();
    M_rgb = M_d; // 后面需要修改一下
    projParams_d = camera_intrinsic;
    projParams_rgb = camera_intrinsic; // 后面需要修改一下

    float mu = truncation_distance;
    int maxW = MAX_WEIGHT;

    TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
    ITMHashEntry *hashTable = scene->index.GetEntries();

    int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs(); // 存在可见block的hash索引

    dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
    dim3 gridSize(renderState_vh->noVisibleEntries);

    integrateIntoScene_device<TVoxel, true><<<gridSize, cudaBlockSize>>>(localVBA, hashTable, visibleEntryIDs, depth_map, rgb,
                                                                         M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
    cudaCheckError();
}

template <class TVoxel>
bool ITMSceneReconstructionEngine_CUDA<TVoxel>::showHashTableAndVoxelAllocCondition(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, ITMRenderState_VH *renderState_vh)
{
    ITMHashEntry *hashTable = scene->index.GetEntries(); // 有序hash表数组

    int *voxelAllocationList = scene->localVBA.GetAllocationList();

    int noTotalEntries = scene->index.noTotalEntries;                    // 最多能存放的block块(或者说hash条目)
    uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType(); // 表明block是否可见的列表
    dim3 cudaBlockSizeAL(256, 1);                                        // 给block处理的线程、网格块
    dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

    // 1、我想统计使用了多少gpu的体素
    std::cout << "gpu voxel used condition:" << scene->localVBA.lastFreeBlockId << std::endl;
    std::cout << "extra list condition: " << scene->index.GetLastFreeExcessListId() << std::endl;
    // 2、我想统计hashTable全部链表和额外链表的使用情况，还有在gpu和在cpu的block情况
    int *allHash_device;
    cudaMalloc((void **)&allHash_device, sizeof(int));
    cudaMemset(allHash_device, 0, sizeof(int));
    int *extraHash_device;
    cudaMalloc((void **)&extraHash_device, sizeof(int));
    cudaMemset(extraHash_device, 0, sizeof(int));
    int *block_cpu_device;
    cudaMalloc((void **)&block_cpu_device, sizeof(int));
    cudaMemset(block_cpu_device, 0, sizeof(int));
    int *block_gpu_device;
    cudaMalloc((void **)&block_gpu_device, sizeof(int));
    cudaMemset(block_gpu_device, 0, sizeof(int));
    CheckHashTableCondition<<<gridSizeAL, cudaBlockSizeAL>>>(hashTable, noTotalEntries, allHash_device, extraHash_device, block_cpu_device, block_gpu_device);
    cudaCheckError();
    int allHash;
    int extraHash;
    int block_cpu;
    int block_gpu;
    cudaMemcpy(&allHash, allHash_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&extraHash, extraHash_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&block_cpu, block_cpu_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&block_gpu, block_gpu_device, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "hashTable used condition:" << allHash << " extra hash list:" << extraHash << std::endl;
    std::cout << "block in cpu:" << block_cpu << " block in gpu: " << block_gpu << std::endl;
    // std::cout<<"*************************************************"<<std::endl;
    cudaFree(allHash_device);
    cudaFree(extraHash_device);
    cudaFree(block_cpu_device);
    cudaFree(block_gpu_device);

#ifdef SUBMAP
    if (SDF_EXCESS_LIST_SIZE - extraHash < 150)
        return true; // 应该创立新子图了
    else
        return false;
#else
    if (allHash >= 5000)
        return true; // 应该创立新子图了
    else
        return false;
#endif
}

namespace
{

    template <class TVoxel, bool stopMaxW>
    __global__ void integrateIntoScene_device(TVoxel *localVBA,
                                              const ITMHashEntry *hashTable,
                                              int *visibleEntryIDs,
                                              const PtrStepSz<float> depth_map,
                                              const PtrStepSz<uchar3> color_map,
                                              Eigen::Matrix4f M_d,
                                              Eigen::Matrix4f M_rgb,
                                              Vector4f projParams_d,
                                              Vector4f projParams_rgb,
                                              float _voxelSize,
                                              float mu,
                                              int maxW)
    {
        int entryId = visibleEntryIDs[blockIdx.x];

        const ITMHashEntry &currentHashEntry = hashTable[entryId];

        if (currentHashEntry.ptr < 0)
            return;

        Vector3i globalPos;
        globalPos.x() = currentHashEntry.pos.x * SDF_BLOCK_SIZE;
        globalPos.y() = currentHashEntry.pos.y * SDF_BLOCK_SIZE;
        globalPos.z() = currentHashEntry.pos.z * SDF_BLOCK_SIZE;

        TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

        int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
        int locId;
        locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

        // if (stopMaxW) if (localVoxelBlock[locId].w_depth == maxW) return;
        if (localVoxelBlock[locId].w_depth == maxW)
            return; // 这里可能是个问题,

        Vector4f pt_model;
        pt_model(0, 0) = (float)(globalPos.x() + x) * _voxelSize;
        pt_model(1, 0) = (float)(globalPos.y() + y) * _voxelSize;
        pt_model(2, 0) = (float)(globalPos.z() + z) * _voxelSize;
        pt_model(3, 0) = 1.0f;
        // 终于知道颜色乱投影的原因了，因为csm子图在划分范围，划分了一大片都是可见，所以在这里这些可见的block都会被更新颜色，但是只要block在相机的投影范围内就都会被更新颜色，
        // 这就导致了中间一些block被赋上一些奇怪的颜色
        ComputeUpdatedVoxelInfo<true, false, TVoxel>::compute(localVoxelBlock[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth_map, color_map);
    }

    __global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries)
    {
        int entryId = threadIdx.x + blockIdx.x * blockDim.x;
        if (entryId > noVisibleEntries - 1)
            return;
        entriesVisibleType[visibleEntryIDs[entryId]] = 3;
    }

    __global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType,
                                                        uchar *entriesVisibleType,
                                                        Vector4s *blockCoords,
                                                        const PtrStepSz<float> depth,
                                                        Eigen::Matrix4f invM_d,
                                                        Vector4f projParams_d,
                                                        float mu,
                                                        float _voxelSize,
                                                        ITMHashEntry *hashTable,
                                                        float viewFrustum_min,
                                                        float viewFrustum_max,
                                                        uchar *emptyBlockEntries)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x > depth.cols - 1 || y > depth.rows - 1)
            return;

        buildHashAllocAndVisibleTypePP(entriesAllocType,
                                       entriesVisibleType,
                                       x, y,
                                       blockCoords,
                                       depth,
                                       invM_d,
                                       projParams_d,
                                       mu,
                                       _voxelSize,
                                       hashTable,
                                       viewFrustum_min,
                                       viewFrustum_max,
                                       emptyBlockEntries);
    }

    __global__ void allocateVoxelBlocksList_device(int *voxelAllocationList,
                                                   int *excessAllocationList,
                                                   ITMHashEntry *hashTable,
                                                   int noTotalEntries,
                                                   AllocationTempData *allocData,
                                                   uchar *entriesAllocType,
                                                   uchar *entriesVisibleType,
                                                   Vector4s *blockCoords,
                                                   int *alloc_device,
                                                   int *alloc_extra_device,
                                                   uchar *emptyBlockEntries)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;

        int vbaIdx, exlIdx;

        switch (entriesAllocType[targetIdx])
        {
        case 1: // needs allocation, fits in the ordered list
            vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);

            if (vbaIdx >= 0) // there is room in the voxel block array
            {
                Vector4s pt_block_all = blockCoords[targetIdx];

                ITMHashEntry hashEntry{};
                hashEntry.pos.x = pt_block_all(0, 0);
                hashEntry.pos.y = pt_block_all(1, 0);
                hashEntry.pos.z = pt_block_all(2, 0);
                hashEntry.ptr = voxelAllocationList[vbaIdx];
                hashEntry.offset = 0;

                hashTable[targetIdx] = hashEntry;
                atomicAdd(alloc_device, 1);
            }
            else
            {
                // Mark entry as not visible since we couldn't allocate it but buildHashAllocAndVisibleTypePP changed its state.
                entriesVisibleType[targetIdx] = 0;

                // Restore the previous value to avoid leaks.
                atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
            }
            break;

        case 2: // needs allocation in the excess list
            vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
            exlIdx = atomicSub(&allocData->noAllocatedExcessEntries, 1); //

            if (vbaIdx >= 0 && exlIdx >= 0) // there is room in the voxel block array and excess list
            {
                Vector4s pt_block_all = blockCoords[targetIdx];

                ITMHashEntry hashEntry{};
                hashEntry.pos.x = pt_block_all(0, 0);
                hashEntry.pos.y = pt_block_all(1, 0);
                hashEntry.pos.z = pt_block_all(2, 0);
                hashEntry.ptr = voxelAllocationList[vbaIdx];
                hashEntry.offset = 0;

                int exlOffset = excessAllocationList[exlIdx];
                hashTable[targetIdx].offset = exlOffset + 1;        // connect to child
                hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry;  // add child to the excess list
                entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; // make child visible
                emptyBlockEntries[SDF_BUCKET_NUM + exlOffset] = 1;
                atomicAdd(alloc_extra_device, 1);
            }
            else
            {
                // No need to mark the entry as not visible since buildHashAllocAndVisibleTypePP did not mark it.
                // Restore the previous values to avoid leaks.
                atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
                atomicAdd(&allocData->noAllocatedExcessEntries, 1);
            }

            break;
        }
    }

    template <bool useSwapping>
    __global__ void buildVisibleList_device(ITMHashEntry *hashTable,
                                            ITMHashSwapState *swapStates,
                                            int noTotalEntries,
                                            int *visibleEntryIDs,
                                            AllocationTempData *allocData,
                                            uchar *entriesVisibleType,
                                            Eigen::Matrix4f M_d,
                                            Vector4f projParams_d,
                                            Vector2i depthImgSize,
                                            float voxelSize, int *csm_size)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;

        __shared__ bool shouldPrefix;
        shouldPrefix = false;
        __syncthreads();

        unsigned char hashVisibleType = entriesVisibleType[targetIdx];
        const ITMHashEntry &hashEntry = hashTable[targetIdx];
        if (hashVisibleType == 3)
        {
            bool isVisibleEnlarged, isVisible;

            if (useSwapping)
            {
                checkCsmBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize, csm_size);
                checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
                if (!isVisibleEnlarged)
                    hashVisibleType = 0;
            }
            else
            {
                checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
                if (!isVisible)
                    hashVisibleType = 0;
            }
            entriesVisibleType[targetIdx] = hashVisibleType;
        }

        if (hashVisibleType > 0)
            shouldPrefix = true;

        if (useSwapping)
        {
            if (hashVisibleType > 0 && swapStates[targetIdx].state != 2)
                swapStates[targetIdx].state = 1;
        }

        __syncthreads();

        if (shouldPrefix)
        {
            int offset = computePrefixSum_device<int>(hashVisibleType > 0, &allocData->noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
            if (offset != -1)
                visibleEntryIDs[offset] = targetIdx;
        }
    }

    __global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList,
                                                           ITMHashEntry *hashTable,
                                                           int noTotalEntries,
                                                           AllocationTempData *allocData,
                                                           uchar *entriesVisibleType,
                                                           ITMHashSwapState *States,
                                                           int *swapout_num_device)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;

        int vbaIdx;
        int hashEntry_ptr = hashTable[targetIdx].ptr;
        // 可见且在cpu上的需要在gpu上分配内存
        if (entriesVisibleType[targetIdx] > 0 &&
            hashEntry_ptr == -1) // it is visible and has been previously allocated inside the hash, but deallocated from VBA
        {
            vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
            if (vbaIdx >= 0)
            {
                hashTable[targetIdx].ptr = voxelAllocationList[vbaIdx];
                States[targetIdx].state = 1;
                atomicAdd(swapout_num_device, 1);
            }
            else
            {
                atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
            }
        }
    }

    __global__ void buildCsmAllocAndVisibleType_device(uchar *entriesAllocType,
                                                       uchar *entriesVisibleType,
                                                       Vector4s *blockCoords,
                                                       float _voxelSize,
                                                       ITMHashEntry *hashTable,
                                                       int *csm_size,
                                                       int *voxelAllocationList,
                                                       int *excessAllocationList,
                                                       AllocationTempData *allocData,
                                                       int *found_num_device,
                                                       int *found_extra_num_device,
                                                       int *unfound_num_device,
                                                       int *alloc_device,
                                                       int *alloc_extra_device)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int z = blockIdx.z * blockDim.z + threadIdx.z;
        if (x > (csm_size[3] - csm_size[0]) || y > (csm_size[4] - csm_size[1]) ||
            z > (csm_size[5] - csm_size[2]))
        {
            return;
        }
        Vector3s blockPos;
        unsigned int hashIdx;

        blockPos.x() = x + csm_size[0];
        blockPos.y() = y + csm_size[1];
        blockPos.z() = z + csm_size[2];

        hashIdx = hashIndex(blockPos);
        bool isFound = false;

        ITMHashEntry hashEntry = hashTable[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
        {
            // entry has been streamed out but is visible or in memory and visible，如果hashEntry.ptr == -1是这个block被观察到，但是却在cpu上
            entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? 2 : 1; //>1表示这个block被观察到且在GPU上

            isFound = true;
            if (hashEntry.ptr == -1)
            {
                atomicAdd(found_num_device, 1);
            }
        }

        if (!isFound)
        {
            if (hashEntry.ptr >= -1) // 如果这个hashEntry.ptr<-1说明这个block还没被分配直接到最后的条件语句中对其进行可见性和是否需要分配进行标记（可见的，需要分配的）
            {
                while (hashEntry.offset >= 1)
                {
                    hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
                    hashEntry = hashTable[hashIdx];

                    if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
                    {
                        // entry has been streamed out but is visible or in memory and visible
                        entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? 2 : 1;

                        isFound = true;
                        atomicAdd(found_extra_num_device, 1);
                        if (hashEntry.ptr == -1)
                        {
                            atomicAdd(found_num_device, 1);
                        }
                        break;
                    }
                }
            }
        }
    }

    template <bool useSwapping>
    __global__ void buildCsmVisibleList_device(ITMHashEntry *hashTable,
                                               ITMHashSwapState *swapStates,
                                               int noTotalEntries,
                                               int *visibleEntryIDs,
                                               AllocationTempData *allocData,
                                               uchar *entriesVisibleType,
                                               Eigen::Matrix4f M_d,
                                               Vector4f projParams_d,
                                               Vector2i depthImgSize,
                                               float voxelSize,
                                               int *csm_size,
                                               int *sum)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;

        __shared__ bool shouldPrefix;
        shouldPrefix = false;
        __syncthreads();

        unsigned char hashVisibleType = entriesVisibleType[targetIdx];

        if (hashVisibleType > 0)
            shouldPrefix = true;

        if (useSwapping)
        {
            if (hashVisibleType > 0 && swapStates[targetIdx].state != 2)
            {
                atomicAdd(sum, 1);
                swapStates[targetIdx].state = 1; // 可见的，同时不是刚刚更新的block
            }
        }

        __syncthreads();

        if (shouldPrefix)
        {
            int offset = computePrefixSum_device<int>(hashVisibleType > 0, &allocData->noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
            if (offset != -1)
            {
                visibleEntryIDs[offset] = targetIdx;
            }
        }
    }

    __global__ void findMapBlock(ITMHashEntry *hashTable,
                                 int *Triangles_device,
                                 int noTotalEntries,
                                 Vector3s *blockPos_device)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;

        ITMHashEntry hashEntry = hashTable[targetIdx];

        if (hashEntry.ptr >= -1)
        {
            atomicAdd(Triangles_device, 1);
            blockPos_device[Triangles_device[0] - 1] = Vector3s(hashEntry.pos.x, hashEntry.pos.y, hashEntry.pos.z);
        }
    }

    __global__ void findNeedtoSwapInBlocks_device(ITMHashEntry *hashTable,
                                                  int noTotalEntries,
                                                  uchar *entriesVisibleType,
                                                  ITMHashSwapState *States,
                                                  ITMHashSwapState *swapStates,
                                                  int *NeedToSwapIn_device)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;
        bool flag = (hashTable[targetIdx].ptr == -1);
        if (flag)
        {
            entriesVisibleType[targetIdx] = 1;
            States[targetIdx].state = 1;
            swapStates[targetIdx].state = 1;
            atomicAdd(NeedToSwapIn_device, 1);
        }
    }

    __global__ void CheckHashTableCondition(ITMHashEntry *hashTable, int noTotalEntries, int *allHash_device, int *extraHash_device,
                                            int *block_cpu_device, int *block_gpu_device)
    {
        int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (targetIdx > noTotalEntries - 1)
            return;
        if (hashTable[targetIdx].ptr >= -1)
        {
            atomicAdd(allHash_device, 1);
            if (targetIdx > SDF_BUCKET_NUM - 1)
            {
                // printf("extra hashEntry:%d\n",targetIdx);
                atomicAdd(extraHash_device, 1);
            }
            if (hashTable[targetIdx].ptr > -1)
            {
                atomicAdd(block_gpu_device, 1);
            }
            else
            {
                atomicAdd(block_cpu_device, 1);
            }
        }
    }

}
