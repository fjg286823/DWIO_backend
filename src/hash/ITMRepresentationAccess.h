
#pragma once

#define THREADPTR(x) x
#define CONSTPTR(x) x

#include "ITMVoxelBlockHash.h"
#include "../cuda/include/common.h"
#include "Submap.h"
#include "ITMVoxelTypes.h"

template<typename T>
__device__ inline int hashIndex(const THREADPTR(T) &blockPos) {
    return (((uint) blockPos.x() * 73856093u) ^ ((uint) blockPos.y() * 19349669u) ^ ((uint) blockPos.z() * 83492791u)) & (uint) SDF_HASH_MASK;
}

template<typename T>
inline int hashIndexCPU(const THREADPTR(T) &blockPos) {
    return (((uint) blockPos.x() * 73856093u) ^ ((uint) blockPos.y() * 19349669u) ^ ((uint) blockPos.z() * 83492791u)) & (uint) SDF_HASH_MASK;
}


template<typename T>
inline int hashIndexGlobal(const THREADPTR(T) &blockPos) {
    return (((uint) blockPos.x() * 73856093u) ^ ((uint) blockPos.y() * 19349669u) ^ ((uint) blockPos.z() * 83492791u)) & (uint) MAP_HASH_MASK;
}

//这种计算方式，保证子图内部一定都是加吗？
__device__ inline int pointToVoxelBlockPos(const THREADPTR(Vector3i) &point, Vector3s &blockPos) {
    blockPos.x() = ((point.x() < 0) ? point.x() - SDF_BLOCK_SIZE + 1 : point.x()) / SDF_BLOCK_SIZE;
    blockPos.y() = ((point.y() < 0) ? point.y() - SDF_BLOCK_SIZE + 1 : point.y()) / SDF_BLOCK_SIZE;
    blockPos.z() = ((point.z() < 0) ? point.z() - SDF_BLOCK_SIZE + 1 : point.z()) / SDF_BLOCK_SIZE;

    return point.x() + (point.y() - blockPos.x()) * SDF_BLOCK_SIZE + (point.z() - blockPos.y()) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE -
           blockPos.z() * SDF_BLOCK_SIZE3;
}

inline int pointToVoxelBlockPosCpu(const THREADPTR(Vector3i) &point, Vector3s &blockPos) {
    blockPos.x() = ((point.x() < 0) ? point.x() - SDF_BLOCK_SIZE + 1 : point.x()) / SDF_BLOCK_SIZE;
    blockPos.y() = ((point.y() < 0) ? point.y() - SDF_BLOCK_SIZE + 1 : point.y()) / SDF_BLOCK_SIZE;
    blockPos.z() = ((point.z() < 0) ? point.z() - SDF_BLOCK_SIZE + 1 : point.z()) / SDF_BLOCK_SIZE;

    return point.x() + (point.y() - blockPos.x()) * SDF_BLOCK_SIZE + (point.z() - blockPos.y()) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE -
           blockPos.z() * SDF_BLOCK_SIZE3;
}

template<class TVoxel>
__device__ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   const THREADPTR(Vector3i) &point,
                                   THREADPTR(int) &vmIndex,
                                   THREADPTR(DWIO::ITMVoxelBlockHash::IndexCache) &cache) {
    Vector3s blockPos;
    int linearIdx = pointToVoxelBlockPos(point, blockPos);

    if IS_EQUAL3(cache.blockPos, blockPos) {
        vmIndex = true;
        return voxelData[cache.blockPtr + linearIdx];
    }

    int hashIdx = hashIndex(blockPos);

    while (true) {
        ITMHashEntry hashEntry = voxelIndex[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0) {
            cache.blockPos.x = blockPos.x();
            cache.blockPos.y = blockPos.y();
            cache.blockPos.z = blockPos.z();
            cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
            vmIndex = hashIdx + 1; // add 1 to support legacy true / false operations for isFound

            return voxelData[cache.blockPtr + linearIdx];
        }

        if (hashEntry.offset < 1) break;
        hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
    }

    vmIndex = false;
    return TVoxel();
}


template<class TVoxel>
__device__ inline TVoxel readVoxelCpu(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   const THREADPTR(Vector3i) &point,
                                   THREADPTR(int) &vmIndex,
                                   THREADPTR(DWIO::ITMVoxelBlockHash::IndexCache) &cache) {
    Vector3s blockPos;
    int linearIdx = pointToVoxelBlockPos(point, blockPos);
    //根据blockpos计算出hash索引，返回对应的体素

    int hashIdx = hashIndex(blockPos);
    if IS_EQUAL3(cache.blockPos, blockPos) {
        vmIndex = true;
        return voxelData[hashIdx*SDF_BLOCK_SIZE3 + linearIdx];
    }

    while (true) {
        ITMHashEntry hashEntry = voxelIndex[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) ) {
            cache.blockPos.x = blockPos.x();
            cache.blockPos.y = blockPos.y();
            cache.blockPos.z = blockPos.z();
            cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
            vmIndex = hashIdx + 1; // add 1 to support legacy true / false operations for isFound

            return voxelData[hashIdx*SDF_BLOCK_SIZE3 + linearIdx];
        }

        if (hashEntry.offset < 1) break;
        hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
    }

    vmIndex = false;
    return TVoxel();
}

inline ITMVoxel_d readVoxel_new_submap_core( std::map<int,DWIO::BlockData>& blocks,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   const THREADPTR(Vector3i) &point,
                                   THREADPTR(int) &vmIndex) {
    Vector3s blockPos;
    int linearIdx = pointToVoxelBlockPosCpu(point, blockPos);
    //根据blockpos计算出hash索引，返回对应的体素

    int hashIdx = hashIndexCPU(blockPos);

    while (true) {
        ITMHashEntry hashEntry = voxelIndex[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) ) {
            vmIndex = hashIdx + 1; 
            return blocks[hashIdx].voxel_data[linearIdx];
        }

        if (hashEntry.offset < 1) break;
        hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
    }

    vmIndex = false;
    return ITMVoxel_d();
}


template<class TVoxel>
__device__ inline TVoxel readVoxelGlobal(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   const THREADPTR(Vector3i) &point,
                                   THREADPTR(int) &vmIndex,
                                   THREADPTR(DWIO::ITMVoxelBlockHash::IndexCache) &cache) {
    Vector3s blockPos;
    int linearIdx = pointToVoxelBlockPos(point, blockPos);
    //根据blockpos计算出hash索引，返回对应的体素

    int hashIdx = hashIndexGlobal(blockPos);
    if IS_EQUAL3(cache.blockPos, blockPos) {
        vmIndex = true;
        return voxelData[hashIdx*SDF_BLOCK_SIZE3 + linearIdx];
    }

    while (true) {
        ITMHashEntry hashEntry = voxelIndex[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) ) {
            cache.blockPos.x = blockPos.x();
            cache.blockPos.y = blockPos.y();
            cache.blockPos.z = blockPos.z();
            cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
            vmIndex = hashIdx + 1; // add 1 to support legacy true / false operations for isFound

            return voxelData[hashIdx*SDF_BLOCK_SIZE3 + linearIdx];
        }

        if (hashEntry.offset < 1) break;
        hashIdx = MAP_BUCKET_NUM + hashEntry.offset - 1;
    }

    vmIndex = false;
    return TVoxel();
}

template<class TVoxel>
inline TVoxel readVoxelGlobal(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   Vector3i point,
                                   THREADPTR(int) &vmIndex) {
    DWIO::ITMVoxelBlockHash::IndexCache cache;
    return readVoxelGlobal(voxelData, voxelIndex, point, vmIndex, cache);
}


inline ITMVoxel_d readVoxel_new_submap( std::map<int,DWIO::BlockData>& blocks,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   Vector3i point,
                                   THREADPTR(int) &vmIndex) {

    return readVoxel_new_submap_core(blocks, voxelIndex, point, vmIndex);
}


template<class TVoxel>
__device__ inline TVoxel readVoxelCpu(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   Vector3i point,
                                   THREADPTR(int) &vmIndex) {
    DWIO::ITMVoxelBlockHash::IndexCache cache;
    return readVoxelCpu(voxelData, voxelIndex, point, vmIndex, cache);
}

template<class TVoxel>
__device__ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   Vector3i point,
                                   THREADPTR(int) &vmIndex) {
    DWIO::ITMVoxelBlockHash::IndexCache cache;
    return readVoxel(voxelData, voxelIndex, point, vmIndex, cache);
}

template<class TVoxel>
__device__ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData,
                                   const CONSTPTR(DWIO::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                   Vector3i point,
                                   THREADPTR(bool) &foundPoint) {
    int vmIndex;
    DWIO::ITMVoxelBlockHash::IndexCache cache;
    TVoxel result = readVoxel(voxelData, voxelIndex, point, vmIndex, cache);
    foundPoint = vmIndex != 0;
    return result;
}

template<class TVoxel, class TIndex>
__device__ inline int16_t readFromSDF_float_uninterpolated(const CONSTPTR(TVoxel) *voxelData,
                                                           const CONSTPTR(TIndex) *voxelIndex,
                                                           Vector3f point,
                                                           THREADPTR(int) &vmIndex) {
    TVoxel res = readVoxel(voxelData, voxelIndex, Vector3i((int) ROUND(point.x()), (int) ROUND(point.y()), (int) ROUND(point.z())), vmIndex);
    return res.tsdf;
}