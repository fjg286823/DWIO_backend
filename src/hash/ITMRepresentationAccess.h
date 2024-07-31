
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

inline ITMVoxel_d readVoxel_new_submap_core( std::map<int,DWIO::BlockData*>& blocks,
                                   const DWIO::ITMVoxelBlockHash::IndexData *voxelIndex,
                                   const Vector3i &point,int &vmIndex) {
    Vector3s blockPos;
    int linearIdx = pointToVoxelBlockPosCpu(point, blockPos);
    //根据blockpos计算出hash索引，返回对应的体素

    int hashIdx = hashIndexCPU(blockPos);

    while (true) {
        ITMHashEntry hashEntry = voxelIndex[hashIdx];

        if (IS_EQUAL3(hashEntry.pos, blockPos) ) {
            vmIndex = hashIdx + 1; 
            return blocks[hashIdx]->voxel_data[linearIdx];
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


inline ITMVoxel_d readVoxel_new_submap( std::map<int,DWIO::BlockData*>& blocks,
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

inline void TO_INT_FLOOR3(Vector3i& pos,Vector3f& coeff, Vector3f& point)
{
    pos.x() = (int)floor(point.x());
    pos.y() = (int)floor(point.y());
    pos.z() = (int)floor(point.z());
    coeff.x() = point.x() - pos.x();
    coeff.y() = point.y() - pos.y();
    coeff.z() = point.z() - pos.z();
}


template<class TVoxel, class TIndex>
__device__ inline int16_t readFromSDF_float_uninterpolated(const CONSTPTR(TVoxel) *voxelData,
                                                           const CONSTPTR(TIndex) *voxelIndex,
                                                           Vector3f point,
                                                           THREADPTR(int) &vmIndex) {
    TVoxel res = readVoxel(voxelData, voxelIndex, Vector3i((int) ROUND(point.x()), (int) ROUND(point.y()), (int) ROUND(point.z())), vmIndex);
    return res.tsdf;
}

//从子图的8个体素插值出一个tsdf值，这个TCache要改改
inline float readFromSDF_voxel_interpolated(std::map<int,DWIO::BlockData*>& blocks,
	DWIO::ITMVoxelBlockHash::IndexData* voxelIndex, Vector3f point, int& vmIndex, int& maxW)
{
	float res1, res2, v1, v2;
	Vector3f coeff; Vector3i pos; 
    TO_INT_FLOOR3(pos, coeff, point);

	{                       //从局部子图读取
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(0, 0, 0), vmIndex);
		v1 = v.tsdf;
		maxW = v.w_depth;
	}
	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(1, 0, 0), vmIndex);
		v2 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.x()) * v1 + coeff.x() * v2;

	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(0, 1, 0), vmIndex);
		v1 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(1, 1, 0), vmIndex);
		v2 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.y()) * res1 + coeff.y() * ((1.0f - coeff.x()) * v1 + coeff.x() * v2);

	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(0, 0, 1), vmIndex);
		v1 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(1, 0, 1), vmIndex);
		v2 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.x()) * v1 + coeff.x() * v2;

	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(0, 1, 1), vmIndex);
		v1 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const ITMVoxel_d & v = readVoxel_new_submap(blocks, voxelIndex, pos + Vector3i(1, 1, 1), vmIndex);
		v2 = v.tsdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.y()) * res2 + coeff.y() * ((1.0f - coeff.x()) * v1 + coeff.x() * v2);

	vmIndex = true;
	return ITMVoxel_d::valueToFloat((1.0f - coeff.z()) * res1 + coeff.z() * res2);
}



//对一个世界点转到各个子图中得到tsdf值，再加权融合                              //所有子图数据和 hash表数据
inline float readFromSDF_float_interpolated(std::map<uint32_t,DWIO::submap*>&submaps_, const Vector3f & point, int & vmIndex)//最后一个参数告诉上一级找到了
{
	float sum_sdf = 0.0f;
	int sum_weights = 0;
	vmIndex = false;
     
    for( auto& it : submaps_)
    {
    auto& submap = it.second; 
    //转到子图坐标系
    Vector3f point_local =submap->local_rotation.transpose().cast<float>() *(point - submap->local_translation.cast<float>());
    int vmIndex_tmp, maxW;
    float sdf = readFromSDF_voxel_interpolated(submap->blocks_, submap->hashEntries_submap->GetData(MEMORYDEVICE_CPU), 
                        point_local, vmIndex_tmp, maxW);
    if (!vmIndex_tmp) continue;
    vmIndex = true;

    sum_sdf += (float)maxW * sdf;
    sum_weights += maxW;

    }
    if (sum_weights == 0) return 1.0f;

    return (sum_sdf / (float)sum_weights);
}