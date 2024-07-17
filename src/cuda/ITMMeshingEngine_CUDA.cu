// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "../cuda/include/ITMMeshingEngine_CUDA.h"

#include <algorithm>

#include "../cuda/include/ITMMeshingEngine_Shared.h"
#include "../hash/CUDADefines.h"
//#include "../hash/ITMVoxelBlockHash.h"
#include "../hash/ITMScene.h"



using namespace DWIO;

template<class TVoxel>
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries,
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable);

template<int dummy>
__global__ void findAllocateBlocks(Vector4s *visibleBlockGlobalPos, const ITMHashEntry *hashTable, int noTotalEntries)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noTotalEntries - 1) return;

	const ITMHashEntry &currentHashEntry = hashTable[entryId];

	if (currentHashEntry.ptr >= 0) 
		visibleBlockGlobalPos[currentHashEntry.ptr] = Vector4s(currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z, 1);
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel>::ITMMeshingEngine_CUDA(void) 
{
	DWIOcudaSafeCall(cudaMalloc((void**)&visibleBlockGlobalPos_device, (SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE) * sizeof(Vector4s)));//保存实际存在的block位置
	DWIOcudaSafeCall(cudaMalloc((void**)&noTriangles_device, sizeof(unsigned int)));//保存提取到点的数目
	DWIOcudaSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel>::~ITMMeshingEngine_CUDA(void) 
{
	DWIOcudaSafeCall(cudaFree(visibleBlockGlobalPos_device));
	DWIOcudaSafeCall(cudaFree(noTriangles_device));
}

template<class TVoxel>
void ITMMeshingEngine_CUDA<TVoxel>::MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMVoxelBlockHash> *scene,int& TotalTriangles)
{
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CUDA);
	const TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	const ITMHashEntry *hashTable = scene->index.GetEntries();

	int noMaxTriangles = mesh->noMaxTriangles, noTotalEntries = scene->index.noTotalEntries;//所有hash条目的总数2^21+2^17
	float factor = scene->voxel_resolution;//体素大小

	//DWIOcudaSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));//要让他每次累积起来
	DWIOcudaSafeCall(cudaMemset(visibleBlockGlobalPos_device, 0, sizeof(Vector4s) * SDF_LOCAL_BLOCK_NUM));

	{ // identify used voxel blocks
		dim3 cudaBlockSize(256); 
		dim3 gridSize((int)ceil((float)noTotalEntries / (float)cudaBlockSize.x));
		//将hash表中ptr>=0的block的位置存放在visibleBlockGlobalPos_device中，我的目标将ptr>=-1的都放进去，首先先对已经在gpu中的计算出点云，然后将一部分的ptr>-1的移进去
		findAllocateBlocks<-1><<<gridSize, cudaBlockSize>>>(visibleBlockGlobalPos_device, hashTable, noTotalEntries);
	}

	{ // mesh used voxel blocks
		dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		dim3 gridSize(noTotalEntries / 16, 16);
		DWIOcudaSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));
		meshScene_device<TVoxel> << <gridSize, cudaBlockSize >> >(triangles, noTriangles_device, factor, noTotalEntries, noMaxTriangles,
			visibleBlockGlobalPos_device, localVBA, hashTable);

		DWIOcudaSafeCall(cudaMemcpy(&mesh->noTotalTriangles, noTriangles_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // meshScene_device made sure up to noMaxTriangles triangles were copied in the output array but,
    // since the check was performed with atomicAdds, the actual number stored in noTriangles_device
    // might be greater than noMaxTriangles.
    // We coerce it to be lesser or equal to that number, not doing it causes a segfault when using the mesh later.
    mesh->noTotalTriangles = std::min<uint>(mesh->noTotalTriangles, static_cast<uint>(noMaxTriangles));
	TotalTriangles = mesh->noTotalTriangles;
	printf("generate vertex number: %d\n",mesh->noTotalTriangles);
  }
}
//想要加上颜色信息，修改Triabgle带有颜色信息float3 color,然后修改buildVertList中的根据周围8个点插值当前点的tsdf,color好像就行，晚上试试
template<class TVoxel>//计算出顶点
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries, 
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable)
{
	const Vector4s globalPos_4s = visibleBlockGlobalPos[blockIdx.x + gridDim.x * blockIdx.y];

	if (globalPos_4s.w() == 0) return;

	Vector3i globalPos = Vector3i(globalPos_4s.x(), globalPos_4s.y(), globalPos_4s.z()) * SDF_BLOCK_SIZE;
	//这个不知道有没有问题，再看
	Vertex vertList[12];								//block中的体素位置
	int cubeIndex = buildVertList(vertList, globalPos, Vector3i(threadIdx.x, threadIdx.y, threadIdx.z), localVBA, hashTable,factor);

	if (cubeIndex < 0) return;

	for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
	{
		int triangleId = atomicAdd(noTriangles_device, 1);

		if (triangleId < noMaxTriangles - 1)
		{
			triangles[triangleId].p0 = vertList[triangleTable[cubeIndex][i]] ;
			triangles[triangleId].p1 = vertList[triangleTable[cubeIndex][i + 1]];
			triangles[triangleId].p2 = vertList[triangleTable[cubeIndex][i + 2]] ;
		}
	}
}
