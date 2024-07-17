
#include "../include/ITMMeshingEngine_CPU.h"
#include "cuda/include/ITMMeshingEngine_Shared.h"

using namespace DWIO;

template<class TVoxel>
void ITMMeshingEngine_CPU<TVoxel>::MeshScene(ITMMesh *mesh, ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);
	//const TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();//改成global中的体素数据
    const TVoxel *globalVBA = scene->globalCache->GetVoxelData();
	ITMHashEntry *hashTable = scene->index.GetEntriesCpu();//报错是因为scene被const修饰，则调用的函数不能修改对象里的属性状态

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles, noTotalEntries = scene->index.noTotalEntries;
	float factor = scene->voxel_resolution;

	mesh->triangles->Clear();
    //printf("1\n");
	for (int entryId = 0; entryId < noTotalEntries; entryId++)
	{
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[entryId];//是因为hash表是gpu的吧
		if (currentHashEntry.ptr <-1) continue;//先只生成cpu上的地图,这里出的问题

        globalPos.x() = currentHashEntry.pos.x * SDF_BLOCK_SIZE;
        globalPos.y() = currentHashEntry.pos.y * SDF_BLOCK_SIZE;
        globalPos.z() = currentHashEntry.pos.z * SDF_BLOCK_SIZE;

		//printf("1.2\n");
		for (int z = 0; z < SDF_BLOCK_SIZE; z++) {
			for (int y = 0; y < SDF_BLOCK_SIZE; y++) {
				for (int x = 0; x < SDF_BLOCK_SIZE; x++)
				{
					Vertex vertList[12];
					//printf("entryIdx:%d,ptr:%d\n",entryId,currentHashEntry.ptr);
					//int cubeIndex = buildVertList(vertList, globalPos, Vector3i(x, y, z), localVBA, hashTable);
					int cubeIndex = buildVertListCpu(vertList, globalPos, Vector3i(x, y, z), globalVBA, hashTable,factor);
					
					if (cubeIndex < 0) continue;

					for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
					{
						triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
						triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
						triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];

						if (noTriangles < noMaxTriangles - 1) noTriangles++;
					}
				}
			}
		}
	}

	mesh->noTotalTriangles = noTriangles;
}