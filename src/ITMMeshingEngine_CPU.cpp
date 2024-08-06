
#include "../include/ITMMeshingEngine_CPU.h"
#include "cuda/include/ITMMeshingEngine_Shared.h"

using namespace DWIO;

inline void transformPoint(Vertex& P,Eigen::Matrix4f& Trans)
{
	Vector4f P0;
	P0(0) = P.p.x;
	P0(1) = P.p.y;
	P0(2) = P.p.z;
	P0(3) = 1.0f;	
	P0 = Trans * P0;
	P.p.x = P0(0);
	P.p.y = P0(1);
	P.p.z = P0(2);
}

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

template<class TVoxel>//for submap
void ITMMeshingEngine_CPU<TVoxel>::MeshScene(ITMMesh *mesh, std::map<int,DWIO::BlockData*>& blocks , ITMHashEntry* hashTable,
		int noTotalEntries,float factor,Eigen::Matrix4f Trans)
{
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;

	mesh->triangles->Clear();
	for (int entryId = 0; entryId < noTotalEntries; entryId++)
	{
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[entryId];
		if (currentHashEntry.ptr <-1) continue;

        globalPos.x() = currentHashEntry.pos.x * SDF_BLOCK_SIZE;
        globalPos.y() = currentHashEntry.pos.y * SDF_BLOCK_SIZE;
        globalPos.z() = currentHashEntry.pos.z * SDF_BLOCK_SIZE;

		for (int z = 0; z < SDF_BLOCK_SIZE; z++) {
			for (int y = 0; y < SDF_BLOCK_SIZE; y++) {
				for (int x = 0; x < SDF_BLOCK_SIZE; x++)
				{
					Vertex vertList[12];                                                        //传入voxel
					int cubeIndex = buildVertList_new_submap(vertList, globalPos, Vector3i(x, y, z), blocks , hashTable,factor);
					
					if (cubeIndex < 0) continue;

					for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
					{
						triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
						triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
						triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];
						transformPoint(triangles[noTriangles].p0,Trans);
						transformPoint(triangles[noTriangles].p1,Trans);
						transformPoint(triangles[noTriangles].p2,Trans);


						if (noTriangles < noMaxTriangles - 1) noTriangles++;
					}
				}
			}
		}
	}

	mesh->noTotalTriangles = noTriangles;
}


template<class TVoxel>//for Submap
void ITMMeshingEngine_CPU<TVoxel>::MeshScene(ITMMesh *mesh, TVoxel* globalVBA , ITMHashEntry* hashTable,
		int noTotalEntries,float factor)
{
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;

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

template<class TVoxel>//for Globalmap
void ITMMeshingEngine_CPU<TVoxel>::MeshScene_global(ITMMesh *mesh, std::map<uint32_t,DWIO::submap*>&submaps_,float factor,ITMHashEntry* global_hashTable)
{
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;

	int extraList = SDF_EXCESS_LIST_SIZE;

	//1、遍历所有子图
	std::cout<<"遍历子图"<<std::endl;
	int nums=0;
	for(auto& it : submaps_)
	{
		auto& submap = it.second;
		nums++;
		// if(nums>2)
		// {
		// 	break;
		// }
		submap->local_translation = submap->local_translation / factor;

		for(auto& block : submap->blocks_)
		{
			//不能每个重复的block被提取，要用一个global hash来表示然后去除重复的
			//每次开始前先进行查找，查找有了就跳过，没有就插入对应的hash条目，这个hash条目可以被设计的小一点这样就可以避免重复的提取了
			Eigen::Vector3i submap_pose = block.second->block_pos.cast<int>() * SDF_BLOCK_SIZE;
			std::cout << "1" << std::endl;
			Eigen::Vector3f global_pose = submap->local_rotation * submap_pose.cast<float>() + submap->local_translation;
			std::cout<<"2"<<std::endl;
			Eigen::Vector3i global_pose_i;
			global_pose_i.x() = std::round(global_pose.x() / SDF_BLOCK_SIZE);
			global_pose_i.y() = std::round(global_pose.y() / SDF_BLOCK_SIZE);
			global_pose_i.z() = std::round(global_pose.z() / SDF_BLOCK_SIZE);
			int globalHashIndex = hashIndexCPU(global_pose_i);
			std::cout<<"3"<<std::endl;
            ITMHashEntry hashEntry = global_hashTable[globalHashIndex];
			bool isFound =false;
			bool isExtra =false;

			if(IS_EQUAL3(hashEntry.pos, global_pose_i) && hashEntry.ptr >= -1)
			{
				isFound =true;
			}
			if(!isFound)//是否在额外链表中
			{
				if(hashEntry.ptr >= -1){

					while(hashEntry.offset >= 1){
						globalHashIndex = SDF_BUCKET_NUM + hashEntry.offset - 1;
						hashEntry = global_hashTable[globalHashIndex];
						if(IS_EQUAL3(hashEntry.pos, global_pose_i))
						{
							isFound = true;
							break;
						}
					}
					isExtra =true;//用来表示是否是在额外列表区域没找到
				}
			}
			std::cout<<"4"<<std::endl;
			if(isFound)
				continue;
			else{
				if(isExtra)//hash条目要创建在额外列表的地方
				{
					ITMHashEntry hashEntry_temp{};
					hashEntry_temp.pos.x = global_pose_i.x();
					hashEntry_temp.pos.y = global_pose_i.y();
					hashEntry_temp.pos.z = global_pose_i.z();
					hashEntry_temp.ptr = -1;

					int exlOffset = extraList--;
					if(exlOffset < 0)//如果额外链表不够了就跳过这个block的处理
						continue;
					
					global_hashTable[globalHashIndex].offset = exlOffset + 1; 
					global_hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry_temp; 
				}
				else{
					//顺序部分插入
					ITMHashEntry hashEntry_temp{};
					hashEntry_temp.pos.x = global_pose_i.x();
					hashEntry_temp.pos.y = global_pose_i.y();
					hashEntry_temp.pos.z = global_pose_i.z();
					hashEntry_temp.ptr = -1;
					hashEntry_temp.offset = 0;

					global_hashTable[globalHashIndex] = hashEntry_temp;
				}
			}

			std::cout<<"5"<<std::endl;
			//遍历block
			for (int z = 0; z < SDF_BLOCK_SIZE; z++){
				for (int y = 0; y < SDF_BLOCK_SIZE; y++){
					for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
						Vertex vertList[12];
						int cubeIndex = buildVertListMulti(vertList, submap_pose, Vector3i(x, y, z),submaps_, it.first, factor);
						if (cubeIndex < 0) continue;
						//std::cout<<"get a triangles!"<<"nototal: "<<noTriangles<<std::endl;
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
	}

	mesh->noTotalTriangles = noTriangles;

}