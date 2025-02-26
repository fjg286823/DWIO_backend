
#include "../include/ITMMeshingEngine_CPU.h"
#include "cuda/include/ITMMeshingEngine_Shared.h"
#include "ITMVoxelBlockHash.h"
#include "ITMRepresentationAccess.h"

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

//for submap
template<class TVoxel>
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
void ITMMeshingEngine_CPU<TVoxel>::MeshScene_global(ITMMesh *mesh, std::map<uint32_t,DWIO::submap*>&submaps_,float factor)
{
	std::shared_ptr<spdlog::logger> mylogger = spdlog::get("spdlog");
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;
	std::cout<<" 子图个数： "<<submaps_.size()<<std::endl;
	int extraList = MAP_EXCESS_LIST_SIZE;//看看换成大的hash会咋样
	//对子图的平移做预处理!
	for(auto& it : submaps_){
		auto& submap = it.second;
		submap->local_translation = submap->local_translation / factor;
	}

	//1、遍历所有子图
	std::cout<<"遍历子图"<<std::endl;
	mylogger->debug("遍历子图");
	mylogger->info("共有 {0}个子图", submaps_.size());
	int nums = 0;

	for(auto& it : submaps_)
	{
		auto& submap = it.second;
		// nums++;
		// if(nums>2)
		// {
		// 	break;
		// }
		std::cout << "extract " << it.first << " submap" << std::endl;
		mylogger->info("extract {0} submap", it.first );
		for (auto &block : submap->blocks_)
		{
			Eigen::Vector3i submap_block_pose = block.second->block_pos.cast<int>() * SDF_BLOCK_SIZE;

			//mylogger->debug("start get triangles");
			// 遍历block
			for (int z = 0; z < SDF_BLOCK_SIZE; z++){
				for (int y = 0; y < SDF_BLOCK_SIZE; y++){
					for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
						Vertex vertList[12];
						int cubeIndex = buildVertListMulti(vertList, submap_block_pose, Vector3i(x, y, z),submaps_, it.first, factor);
						if (cubeIndex < 0) continue;
						for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
						{
							triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
							triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
							triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];

							if (noTriangles < noMaxTriangles - 1) 
								noTriangles++;
							else{
								mylogger->debug("points num is too large!");
								std::cout << "points num is too large!" << std::endl;
							}
						}
						//std::cout << "noTriangles: " << noTriangles << std::endl;
					}
				}
			}
			//mylogger->debug("noTriangles: {0} , noMaxTriangles {1}", noTriangles, noMaxTriangles);
			//mylogger->debug("a block done!");
		}
	}

	mesh->noTotalTriangles = noTriangles;

}

template<class TVoxel>
void ITMMeshingEngine_CPU<TVoxel>::MeshScene_global_hash(ITMMesh *mesh, std::map<uint32_t,DWIO::submap*>&submaps_,float factor){
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);
	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;
	for(auto& it : submaps_){
		auto& submap = it.second;
		submap->local_translation = submap->local_translation / factor;
	}

	//申明一个大的hash表！
	ITMHashEntry *map_hashTable;
	cudaMallocHost((void **)&map_hashTable, (MAP_BUCKET_NUM + MAP_EXCESS_LIST_SIZE) * sizeof(ITMHashEntry));
	ITMHashEntry tmpEntry{};
	memset(&tmpEntry, 0, sizeof(ITMHashEntry));
	tmpEntry.ptr = -2;
	std::fill(map_hashTable, map_hashTable + (MAP_BUCKET_NUM + MAP_EXCESS_LIST_SIZE) , tmpEntry);
	int excessAllocationList = MAP_EXCESS_LIST_SIZE-1;

	for(auto& it : submaps_)
	{
		auto& submap = it.second;

		std::cout << "extract " << it.first << " submap" << std::endl;
		for (auto &block : submap->blocks_)
		{
			Eigen::Vector3i submap_block_pose = block.second->block_pos.cast<int>() * SDF_BLOCK_SIZE;

			//这里执行hash
			Eigen::Vector3d submap_pose;
			submap_pose.x() = (double)submap_block_pose.x() + 0.5;
			submap_pose.y() = (double)submap_block_pose.y() + 0.5;
			submap_pose.z() = (double)submap_block_pose.z() + 0.5;
			Eigen::Vector3d global_pose_temp = submap->local_rotation * submap_pose + submap->local_translation;
			Eigen::Vector3i global_pose(std::round(global_pose_temp.x()),std::round(global_pose_temp.y()),std::round(global_pose_temp.z()));

			Vector3s global_block_pos;
			pointToVoxelBlockPosCpu(global_pose, global_block_pos);
			int globalHashIndex = hashIndexGlobal(global_block_pos);

			ITMHashEntry hashEntry = map_hashTable[globalHashIndex];

			bool isFound =false;
			bool isExtra =false;

			if(IS_EQUAL3(hashEntry.pos, global_block_pos) && hashEntry.ptr >= -1)//初始化这个值都是0，如果查到block为000的就会导致报错！
			{
				isFound =true;
			}
			if(!isFound)//是否在额外链表中
			{
				if(hashEntry.ptr >= -1){

					while(hashEntry.offset >= 1){
						globalHashIndex = MAP_BUCKET_NUM + hashEntry.offset - 1;
						hashEntry = map_hashTable[globalHashIndex];
						if(IS_EQUAL3(hashEntry.pos, global_block_pos)&&hashEntry.ptr>=-1)
						{
							isFound = true;
							break;
						}
					}
					isExtra =true;//用来表示是否是在额外列表区域没找到
				}
			}

			if(isFound){
				continue;
			}else{
				if(isExtra)//hash条目要创建在额外列表的地方
				{ 
					ITMHashEntry hashEntry_temp{};
					hashEntry_temp.pos.x = global_block_pos.x();
					hashEntry_temp.pos.y = global_block_pos.y();
					hashEntry_temp.pos.z = global_block_pos.z();
					hashEntry_temp.ptr = -1;
					int exlOffset =-1;
					if (excessAllocationList > 0)
					{
						exlOffset = excessAllocationList;
						excessAllocationList--;
					}else{
						std::cout << " extra list don't have pose" << std::endl;
					}

					if(exlOffset < 0)//如果额外链表不够了就跳过这个block的处理
						continue;
					
					map_hashTable[globalHashIndex].offset = exlOffset + 1; 
					map_hashTable[MAP_BUCKET_NUM + exlOffset] = hashEntry_temp; 
				}
				else{
					//顺序部分插入
					ITMHashEntry hashEntry_temp{};
					hashEntry_temp.pos.x = global_block_pos.x();
					hashEntry_temp.pos.y = global_block_pos.y();
					hashEntry_temp.pos.z = global_block_pos.z();
					hashEntry_temp.ptr = -1;
					hashEntry_temp.offset = 0;

					map_hashTable[globalHashIndex] = hashEntry_temp;
					//std::cout<<"generate a hashEntry"<<std::endl;
				}
			}



			for (int z = 0; z < SDF_BLOCK_SIZE; z++){
				for (int y = 0; y < SDF_BLOCK_SIZE; y++){
					for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
						Vertex vertList[12];
						int cubeIndex = buildVertListMulti(vertList, submap_block_pose, Vector3i(x, y, z),submaps_, it.first, factor);
						if (cubeIndex < 0) continue;
						for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
						{
							triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
							triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
							triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];

							if (noTriangles < noMaxTriangles - 1) 
								noTriangles++;
							else{
								std::cout << "points num is too large!" << std::endl;
							}
						}
						//std::cout << "noTriangles: " << noTriangles << std::endl;
					}
				}
			}

			// // 遍历block
			// for (int z = 0; z < SDF_BLOCK_SIZE; z++){
			// 	for (int y = 0; y < SDF_BLOCK_SIZE; y++){
			// 		for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
			// 			Vertex vertList[12];
			// 			int cubeIndex = buildVertListMultiBox(vertList, global_block_pos.cast<int>(), Vector3i(x, y, z),submaps_, factor);
			// 			if (cubeIndex < 0) continue;
			// 			for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
			// 			{
			// 				triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
			// 				triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
			// 				triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];

			// 				if (noTriangles < noMaxTriangles - 1) 
			// 					noTriangles++;
			// 				else{
			// 					std::cout << "points num is too large!" << std::endl;
			// 				}
			// 			}
						
			// 		}
			// 	}
			// }

		}
	}

	mesh->noTotalTriangles = noTriangles;
}

template<class TVoxel>
void ITMMeshingEngine_CPU<TVoxel>::MeshScene_global_Box(ITMMesh *mesh, std::map<uint32_t,DWIO::submap*>&submaps_,float factor) {
	std::shared_ptr<spdlog::logger> mylogger = spdlog::get("spdlog");
	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);

	int noTriangles = 0, noMaxTriangles = mesh->noMaxTriangles;
	std::cout<<" 子图个数： "<<submaps_.size()<<std::endl;

	//对子图的平移做预处理!
	for(auto& it : submaps_){
		auto& submap = it.second;
		submap->local_translation = submap->local_translation / factor;
	}


	for(int g_z = -50;g_z<=50;g_z++) {
		for(int g_y = -50;g_y<=50;g_y++) {
			for(int g_x = -50;g_x<=50;g_x++) {

				Eigen::Vector3i global_block_pos(g_x* SDF_BLOCK_SIZE,g_y* SDF_BLOCK_SIZE,g_z* SDF_BLOCK_SIZE);
				for (int z = 0; z < SDF_BLOCK_SIZE; z++) {
					for (int y = 0; y < SDF_BLOCK_SIZE; y++) {
						for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
							Vertex vertList[12];
							int cubeIndex = buildVertListMultiBox(vertList, global_block_pos, Vector3i(x, y, z),submaps_, factor);

							if (cubeIndex < 0) continue;
							for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
							{
								triangles[noTriangles].p0 = vertList[triangleTable[cubeIndex][i]] ;
								triangles[noTriangles].p1 = vertList[triangleTable[cubeIndex][i + 1]];
								triangles[noTriangles].p2 = vertList[triangleTable[cubeIndex][i + 2]];

								if (noTriangles < noMaxTriangles - 1)
									noTriangles++;
								else{
									mylogger->debug("points num is too large!");
								}
							}
						}
					}
				}

			}
		}
	}

	mesh->noTotalTriangles = noTriangles;

}