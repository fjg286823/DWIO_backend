
#pragma once

#include <Eigen/Dense>

#include "MemoryBlock.h"
#include "ITMVoxelBlockHash.h"
#include "ITMVoxelTypes.h"

namespace DWIO
{
    struct BlockData{
        BlockData(){
            voxel_data =  (ITMVoxel_d *) malloc(sizeof(ITMVoxel_d) * SDF_BLOCK_SIZE3);
            block_pos.x() = 32765;
            block_pos.y() = 32765;
            block_pos.z() = 32765;
        }
        ~BlockData(){
            free(voxel_data);
        }

        ITMVoxel_d* voxel_data;
        Vector3s block_pos;
    };


    /*这种子图的方式更加节省体素占用的cpu内存空间，且使用的都是一个hash函数，不会有潜在的风险（由于不同hash函数带来计算的索引不一致导致的）
    */
    class submap{//这个子图结构还能减少，不保存对应的hash表，只blocks_就够用了，有位置和hash索引,大大减少内存啊！
    public:
        u_int32_t submap_id;
        int noTotalEntries;
        Eigen::Matrix4d submap_pose;

        Eigen::Matrix3d local_rotation;
        Eigen::Vector3d local_translation;

        std::map<int,BlockData*> blocks_;
        DWIO::MemoryBlock<ITMHashEntry> *hashEntries_submap;
        submap()
        {
            submap_id = -1;
            noTotalEntries =0;
            submap_pose.setIdentity();
        }
        ~submap()
        {
            hashEntries_submap->Free();
            delete hashEntries_submap;
            blocks_.clear();
        }
        submap(Eigen::Matrix4d pose_,u_int32_t id_,int bucket_nums,int extra_nums)
        {
            submap_id = id_;
            submap_pose = pose_;  
            local_rotation = pose_.block(0,0,3,3); 
            local_translation = pose_.block(0,3,3,1);
            noTotalEntries = bucket_nums + extra_nums;
            hashEntries_submap = new DWIO::MemoryBlock<ITMHashEntry>(noTotalEntries, MEMORYDEVICE_CPU);   
            ITMHashEntry tmpEntry{};
            memset(&tmpEntry, 0, sizeof(ITMHashEntry));
            tmpEntry.ptr = -2;     
            std::fill(hashEntries_submap->GetData(MEMORYDEVICE_CPU), hashEntries_submap->GetData(MEMORYDEVICE_CPU) + noTotalEntries , tmpEntry);
        }

        void genereate_blocks(ITMVoxel_d* voxel_datas)
        {
            ITMHashEntry* submap_hashTable = hashEntries_submap->GetData(MEMORYDEVICE_CPU);
            for(int i=0;i<noTotalEntries;i++)
            {
                if(submap_hashTable[i].ptr<-1) continue;

                BlockData* block_data = new BlockData();
                block_data->block_pos.x() = submap_hashTable[i].pos.x;
                block_data->block_pos.y() = submap_hashTable[i].pos.y;
                block_data->block_pos.z() = submap_hashTable[i].pos.z;
                memcpy(block_data->voxel_data, voxel_datas + i*SDF_BLOCK_SIZE3,sizeof(ITMVoxel_d)*SDF_BLOCK_SIZE3);
                blocks_[i] = block_data;
            }
            std::cout<<"generate :"<<blocks_.size()<<" blocks data"<<std::endl;
        }

        ITMVoxel_d* GetVoxel(int index)
        {
            ITMVoxel_d* res = blocks_[index]->voxel_data;
            return res;
        }
    };


    /*使用这钟子图的结构需要，把SDF_BUCKET_NUM改小，同时改变该生成子图的逻辑，还得改三角化点的最大个数限制*/
    class Submap{
    public:

        static const int voxelBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
        Eigen::Matrix4d submap_pose;
        u_int32_t submap_id;
        int excessAllocationList = 0;

        Submap()
        {
            submap_pose.setIdentity();
            submap_id = -1;
            noTotalEntries =0;
        }
        ~Submap()
        {
            hashEntries_submap->Free();
            free(storedVoxelBlocks_submap);
        }

        Submap(Eigen::Matrix4d pose_,u_int32_t id_,int bucket_nums,int extra_nums)
        {            
            submap_id = id_;
            submap_pose = pose_;
            excessAllocationList = extra_nums-1;//这里需要确定一下，就是要减一
            noTotalEntries =bucket_nums + extra_nums;
            hashEntries_submap = new DWIO::MemoryBlock<ITMHashEntry>(noTotalEntries, MEMORYDEVICE_CPU);
            storedVoxelBlocks_submap = (ITMVoxel_d *) malloc(noTotalEntries * sizeof(ITMVoxel_d) * SDF_BLOCK_SIZE3);
            ITMHashEntry tmpEntry{};
            memset(&tmpEntry, 0, sizeof(ITMHashEntry));
            tmpEntry.ptr = -2;
            std::fill(hashEntries_submap->GetData(MEMORYDEVICE_CPU), hashEntries_submap->GetData(MEMORYDEVICE_CPU) + noTotalEntries , tmpEntry);
            std::fill(storedVoxelBlocks_submap, storedVoxelBlocks_submap + (noTotalEntries * SDF_BLOCK_SIZE3), ITMVoxel_d());

        }

        //只需要位姿、hash表和体素数据，只是用来最后提取体素数据，其实
        DWIO::MemoryBlock<ITMHashEntry> *hashEntries_submap;
        ITMVoxel_d* storedVoxelBlocks_submap;
        int noTotalEntries;
        //还可能附带一个生成的描述子信息！用于回环检测

        int GetExtraListPos()
        {
            int pose =-1;
            if(excessAllocationList>=0){
                pose = excessAllocationList;
                excessAllocationList--;
            }
            return pose;
        }

        ITMVoxel_d* GetVoxel(int index)
        {
            ITMVoxel_d* res = &(storedVoxelBlocks_submap[index*SDF_BLOCK_SIZE3]);
            return res;
        }
    };
}