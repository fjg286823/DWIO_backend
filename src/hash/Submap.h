
#include <Eigen/Dense>

#include "MemoryBlock.h"
#include "ITMVoxelBlockHash.h"
#include "ITMVoxelTypes.h"

namespace DWIO
{
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
            if(excessAllocationList>0){
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