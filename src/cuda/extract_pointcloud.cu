#include "include/common.h"
#include "../hash/ITMScene.h"
#include "../hash/ITMVoxelTypes.h"
#include "../hash/ITMRepresentationAccess.h"

namespace DWIO {
    namespace internal {
        namespace cuda {

            __global__
            void extract_points_kernel(const PtrStep<short> tsdf_volume, const PtrStep<short> weight_volume, const PtrStep<uchar3> color_volume,
                                       const int3 volume_size, const float voxel_scale,
                                       PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color, int *point_num) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume_size.x - 1 || y >= volume_size.y - 1)
                    return;

                for (int z = 0; z < volume_size.z - 1; ++z) {

                    const float tsdf = static_cast<float>(tsdf_volume.ptr(z * volume_size.y + y)[x]) * DIVSHORTMAX;
                    if (fabs(tsdf - 1) < 1e-5 || tsdf <= -0.99f || tsdf >= 0.99f)
                        continue;

                    short vx = tsdf_volume.ptr((z) * volume_size.y + y)[x + 1];
                    short vy = tsdf_volume.ptr((z) * volume_size.y + y + 1)[x];
                    short vz = tsdf_volume.ptr((z + 1) * volume_size.y + y)[x];


                    short w_vx = weight_volume.ptr((z) * volume_size.y + y)[x + 1];
                    short w_vy = weight_volume.ptr((z) * volume_size.y + y + 1)[x];
                    short w_vz = weight_volume.ptr((z + 1) * volume_size.y + y)[x];


                    if (w_vx <= 0 || w_vy <= 0 || w_vz <= 0)
                        continue;

                    const float tsdf_x = static_cast<float>(vx) * DIVSHORTMAX;
                    const float tsdf_y = static_cast<float>(vy) * DIVSHORTMAX;
                    const float tsdf_z = static_cast<float>(vz) * DIVSHORTMAX;

                    const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
                    const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
                    const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));

                    if (is_surface_x || is_surface_y || is_surface_z) {
                        Eigen::Vector3f normal;
                        normal.x() = (tsdf_x - tsdf);
                        normal.y() = (tsdf_y - tsdf);
                        normal.z() = (tsdf_z - tsdf);
                        if (normal.norm() == 0)
                            continue;
                        normal.normalize();

                        int count = 0;
                        if (is_surface_x) count++;
                        if (is_surface_y) count++;
                        if (is_surface_z) count++;
                        int index = atomicAdd(point_num, count);

                        Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                         (static_cast<float>(y) + 0.5f) * voxel_scale,
                                         (static_cast<float>(z) + 0.5f) * voxel_scale);
                        if (is_surface_x) {
                            position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                        if (is_surface_y) {
                            position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                        if (is_surface_z) {
                            position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                    }
                }
            }

            __device__ void extractSurfaceVertex(Vector3i blockPos,Vector3i localPos,ITMVoxel_d *localVBA,ITMHashEntry *hashTable,  const float voxel_scale,
                        PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color,int *point_num)
            {
                Vector3i globalPos = blockPos + localPos;//最初是这个东西没写，所以出问题
                int vmIndex =0;
                ITMVoxel_d voxel = readVoxel(localVBA,hashTable,globalPos,vmIndex);
                float tsdf =  ITMVoxel_d::valueToFloat(voxel.tsdf);
                if (fabs(tsdf - 1) < 1e-5 || tsdf <= -0.99f || tsdf >= 0.99f){
                    return;
                }
                Vector3i globalPos_x(globalPos.x()+1,globalPos.y(),globalPos.z());
                Vector3i globalPos_y(globalPos.x(),globalPos.y()+1,globalPos.z());
                Vector3i globalPos_z(globalPos.x(),globalPos.y(),globalPos.z()+1);
                ITMVoxel_d voxel_x = readVoxel(localVBA,hashTable,globalPos_x,vmIndex);
                ITMVoxel_d voxel_y = readVoxel(localVBA,hashTable,globalPos_y,vmIndex);
                ITMVoxel_d voxel_z = readVoxel(localVBA,hashTable,globalPos_z,vmIndex);

                short w_vx = voxel_x.w_depth;
                short w_vy = voxel_y.w_depth;
                short w_vz = voxel_z.w_depth;
                if (w_vx <= 0 || w_vy <= 0 || w_vz <= 0)
                    return;
                float tsdf_x = ITMVoxel_d::valueToFloat(voxel_x.tsdf);
                float tsdf_y = ITMVoxel_d::valueToFloat(voxel_y.tsdf);
                float tsdf_z = ITMVoxel_d::valueToFloat(voxel_z.tsdf);
                const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
                const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
                const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));
                if (is_surface_x || is_surface_y || is_surface_z) {
                    Eigen::Vector3f normal;
                    normal.x() = (tsdf_x - tsdf);
                    normal.y() = (tsdf_y - tsdf);
                    normal.z() = (tsdf_z - tsdf);
                    if (normal.norm() == 0)
                        return;
                    normal.normalize();

                    int count = 0;
                    if (is_surface_x) count++;
                    if (is_surface_y) count++;
                    if (is_surface_z) count++;
                    int index = atomicAdd(point_num, count);

                    Vec3fda position((static_cast<float>(globalPos.x()) + 0.5f) * voxel_scale,
                                        (static_cast<float>(globalPos.y()) + 0.5f) * voxel_scale,
                                        (static_cast<float>(globalPos.z()) + 0.5f) * voxel_scale);
                    if (is_surface_x) {
                        position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;

                        vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};
                        normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                        color.ptr(0)[index] = voxel.clr;//这里可能会出问题
                        index++;
                    }
                    if (is_surface_y) {
                        position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;

                        vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                        normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                        color.ptr(0)[index] = voxel.clr;
                        index++;
                    }
                    if (is_surface_z) {
                        position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;

                        vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                        normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                        color.ptr(0)[index] = voxel.clr;
                        index++;
                    }
                }
                
            }
            __global__
            void extract_points_device(ITMHashEntry *hashTable,int noTotalEntries,ITMVoxel_d *localVBA,const float voxel_scale,
                PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color,int *point_num)
            {
                int targetIdx = blockIdx.x + blockIdx.y * gridDim.x;
                if(targetIdx > noTotalEntries-1)
                    return;
                ITMHashEntry  hashentry = hashTable[targetIdx];
                if(hashentry.ptr <= 0){
                    return;
                }

                Vector3i blockPos(hashentry.pos.x*SDF_BLOCK_SIZE,hashentry.pos.y*SDF_BLOCK_SIZE,hashentry.pos.z*SDF_BLOCK_SIZE);
                Vector3i localPos(threadIdx.x, threadIdx.y, threadIdx.z);
                extractSurfaceVertex(blockPos,localPos,localVBA,hashTable,voxel_scale,vertices,normals,color,point_num);

            }

            Cloud extract_points(const VolumeData &volume, const int buffer_size) {
                CloudData cloud_data{buffer_size};

                dim3 threads(32, 32);
                dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                            (volume.volume_size.y + threads.y - 1) / threads.y);

                extract_points_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.weight_volume, volume.color_volume,
                                                           volume.volume_size,
                                                           volume.voxel_scale,
                                                           cloud_data.vertices, cloud_data.normals, cloud_data.color,
                                                           cloud_data.point_num);

                cudaDeviceSynchronize();
                cloud_data.download();

                return Cloud{cloud_data.host_vertices, cloud_data.host_normals,
                             cloud_data.host_color, cloud_data.host_point_num};
            }

            Cloud extract_points_hash(ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene){

                ITMHashEntry *hashTable = scene->index.GetEntries();
                int noTotalEntries = scene->index.noTotalEntries;
                CloudData cloud_data(noTotalEntries*3);
                float voxel_scale = scene->voxel_resolution;
                ITMVoxel_d *localVBA = scene->localVBA.GetVoxelBlocks();
                //遍历hashTable，ptr > 0遍历其中的小体素，先检查周围8个体素是否有值，没值直接返回不提取，有值开始做提取点操作
                dim3 threads(SDF_BLOCK_SIZE,SDF_BLOCK_SIZE,SDF_BLOCK_SIZE);
                dim3 blocks((noTotalEntries+32-1)/32,32);
                //printf("start extract points\n");
                //因为遍历了所有的block
                extract_points_device<<<blocks,threads>>>(hashTable,noTotalEntries,localVBA,voxel_scale,cloud_data.vertices, cloud_data.normals, cloud_data.color,cloud_data.point_num);
                cudaDeviceSynchronize();

                cloud_data.download();
                return Cloud{cloud_data.host_vertices, cloud_data.host_normals,
                             cloud_data.host_color, cloud_data.host_point_num};
            }

        }
    }
}