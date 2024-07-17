#include "include/common.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include "../../hash/ITMRepresentationAccess.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf31fa = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace DWIO {
    namespace internal {
        namespace cuda {

            __global__
            void CSM_kernel(const ITMVoxel_d *voxelData,
                            const ITMHashEntry *hashTable,
                            const PtrStep<float3> vertex_map_current,
                            const PtrStep<float> parallel_label,
                            PtrStep<int> search_value,
                            PtrStep<int> search_count,
                            const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,
                            const Matf31fa translation_current,
                            const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_previous_inv,
                            const Matf31fa translation_previous,
                            const PtrStep<float> candidates,
                            const CameraConfiguration cam_params,
                            const float voxel_scale,
                            const int num_candidate,
                            const int cols,
                            const int rows,
                            const int level,
                            const int level_index) {
                const int p = blockIdx.x * blockDim.x + threadIdx.x;
                const int x = (blockIdx.y * blockDim.y + threadIdx.y) * level + level_index;
                const int y = (blockIdx.z * blockDim.z + threadIdx.z) * level + level_index;

                if (x >= cols || y >= rows || p >= num_candidate) {
                    return;
                }

                if (parallel_label.ptr(y)[x] > 0.1)
                    return;

                Matf31fa vertex_current;
                vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                Matf31fa vertex_current_global = rotation_current * vertex_current;

                const float t_x = candidates.ptr(p)[0];
                const float t_y = candidates.ptr(p)[1];
                const float t_z = candidates.ptr(p)[2];

                const float q0 = candidates.ptr(p)[3];
                const float q1 = candidates.ptr(p)[4];
                const float q2 = candidates.ptr(p)[5];
                const float q3 = candidates.ptr(p)[6];

                float q_w = -(vertex_current_global.x() * q1 + vertex_current_global.y() * q2 + vertex_current_global.z() * q3);
                float q_x = q0 * vertex_current_global.x() - q3 * vertex_current_global.y() + q2 * vertex_current_global.z();
                float q_y = q3 * vertex_current_global.x() + q0 * vertex_current_global.y() - q1 * vertex_current_global.z();
                float q_z = -q2 * vertex_current_global.x() + q1 * vertex_current_global.y() + q0 * vertex_current_global.z();

                vertex_current_global.x() = q_x * q0 + q_w * (-q1) - q_z * (-q2) + q_y * (-q3) + t_x + translation_current.x();
                vertex_current_global.y() = q_y * q0 + q_z * (-q1) + q_w * (-q2) - q_x * (-q3) + t_y + translation_current.y();
                vertex_current_global.z() = q_z * q0 - q_y * (-q1) + q_x * (-q2) + q_w * (-q3) + t_z + translation_current.z();

                const Matf31fa vertex_current_camera = rotation_previous_inv * (vertex_current_global - translation_previous);

                Eigen::Vector2i point;
                point.x() = __float2int_rd
                        (vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() + cam_params.principal_x + 0.5f);
                point.y() = __float2int_rd
                        (vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() + cam_params.principal_y + 0.5f);

                if (vertex_current_camera.z() >= 0) {
                    Vector3f grid = (vertex_current_global) / voxel_scale;
                    int vmIndex = 0;
                    int tsdf = static_cast<int>(readFromSDF_float_uninterpolated(voxelData, hashTable, grid, vmIndex));
                    atomicAdd_system(search_value + p, abs(tsdf));
                    atomicAdd_system(search_count + p, 1);
                }
            }

            void CSM(const ITMVoxel_d *voxelData,
                     const ITMHashEntry *hashTable,
                     const TransformCandidates &candidates,
                     SearchData &search_data, float voxel_scale,
                     const Eigen::Matrix3d &rotation_current,
                     const Matf31da &translation_current,
                     const cv::cuda::GpuMat &vertex_map_current,
                     const cv::cuda::GpuMat &parallel_label,
                     const Eigen::Matrix3d &rotation_previous_inv,
                     const Matf31da &translation_previous,
                     const CameraConfiguration &cam_params,
                     const int num_candidate,
                     const int level,
                     const int level_index) {

                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1);
                dim3 grid(1, 1, 1);
                grid.x = static_cast<unsigned int>(std::ceil((float) num_candidate / block.y / block.x));
                grid.y = static_cast<unsigned int>(std::ceil((float) cols / level));
                grid.z = static_cast<unsigned int>(std::ceil((float) rows / level));

                search_data.gpu_search_count.setTo(0);
                search_data.gpu_search_value.setTo(0);
                CSM_kernel<<<grid, block>>>(voxelData, hashTable, vertex_map_current, parallel_label,
                                            search_data.gpu_search_value,
                                            search_data.gpu_search_count,
                                            rotation_current.cast<float>(), translation_current.cast<float>(),
                                            rotation_previous_inv.cast<float>(),
                                            translation_previous.cast<float>(), candidates.q_gpu,
                                            cam_params, voxel_scale, num_candidate,
                                            cols, rows, level, level_index);
                search_data.gpu_search_count.download(search_data.search_count);
                search_data.gpu_search_value.download(search_data.search_value);

                cudaDeviceSynchronize();

            }
        }
    }
}