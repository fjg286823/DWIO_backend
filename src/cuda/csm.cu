#include "include/common.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include "../../hash/ITMRepresentationAccess.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
using Matf31fa = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matf61fa = Eigen::Matrix<float, 6, 1, Eigen::DontAlign>;

namespace DWIO
{
    namespace internal
    {
        namespace cuda
        {

            __global__ void CSM_kernel(const ITMVoxel_d *voxelData,
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
                                       const int level_index)
            {
                const int p = blockIdx.x * blockDim.x + threadIdx.x;
                const int x = (blockIdx.y * blockDim.y + threadIdx.y) * level + level_index;
                const int y = (blockIdx.z * blockDim.z + threadIdx.z) * level + level_index;

                if (x >= cols || y >= rows || p >= num_candidate)
                {
                    return;
                }

                // if (parallel_label.ptr(y)[x] > 0.1)
                //     return;

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
                point.x() = __float2int_rd(vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() + cam_params.principal_x + 0.5f);
                point.y() = __float2int_rd(vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() + cam_params.principal_y + 0.5f);

                if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows && vertex_current_camera.z() >= 0)
                {
                    Vector3f grid = (vertex_current_global) / voxel_scale;
                    int vmIndex = 0;
                    int tsdf = static_cast<int>(readFromSDF_float_uninterpolated(voxelData, hashTable, grid, vmIndex));
                    atomicAdd_system(search_value + p, abs(tsdf));
                    atomicAdd_system(search_count + p, 1);
                }
            }

            __global__ void particle_kernel(const ITMVoxel_d *voxelData, const ITMHashEntry *hashTable,
                                            const PtrStep<float3> vertex_map_current,
                                            const PtrStep<float3> normal_map_current, PtrStep<int> search_value,
                                            PtrStep<int> search_count, const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,
                                            const Matf31fa translation_current, const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_previous_inv,
                                            const Matf31fa translation_previous, const PtrStep<float> quaternion_trans,
                                            const CameraConfiguration cam_params,
                                            const float voxel_scale, const int particle_size, const int cols,
                                            const int rows, const Matf61da search_size, const int level, const int level_index)
            {
                const int p = blockIdx.x * blockDim.x + threadIdx.x;
                const int x = (blockIdx.y * blockDim.y + threadIdx.y) * level + level_index;
                const int y = (blockIdx.z * blockDim.z + threadIdx.z) * level + level_index;
                if (x >= cols || y >= rows || p >= particle_size)
                {
                    return;
                }
                Matf31fa vertex_current;
                vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                vertex_current.z() = vertex_map_current.ptr(y)[x].z;
                Matf31fa vertex_current_global = rotation_current * vertex_current;
                // if(std::abs(vertex_current_global.x())> 0.0 || std::abs(vertex_current_global.y())> 0.0|| std::abs(vertex_current_global.z()) > 0.0)
                //     printf("%f %f %f\n",vertex_current_global.x(),vertex_current_global.y(),vertex_current_global.z());

                const float t_x = quaternion_trans.ptr(p)[0] * search_size(0, 0) * 1000;
                const float t_y = quaternion_trans.ptr(p)[1] * search_size(1, 0) * 1000;
                const float t_z = quaternion_trans.ptr(p)[2] * search_size(2, 0) * 1000;

                const float q1 = quaternion_trans.ptr(p)[3] * search_size(3, 0);
                const float q2 = quaternion_trans.ptr(p)[4] * search_size(4, 0);
                const float q3 = quaternion_trans.ptr(p)[5] * search_size(5, 0);
                const float q0 = sqrt(1 - q1 * q1 - q2 * q2 - q3 * q3);

                float q_w = -(vertex_current_global.x() * q1 + vertex_current_global.y() * q2 + vertex_current_global.z() * q3);
                float q_x = q0 * vertex_current_global.x() - q3 * vertex_current_global.y() + q2 * vertex_current_global.z();
                float q_y = q3 * vertex_current_global.x() + q0 * vertex_current_global.y() - q1 * vertex_current_global.z();
                float q_z = -q2 * vertex_current_global.x() + q1 * vertex_current_global.y() + q0 * vertex_current_global.z();

                vertex_current_global.x() = q_x * q0 + q_w * (-q1) - q_z * (-q2) + q_y * (-q3) + t_x + translation_current.x();
                vertex_current_global.y() = q_y * q0 + q_z * (-q1) + q_w * (-q2) - q_x * (-q3) + t_y + translation_current.y();
                vertex_current_global.z() = q_z * q0 - q_y * (-q1) + q_x * (-q2) + q_w * (-q3) + t_z + translation_current.z();

                const Matf31fa vertex_current_camera =
                    rotation_previous_inv * (vertex_current_global - translation_previous); // 得到图像坐标系下的坐标

                Eigen::Vector2i point;
                point.x() = __float2int_rd(
                    vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() +
                    cam_params.principal_x + 0.5f);
                point.y() = __float2int_rd(
                    vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() +
                    cam_params.principal_y + 0.5f);

                if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows &&
                    vertex_current_camera.z() >= 0)
                {

                    Vec3fda grid = (vertex_current_global) / voxel_scale;
                    // printf("grid %f,%f,%f\n",grid.x(),grid.y(),grid.z());
                    int vmIndex = 0;
                    int tsdf = static_cast<int>(readFromSDF_float_uninterpolated(voxelData, hashTable, grid, vmIndex));
                    // if(vmIndex) {
                    //     printf("found voxel : %d\n",tsdf);
                    // }

                    // int tsdf = readFromSDF_voxel_interpolated(voxelData, hashTable, grid, vmIndex);
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
                     const int level_index)
            {

                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1);
                dim3 grid(1, 1, 1);
                grid.x = static_cast<unsigned int>(std::ceil((float)num_candidate / block.y / block.x));
                grid.y = static_cast<unsigned int>(std::ceil((float)cols / level));
                grid.z = static_cast<unsigned int>(std::ceil((float)rows / level));

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

            bool particle_evaluation(const ITMVoxel_d *voxelData, const ITMHashEntry *hashTable,
                                     const QuaternionData &quaterinons, ParticleSearchData &search_data, const Eigen::Matrix3d &rotation_current,
                                     const Matf31da &translation_current,
                                     const cv::cuda::GpuMat &vertex_map_current, const cv::cuda::GpuMat &normal_map_current,
                                     const Eigen::Matrix3d &rotation_previous_inv, const Matf31da &translation_previous,
                                     const CameraConfiguration &cam_params, const int particle_index, const int particle_size,
                                     const Matf61da &search_size, const int level, const int level_index,
                                     Eigen::Matrix<double, 7, 1> &mean_transform, float *min_tsdf, float voxel_scale)
            {

                std::cout.precision(17);
                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_Y, 1, 1);
                dim3 grid(1, 1, 1);
                grid.x = static_cast<unsigned int>(std::ceil((float)particle_size / block.y / block.x));
                grid.y = static_cast<unsigned int>(std::ceil((float)cols / level));
                grid.z = static_cast<unsigned int>(std::ceil((float)rows / level));

                search_data.gpu_search_count[particle_index / 20].setTo(0);
                search_data.gpu_search_value[particle_index / 20].setTo(0);
                particle_kernel<<<grid, block>>>(voxelData, hashTable, vertex_map_current, normal_map_current,
                                                 search_data.gpu_search_value[particle_index / 20],
                                                 search_data.gpu_search_count[particle_index / 20],
                                                 rotation_current.cast<float>(), translation_current.cast<float>(),
                                                 rotation_previous_inv.cast<float>(), translation_previous.cast<float>(), quaterinons.q[particle_index],
                                                 cam_params, voxel_scale, particle_size, cols, rows, search_size,
                                                 level, level_index);
                cv::Mat search_data_count = search_data.search_count[particle_index / 20];
                cv::Mat search_data_value = search_data.search_value[particle_index / 20];
                search_data.gpu_search_count[particle_index / 20].download(search_data_count);
                search_data.gpu_search_value[particle_index / 20].download(search_data_value);

                cudaDeviceSynchronize();

                double orgin_tsdf = (double)search_data_value.ptr<int>(0)[0] / (double)search_data_count.ptr<int>(0)[0];
                std::cout << "orgin_tsdf: " << orgin_tsdf << std::endl;
                int orgin_count = search_data_count.ptr<int>(0)[0];

                int count_search = 0.0;
                const int iter_rows = particle_size;

                double sum_t_x = 0.0;
                double sum_t_y = 0.0;
                double sum_t_z = 0.0;
                double sum_q_x = 0.0;
                double sum_q_y = 0.0;
                double sum_q_z = 0.0;
                double sum_q_w = 0.0;
                double sum_weight_sum = 0.0;
                double sum_mean_tsdf = 0.0;

                for (int i = 1; i < iter_rows; ++i)
                {

                    double tsdf_value = (double)search_data_value.ptr<int>(i)[0] / (double)search_data_count.ptr<int>(i)[0];

                    if (tsdf_value < orgin_tsdf && ((search_data_count.ptr<int>(i)[0]) > (orgin_count / 2.0)))
                    {

                        const double tx = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[0];
                        const double ty = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[1];
                        const double tz = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[2];
                        double qx = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[3];
                        double qy = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[4];
                        double qz = (double)quaterinons.q_trans[particle_index].ptr<float>(i)[5];

                        const double weight = orgin_tsdf - tsdf_value;

                        sum_t_x += tx * weight;
                        sum_t_y += ty * weight;
                        sum_t_z += tz * weight;
                        sum_q_x += qx * weight;
                        sum_q_y += qy * weight;
                        sum_q_z += qz * weight;

                        qx = qx * (double)search_size(3, 0);
                        qy = qy * (double)search_size(4, 0);
                        qz = qz * (double)search_size(5, 0);

                        const double qw = sqrt(1 - qx * qx - qy * qy - qz * qz);

                        sum_q_w += qw * weight;

                        sum_weight_sum += weight;

                        sum_mean_tsdf += weight * tsdf_value;
                        ++count_search;
                    }
                    if (count_search == 200)
                    {
                        break;
                    }
                }

                mean_transform(0, 0) = sum_t_x;
                mean_transform(1, 0) = sum_t_y;
                mean_transform(2, 0) = sum_t_z;
                mean_transform(3, 0) = sum_q_w;
                mean_transform(4, 0) = sum_q_x;
                mean_transform(5, 0) = sum_q_y;
                mean_transform(6, 0) = sum_q_z;
                const double weight_sum = sum_weight_sum;
                double mean_tsdf = sum_mean_tsdf;

                if (count_search <= 0)
                {

                    *min_tsdf = orgin_tsdf * DIVSHORTMAX;
                    return false;
                }

                mean_transform = mean_transform / weight_sum;
                mean_tsdf = mean_tsdf / weight_sum;

                mean_transform(0, 0) = mean_transform(0, 0) * (double)search_size(0, 0);
                mean_transform(1, 0) = mean_transform(1, 0) * (double)search_size(1, 0);
                mean_transform(2, 0) = mean_transform(2, 0) * (double)search_size(2, 0);

                double qw = mean_transform(3, 0);
                double qx = mean_transform(4, 0) * search_size(3, 0);
                double qy = mean_transform(5, 0) * search_size(4, 0);
                double qz = mean_transform(6, 0) * search_size(5, 0);
                double lens = 1 / sqrt(qw * qw + qx * qx + qy * qy + qz * qz);

                mean_transform(3, 0) = qw * lens;
                mean_transform(4, 0) = qx * lens;
                mean_transform(5, 0) = qy * lens;
                mean_transform(6, 0) = qz * lens;

                *min_tsdf = mean_tsdf * DIVSHORTMAX;

                return true;
            }

        }
    }
}