#include "include/common.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"
#include "../../hash/ITMRepresentationAccess.h"

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf31fa = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace DWIO {
    namespace internal {
        namespace cuda {
            __global__
            void gauss_newton_kernel(const ITMVoxel_d *voxelData,
                                     const ITMHashEntry *hashTable,
                                     const PtrStep<float3> vertex_map_current,
                                     const PtrStep<float> parallel_label,
                                     PtrStep<int> GN_count,
                                     PtrStep<float> GN_value,
                                     PtrStep<float> sum_A,
                                     PtrStep<float> sum_b,
                                     const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> so3_j_left,
                                     const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,
                                     const Matf31fa translation_current,
                                     const float voxel_scale,
                                     const int cols,
                                     const int rows) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= cols || y >= rows) {
                    return;
                }

                if (parallel_label.ptr(y)[x] > 0.1)
                    return;

                Matf31fa vertex_current;
                vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                Matf31fa rotated_vertex = rotation_current * vertex_current;
                Matf31fa vertex_global = rotated_vertex + translation_current;
                Vector3f grid = (vertex_global) / voxel_scale;

                int x_1 = __float2int_rd(grid.x());
                int y_1 = __float2int_rd(grid.y());
                int z_1 = __float2int_rd(grid.z());
                int x_2 = __float2int_ru(grid.x());
                int y_2 = __float2int_ru(grid.y());
                int z_2 = __float2int_ru(grid.z());

                Vector3i grid_111(x_1, y_1, z_1);
                Vector3i grid_211(x_2, y_1, z_1);
                Vector3i grid_121(x_1, y_2, z_1);
                Vector3i grid_112(x_1, y_1, z_2);
                Vector3i grid_122(x_1, y_2, z_2);
                Vector3i grid_212(x_2, y_1, z_2);
                Vector3i grid_221(x_2, y_2, z_1);
                Vector3i grid_222(x_2, y_2, z_2);

                int vmIndex;
                ITMVoxelBlockHash::IndexCache cache;
                auto tsdf_v_111 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_111, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_211 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_211, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_121 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_121, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_112 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_112, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_122 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_122, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_212 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_212, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_221 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_221, vmIndex, cache).tsdf) * 32767;
                auto tsdf_v_222 = ITMVoxel_d::valueToFloat(readVoxel(voxelData, hashTable, grid_222, vmIndex, cache).tsdf) * 32767;

                float t_x = grid.x() - static_cast<float>(x_1);
                float t_y = grid.y() - static_cast<float>(y_1);
                float t_z = grid.z() - static_cast<float>(z_1);
                float t_xx = t_x * t_x;
                float t_xxx = t_xx * t_x;
                float t_yy = t_y * t_y;
                float t_yyy = t_yy * t_y;
                float t_zz = t_z * t_z;
                float t_zzz = t_zz * t_z;

                float tsdf_v_11 = tsdf_v_111 * (2 * t_zzz - 3 * t_zz + 1) + tsdf_v_112 * (-2 * t_zzz + 3 * t_zz);
                float tsdf_v_12 = tsdf_v_121 * (2 * t_zzz - 3 * t_zz + 1) + tsdf_v_122 * (-2 * t_zzz + 3 * t_zz);
                float tsdf_v_21 = tsdf_v_211 * (2 * t_zzz - 3 * t_zz + 1) + tsdf_v_212 * (-2 * t_zzz + 3 * t_zz);
                float tsdf_v_22 = tsdf_v_221 * (2 * t_zzz - 3 * t_zz + 1) + tsdf_v_222 * (-2 * t_zzz + 3 * t_zz);
                float tsdf_v_1 = tsdf_v_11 * (2 * t_yyy - 3 * t_yy + 1) + tsdf_v_12 * (-2 * t_yyy + 3 * t_yy);
                float tsdf_v_2 = tsdf_v_21 * (2 * t_yyy - 3 * t_yy + 1) + tsdf_v_22 * (-2 * t_yyy + 3 * t_yy);
                float tsdf_v = tsdf_v_1 * (2 * t_xxx - 3 * t_xx + 1) + tsdf_v_2 * (-2 * t_xxx + 3 * t_xx);

                float d_tsdf_v_d_x = tsdf_v_1 * (6 * t_xx - 6 * t_x) + tsdf_v_2 * (-6 * t_xx + 6 * t_x);

                float d_tsdf_v_1_d_t_y = tsdf_v_11 * (6 * t_yy - 6 * t_y) + tsdf_v_12 * (-6 * t_yy + 6 * t_y);
                float d_tsdf_v_2_d_t_y = tsdf_v_21 * (6 * t_yy - 6 * t_y) + tsdf_v_22 * (-6 * t_yy + 6 * t_y);
                float d_tsdf_v_d_y = d_tsdf_v_1_d_t_y * (2 * t_xxx - 3 * t_xx + 1) + d_tsdf_v_2_d_t_y * (-2 * t_xxx + 3 * t_xx);

                float d_tsdf_v_11_d_t_z = tsdf_v_111 * (6 * t_zz - 6 * t_z) + tsdf_v_112 * (-6 * t_zz + 6 * t_z);
                float d_tsdf_v_12_d_t_z = tsdf_v_121 * (6 * t_zz - 6 * t_z) + tsdf_v_122 * (-6 * t_zz + 6 * t_z);
                float d_tsdf_v_21_d_t_z = tsdf_v_211 * (6 * t_zz - 6 * t_z) + tsdf_v_212 * (-6 * t_zz + 6 * t_z);
                float d_tsdf_v_22_d_t_z = tsdf_v_221 * (6 * t_zz - 6 * t_z) + tsdf_v_222 * (-6 * t_zz + 6 * t_z);
                float d_tsdf_v_1_d_t_z = d_tsdf_v_11_d_t_z * (2 * t_yyy - 3 * t_yy + 1) + d_tsdf_v_12_d_t_z * (-2 * t_yyy + 3 * t_yy);
                float d_tsdf_v_2_d_t_z = d_tsdf_v_21_d_t_z * (2 * t_yyy - 3 * t_yy + 1) + d_tsdf_v_22_d_t_z * (-2 * t_yyy + 3 * t_yy);
                float d_tsdf_v_d_z = d_tsdf_v_1_d_t_z * (2 * t_xxx - 3 * t_xx + 1) + d_tsdf_v_2_d_t_z * (-2 * t_xxx + 3 * t_xx);

                Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotated_vertex_skew = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>::Zero();
                rotated_vertex_skew(0, 1) = -rotated_vertex(2);
                rotated_vertex_skew(0, 2) = rotated_vertex(1);
                rotated_vertex_skew(1, 0) = rotated_vertex(2);
                rotated_vertex_skew(1, 2) = -rotated_vertex(0);
                rotated_vertex_skew(2, 0) = -rotated_vertex(1);
                rotated_vertex_skew(2, 1) = rotated_vertex(0);
                Eigen::Matrix<float, 3, 3, Eigen::DontAlign> d_v_d_phi = -rotated_vertex_skew * so3_j_left;
                Eigen::Matrix<float, 3, 6, Eigen::DontAlign> d_v_d_xi = Eigen::Matrix<float, 3, 6, Eigen::DontAlign>::Zero();
                d_v_d_xi(0, 0) = 1.0;
                d_v_d_xi(1, 1) = 1.0;
                d_v_d_xi(2, 2) = 1.0;
                d_v_d_xi(0, 3) = d_v_d_phi(0, 0);
                d_v_d_xi(0, 4) = d_v_d_phi(0, 1);
                d_v_d_xi(0, 5) = d_v_d_phi(0, 2);
                d_v_d_xi(1, 3) = d_v_d_phi(1, 0);
                d_v_d_xi(1, 4) = d_v_d_phi(1, 1);
                d_v_d_xi(1, 5) = d_v_d_phi(1, 2);
                d_v_d_xi(2, 3) = d_v_d_phi(2, 0);
                d_v_d_xi(2, 4) = d_v_d_phi(2, 1);
                d_v_d_xi(2, 5) = d_v_d_phi(2, 2);

                Eigen::Matrix<float, 6, 1, Eigen::DontAlign> grad;
                grad(0) = d_tsdf_v_d_x * d_v_d_xi(0, 0) + d_tsdf_v_d_y * d_v_d_xi(1, 0) + d_tsdf_v_d_z * d_v_d_xi(2, 0);
                grad(1) = d_tsdf_v_d_x * d_v_d_xi(0, 1) + d_tsdf_v_d_y * d_v_d_xi(1, 1) + d_tsdf_v_d_z * d_v_d_xi(2, 1);
                grad(2) = d_tsdf_v_d_x * d_v_d_xi(0, 2) + d_tsdf_v_d_y * d_v_d_xi(1, 2) + d_tsdf_v_d_z * d_v_d_xi(2, 2);
                grad(3) = d_tsdf_v_d_x * d_v_d_xi(0, 3) + d_tsdf_v_d_y * d_v_d_xi(1, 3) + d_tsdf_v_d_z * d_v_d_xi(2, 3);
                grad(4) = d_tsdf_v_d_x * d_v_d_xi(0, 4) + d_tsdf_v_d_y * d_v_d_xi(1, 4) + d_tsdf_v_d_z * d_v_d_xi(2, 4);
                grad(5) = d_tsdf_v_d_x * d_v_d_xi(0, 5) + d_tsdf_v_d_y * d_v_d_xi(1, 5) + d_tsdf_v_d_z * d_v_d_xi(2, 5);

                Eigen::Matrix<float, 6, 1, Eigen::DontAlign> b = tsdf_v * grad;
                Eigen::Matrix<float, 6, 6, Eigen::DontAlign> A = grad * grad.transpose();

                atomicAdd(sum_A + 0, A(0, 0));
                atomicAdd(sum_A + 1, A(0, 1));
                atomicAdd(sum_A + 2, A(0, 2));
                atomicAdd(sum_A + 3, A(0, 3));
                atomicAdd(sum_A + 4, A(0, 4));
                atomicAdd(sum_A + 5, A(0, 5));
                atomicAdd(sum_A + 6, A(1, 0));
                atomicAdd(sum_A + 7, A(1, 1));
                atomicAdd(sum_A + 8, A(1, 2));
                atomicAdd(sum_A + 9, A(1, 3));
                atomicAdd(sum_A + 10, A(1, 4));
                atomicAdd(sum_A + 11, A(1, 5));
                atomicAdd(sum_A + 12, A(2, 0));
                atomicAdd(sum_A + 13, A(2, 1));
                atomicAdd(sum_A + 14, A(2, 2));
                atomicAdd(sum_A + 15, A(2, 3));
                atomicAdd(sum_A + 16, A(2, 4));
                atomicAdd(sum_A + 17, A(2, 5));
                atomicAdd(sum_A + 18, A(3, 0));
                atomicAdd(sum_A + 19, A(3, 1));
                atomicAdd(sum_A + 20, A(3, 2));
                atomicAdd(sum_A + 21, A(3, 3));
                atomicAdd(sum_A + 22, A(3, 4));
                atomicAdd(sum_A + 23, A(3, 5));
                atomicAdd(sum_A + 24, A(4, 0));
                atomicAdd(sum_A + 25, A(4, 1));
                atomicAdd(sum_A + 26, A(4, 2));
                atomicAdd(sum_A + 27, A(4, 3));
                atomicAdd(sum_A + 28, A(4, 4));
                atomicAdd(sum_A + 29, A(4, 5));
                atomicAdd(sum_A + 30, A(5, 0));
                atomicAdd(sum_A + 31, A(5, 1));
                atomicAdd(sum_A + 32, A(5, 2));
                atomicAdd(sum_A + 33, A(5, 3));
                atomicAdd(sum_A + 34, A(5, 4));
                atomicAdd(sum_A + 35, A(5, 5));

                atomicAdd(sum_b + 0, b(0, 0));
                atomicAdd(sum_b + 1, b(1, 0));
                atomicAdd(sum_b + 2, b(2, 0));
                atomicAdd(sum_b + 3, b(3, 0));
                atomicAdd(sum_b + 4, b(4, 0));
                atomicAdd(sum_b + 5, b(5, 0));

                atomicAdd_system(GN_count, 1);
                atomicAdd_system(GN_value, abs(tsdf_v));
            }

            void gauss_newton(const ITMVoxel_d *voxelData,
                              const ITMHashEntry *hashTable,
                              SearchData &search_data,
                              const Eigen::MatrixXd &so3_j_left,
                              float voxel_scale,
                              const Eigen::Matrix3d &rotation_current,
                              const Matf31da &translation_current,
                              const cv::cuda::GpuMat &vertex_map_current,
                              const cv::cuda::GpuMat &parallel_label) {

                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 threads(32, 32);
                dim3 blocks((cols + threads.x - 1) / threads.x,
                            (rows + threads.y - 1) / threads.y);

                search_data.gpu_sum_A.setTo(0.);
                search_data.gpu_sum_b.setTo(0.);
                search_data.gpu_GN_count.setTo(0.);
                search_data.gpu_GN_value.setTo(0.);
                gauss_newton_kernel<<<blocks, threads>>>
                        (voxelData, hashTable, vertex_map_current, parallel_label, search_data.gpu_GN_count,
                         search_data.gpu_GN_value, search_data.gpu_sum_A, search_data.gpu_sum_b,
                         so3_j_left.cast<float>(), rotation_current.cast<float>(),
                         translation_current.cast<float>(), voxel_scale, cols, rows);
                search_data.gpu_sum_A.download(search_data.sum_A);
                search_data.gpu_sum_b.download(search_data.sum_b);
                search_data.gpu_GN_count.download(search_data.GN_count);
                search_data.gpu_GN_value.download(search_data.GN_value);
                cudaDeviceSynchronize();
            }
        }
    }
}