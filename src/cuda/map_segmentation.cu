#include "include/common.h"

#define SDF_BLOCK_SIZE 8

using Matf31fa = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Vec2ida = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;

namespace DWIO {
    namespace internal {
        namespace cuda {

            __global__
            void transform_vertex_kernel(const PtrStepSz<float3> vertex_map_current,
                                         const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
                                         const Vec3fda translation,
                                         float *vertex_global_x,
                                         float *vertex_global_y,
                                         float *vertex_global_z) {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                int i = y * vertex_map_current.cols + x;

                if (x >= vertex_map_current.cols || y >= vertex_map_current.rows) {
                    return;
                }

                if (i > vertex_map_current.cols * vertex_map_current.rows) {
                    return;
                }

                Vec3fda vertex;
                vertex.x() = vertex_map_current.ptr(y)[x].x;
                vertex.y() = vertex_map_current.ptr(y)[x].y;
                vertex.z() = vertex_map_current.ptr(y)[x].z;

                Vec3fda vertex_global = rotation * vertex + translation;

                vertex_global_x[i] = vertex_global.x();
                vertex_global_y[i] = vertex_global.y();
                vertex_global_z[i] = vertex_global.z();
            }

            __global__
            void find_max_value_kernel(const float *buffer, float *output, int len) {
                const int tid = threadIdx.x;
                const int i = blockIdx.x * blockDim.x + threadIdx.x;
                extern __shared__ float sdata_max[];
                sdata_max[tid] = (i < len) ? buffer[i] : -FLT_MAX;

                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + s]);
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    output[blockIdx.x] = sdata_max[0];
                }

            }

            __global__
            void find_min_value_kernel(const float *buffer, float *output, int len) {
                const int tid = threadIdx.x;
                const int i = blockIdx.x * blockDim.x + threadIdx.x;
                extern __shared__ float sdata_min[];
                sdata_min[tid] = (i < len) ? buffer[i] : FLT_MAX;

                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + s]);
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    output[blockIdx.x] = sdata_min[0];
                }

            }

            void MapSegmentation(const cv::cuda::GpuMat vertex_map_current,
                                 const Eigen::Matrix4d &pose,
                                 const float voxel_resolution,
                                 int *volume_size,
                                 OptionConfiguration m_option_config) {
                dim3 blockDim_0(32, 32);
                dim3 gridDim_0((vertex_map_current.cols + blockDim_0.x - 1) / blockDim_0.x,
                               (vertex_map_current.rows + blockDim_0.y - 1) / blockDim_0.y);
                const int blockDim_1 = 1024;
                const int gridDim_1 = (vertex_map_current.rows * vertex_map_current.cols + blockDim_1 - 1) / blockDim_1;

                float *output_x;
                float *output_y;
                float *output_z;
                cudaMalloc((void **) &output_x, sizeof(float) * gridDim_1);
                cudaMalloc((void **) &output_y, sizeof(float) * gridDim_1);
                cudaMalloc((void **) &output_z, sizeof(float) * gridDim_1);
                float *vertex_global_x;
                float *vertex_global_y;
                float *vertex_global_z;
                cudaMalloc((void **) &vertex_global_x, sizeof(float) * vertex_map_current.rows * vertex_map_current.cols);
                cudaMalloc((void **) &vertex_global_y, sizeof(float) * vertex_map_current.rows * vertex_map_current.cols);
                cudaMalloc((void **) &vertex_global_z, sizeof(float) * vertex_map_current.rows * vertex_map_current.cols);
                float *result_x;
                float *result_y;
                float *result_z;
                cudaMalloc((void **) &result_x, sizeof(float));
                cudaMalloc((void **) &result_y, sizeof(float));
                cudaMalloc((void **) &result_z, sizeof(float));

                transform_vertex_kernel<<<gridDim_0, blockDim_0>>>
                        (vertex_map_current, pose.block(0, 0, 3, 3).cast<float>(),
                         pose.block(0, 3, 3, 1).cast<float>(), vertex_global_x, vertex_global_y, vertex_global_z);

                cudaDeviceSynchronize();

                find_max_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_x, output_x, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();
                find_max_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_y, output_y, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();
                find_max_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_z, output_z, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();

                const int blockDim_2 = gridDim_1;
                const int gridDim_2 = 1;
                find_max_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_x, result_x, gridDim_1);
                cudaDeviceSynchronize();
                find_max_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_y, result_y, gridDim_1);
                cudaDeviceSynchronize();
                find_max_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_z, result_z, gridDim_1);
                cudaDeviceSynchronize();

                auto *x_max = (float *) malloc(sizeof(float));
                auto *y_max = (float *) malloc(sizeof(float));
                auto *z_max = (float *) malloc(sizeof(float));
                cudaMemcpy(x_max, result_x, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(y_max, result_y, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(z_max, result_z, sizeof(float), cudaMemcpyDeviceToHost);

                volume_size[3] = std::round((x_max[0] + 60) / voxel_resolution);
                volume_size[4] = std::round((y_max[0] + 30) / voxel_resolution);
                volume_size[5] = std::round((z_max[0] + 60) / voxel_resolution);

                auto *x_min = (float *) malloc(sizeof(float));
                auto *y_min = (float *) malloc(sizeof(float));
                auto *z_min = (float *) malloc(sizeof(float));

                find_min_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_x, output_x, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();
                find_min_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_y, output_y, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();
                find_min_value_kernel<<<gridDim_1, blockDim_1, 1024 * sizeof(float)>>>
                        (vertex_global_z, output_z, vertex_map_current.rows * vertex_map_current.cols);
                cudaDeviceSynchronize();

                find_min_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_x, result_x, gridDim_1);
                cudaDeviceSynchronize();
                find_min_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_y, result_y, gridDim_1);
                cudaDeviceSynchronize();
                find_min_value_kernel<<<gridDim_2, blockDim_2, blockDim_2 * sizeof(float)>>>
                        (output_z, result_z, gridDim_1);
                cudaDeviceSynchronize();

                cudaMemcpy(x_min, result_x, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(y_min, result_y, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(z_min, result_z, sizeof(float), cudaMemcpyDeviceToHost);

                volume_size[0] = std::round((x_min[0] - 60) / voxel_resolution);
                volume_size[1] = std::round((y_min[0] - 15) / voxel_resolution);
                volume_size[2] = std::round((z_min[0] - 60) / voxel_resolution);

                volume_size[0] = volume_size[0] - m_option_config.trans_window.x - 100;
                volume_size[1] = volume_size[1] - m_option_config.trans_window.y - 40;
                volume_size[2] = volume_size[2] - m_option_config.trans_window.z - 100;
                volume_size[3] = volume_size[3] + m_option_config.trans_window.x + 100;
                volume_size[4] = volume_size[4] + m_option_config.trans_window.y + 40;
                volume_size[5] = volume_size[5] + m_option_config.trans_window.z + 100;

                volume_size[0] = floorf((float) volume_size[0] / SDF_BLOCK_SIZE);
                volume_size[1] = floorf((float) volume_size[1] / SDF_BLOCK_SIZE);
                volume_size[2] = floorf((float) volume_size[2] / SDF_BLOCK_SIZE);
                volume_size[3] = ceilf((float) volume_size[3] / SDF_BLOCK_SIZE);
                volume_size[4] = ceilf((float) volume_size[4] / SDF_BLOCK_SIZE);
                volume_size[5] = ceilf((float) volume_size[5] / SDF_BLOCK_SIZE);

                cudaFree(vertex_global_x);
                cudaFree(vertex_global_y);
                cudaFree(vertex_global_z);
                cudaFree(output_x);
                cudaFree(output_y);
                cudaFree(output_z);
                cudaFree(result_x);
                cudaFree(result_y);
                cudaFree(result_z);
                free(x_max);
                free(y_max);
                free(z_max);
                free(x_min);
                free(y_min);
                free(z_min);
            }

        }
    }
}
