#include "include/common.h"

using cv::cuda::GpuMat;

#define MINF __int_as_float(0xff800000)
namespace DWIO
{
    namespace internal
    {
        namespace cuda
        {

            __global__ void kernel_compute_vertex_map(const PtrStepSz<float> depth_map, PtrStep<float3> vertex_map,
                                                      const float depth_cutoff_max, const float depth_cutoff_min,
                                                      const CameraConfiguration cam_params)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= depth_map.cols || y >= depth_map.rows)
                    return;

                float depth_value = depth_map.ptr(y)[x] /** cam_params.depth_scale*/;
                if (depth_value > depth_cutoff_max || depth_value < depth_cutoff_min)
                {
                    vertex_map.ptr(y)[x] = make_float3(0., 0., 0.);
                    return;
                }

                Vec3fda vertex((x - cam_params.principal_x) * depth_value / cam_params.focal_x,
                               (y - cam_params.principal_y) * depth_value / cam_params.focal_y,
                               depth_value);
                // printf("[Surface measurement] Vertex: %f %f %f \n", vertex.x(), vertex.y(), vertex.z());

                vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
            }

            __global__ void kernel_compute_ground_vertx(PtrStep<float3> ground_vertex, PtrStep<float3> non_ground_vertex,
                                                        const PtrStep<float3> vertex_map, const CameraConfiguration cam_params)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < 1 || x >= cam_params.image_width || y < 1 || y >= cam_params.image_height)
                    return;

                const Vec3fda center(&vertex_map.ptr(y)[x].x);
                // Ground equation: -y + 250 = 0
                // distance to ground: abs(-y + 250)
                if (abs(-center(1) + 250) < 50 && y >= cam_params.image_height / 2)
                {
                    ground_vertex.ptr(y)[x] = make_float3(center.x(), center.y(), center.z());
                }
                else
                {
                    non_ground_vertex.ptr(y)[x] = make_float3(center.x(), center.y(), center.z());
                }
            }

            __global__ void kernel_compute_normal_map(const PtrStepSz<float3> vertex_map, PtrStep<float3> normal_map)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
                    return;

                const Vec3fda left(&vertex_map.ptr(y)[x - 1].x);
                const Vec3fda right(&vertex_map.ptr(y)[x + 1].x);
                const Vec3fda upper(&vertex_map.ptr(y - 1)[x].x);
                const Vec3fda lower(&vertex_map.ptr(y + 1)[x].x);
                const Vec3fda center(&vertex_map.ptr(y)[x].x);

                Vec3fda normal;

                if (center.z() == 0 || left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
                    normal = Vec3fda(0.f, 0.f, 0.f);
                else
                {
                    Vec3fda hor(left.x() - right.x(), left.y() - right.y(), left.z() - right.z());
                    Vec3fda ver(upper.x() - lower.x(), upper.y() - lower.y(), upper.z() - lower.z());

                    normal = hor.cross(ver);
                    normal.normalize();

                    if (normal.z() > 0)
                        normal *= -1;
                }
                //                printf("[Surface measurement] Normal: %f %f %f \n", normal.x(), normal.y(), normal.z());

                normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
            }

            __global__ void compute_parallel_label_kernel(const PtrStepSz<float3> vertex_map, const PtrStepSz<float3> normal_map,
                                                          PtrStep<float> parallel_label, const CameraConfiguration cam_params)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
                {
                    parallel_label.ptr(y)[x] = 1.0;
                    return;
                }

                const Vec3fda vertex(&vertex_map.ptr(y)[x].x);
                const Vec3fda normal(&normal_map.ptr(y)[x].x);
                if (vertex.z() == 0. || normal.z() == 0.)
                {
                    parallel_label.ptr(y)[x] = 1.0;
                    return;
                }

                float cosine = -1 * normal.z();
                if (cosine < cam_params.max_parallel_cosine)
                {
                    parallel_label.ptr(y)[x] = 1.0;
                }
            }

            __global__ void filterDepth_device(PtrStepSz<float> depth_temp,
                                               const PtrStepSz<float> depth_map,
                                               const CameraConfiguration cam_params,
                                               float dThresh,
                                               float fracReq)
            {
                int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

                if (x >= 0 && x < depth_map.cols && y >= 0 && y < depth_map.rows)
                {
                    unsigned int count = 0;
                    float oldDepth = depth_map.ptr(y)[x] * cam_params.depth_scale;

                    for (int i = -3; i <= 3; i++)
                    {
                        for (int j = -3; j <= 3; j++)
                        {
                            if (x + j >= 0 && x + j < depth_map.cols && y + i >= 0 && y + i < depth_map.rows)
                            {
                                float depth = depth_map.ptr(y + i)[x + j] * cam_params.depth_scale;
                                if (depth == MINF || depth == 0.0f || fabs(depth - oldDepth) > dThresh)
                                {
                                    count++;
                                }
                            }
                        }
                    }

                    unsigned int sum = (2 * 3 + 1) * (2 * 3 + 1);
                    if ((float)count / (float)sum >= fracReq)
                    {
                        depth_temp.ptr(y)[x] = MINF;
                    }
                    else
                    {
                        depth_temp.ptr(y)[x] = depth_map.ptr(y)[x] * cam_params.depth_scale;
                    }
                }
            }

            void compute_vertex_map(FrameData &frame_data, const float depth_cutoff_max,
                                    const float depth_cutoff_min,
                                    const CameraConfiguration cam_params)
            {
                dim3 threads(32, 32);
                dim3 blocks((frame_data.depth_map.cols + threads.x - 1) / threads.x, (frame_data.depth_map.rows + threads.y - 1) / threads.y);

                filterDepth_device<<<blocks, threads>>>(frame_data.depth_temp, frame_data.depth_map, cam_params, 50.0f, 0.3f);
                cudaDeviceSynchronize();
                frame_data.depth_temp.copyTo(frame_data.depth_map);

                frame_data.vertex_map.setTo(0.);
                kernel_compute_vertex_map<<<blocks, threads>>>(frame_data.depth_map, frame_data.vertex_map, depth_cutoff_max, depth_cutoff_min, cam_params);
                frame_data.vertex_map.download(frame_data.host_vertex_map);

                cudaDeviceSynchronize();
            }

            void compute_ground_vertex(FrameData &frame_data, const CameraConfiguration cam_params)
            {
                dim3 threads(32, 32);
                dim3 blocks((frame_data.depth_map.cols + threads.x - 1) / threads.x, (frame_data.depth_map.rows + threads.y - 1) / threads.y);

                frame_data.ground_vertex.setTo(0.);
                frame_data.non_ground_vertex.setTo(0.);
                kernel_compute_ground_vertx<<<blocks, threads>>>(frame_data.ground_vertex, frame_data.non_ground_vertex, frame_data.vertex_map, cam_params);

                cudaDeviceSynchronize();
            }

            void compute_normal_map(const GpuMat &vertex_map, GpuMat &normal_map)
            {
                dim3 threads(32, 32);
                dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
                            (vertex_map.rows + threads.y - 1) / threads.y);

                kernel_compute_normal_map<<<blocks, threads>>>(vertex_map, normal_map);

                cudaDeviceSynchronize();
            }

            void compute_parallel_label(const GpuMat &vertex_map, const GpuMat &normal_map, GpuMat &parallel_label,
                                        const CameraConfiguration cam_params)
            {
                dim3 threads(32, 32);
                dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
                            (vertex_map.rows + threads.y - 1) / threads.y);

                parallel_label.setTo(0.);
                compute_parallel_label_kernel<<<blocks, threads>>>(vertex_map, normal_map, parallel_label, cam_params);

                cudaDeviceSynchronize();
            }

        }
    }
}