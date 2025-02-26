#include "include/common.h"
#include "../cuda/include/ITMSceneReconstructionEngine_CUDA.h"
#include "../cuda/include/ITMSceneReconstructionEngineShared.h"
#include "../hash/ITMRepresentationAccess.h"

using Vec3ida = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;

namespace DWIO
{
    namespace internal
    {
        namespace cuda
        {

            __device__ float get_tsdf(const ITMVoxel_d *localVBA, const ITMHashEntry *hashTable, float grid0, float grid1, float grid2)
            {
                Vector3i globalPos;
                globalPos << __float2int_rd(grid0), __float2int_rd(grid1), __float2int_rd(grid2);
                int vmIndex = 0;
                ITMVoxel_d voxel_get = readVoxel(localVBA, hashTable, globalPos, vmIndex);
                float tsdf_get = ITMVoxel_d::valueToFloat(voxel_get.tsdf);
                return tsdf_get;
            }

            __device__ float get_tsdf(const ITMVoxel_d *localVBA, const ITMHashEntry *hashTable, int grid0, int grid1, int grid2)
            {
                Vector3i globalPos;
                globalPos << grid0, grid1, grid2;
                int vmIndex = 0;
                ITMVoxel_d voxel_get = readVoxel(localVBA, hashTable, globalPos, vmIndex);
                float tsdf_get = ITMVoxel_d::valueToFloat(voxel_get.tsdf);
                return tsdf_get;
            }

            __device__ __inline__ float interpolate_trilinearly(const Vec3fda &point, const ITMVoxel_d *localVBA, const ITMHashEntry *hashTable)
            {
                Vec3ida point_in_grid = point.cast<int>();

                const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
                const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
                const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

                point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
                point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
                point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

                const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
                const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
                const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

                float tsdf_xyz = get_tsdf(localVBA, hashTable, point_in_grid.x(), point_in_grid.y(), point_in_grid.z());
                float tsdf_xyz1 = get_tsdf(localVBA, hashTable, point_in_grid.x(), point_in_grid.y(), point_in_grid.z() + 1);
                float tsdf_xy1z = get_tsdf(localVBA, hashTable, point_in_grid.x(), point_in_grid.y() + 1, point_in_grid.z());
                float tsdf_xy1z1 = get_tsdf(localVBA, hashTable, point_in_grid.x(), point_in_grid.y() + 1, point_in_grid.z() + 1);
                float tsdf_x1yz = get_tsdf(localVBA, hashTable, point_in_grid.x() + 1, point_in_grid.y(), point_in_grid.z());
                float tsdf_x1yz1 = get_tsdf(localVBA, hashTable, point_in_grid.x() + 1, point_in_grid.y(), point_in_grid.z() + 1);
                float tsdf_x1y1z = get_tsdf(localVBA, hashTable, point_in_grid.x() + 1, point_in_grid.y() + 1, point_in_grid.z());
                float tsdf_x1y1z1 = get_tsdf(localVBA, hashTable, point_in_grid.x() + 1, point_in_grid.y() + 1, point_in_grid.z() + 1);

                return tsdf_xyz * (1 - a) * (1 - b) * (1 - c) +
                       tsdf_xyz1 * (1 - a) * (1 - b) * c +
                       tsdf_xy1z * (1 - a) * b * (1 - c) +
                       tsdf_xy1z1 * (1 - a) * b * c +
                       tsdf_x1yz * a * (1 - b) * (1 - c) +
                       tsdf_x1yz1 * a * (1 - b) * c +
                       tsdf_x1y1z * a * b * (1 - c) +
                       tsdf_x1y1z1 * a * b * c;
            }

            __global__ void raycast_tsdf_kernel(ITMVoxel_d *localVBA,
                                                const ITMHashEntry *hashTable,
                                                PtrStepSz<uchar3> shading_buffer,
                                                const float voxel_scale,
                                                const float truncation_distance,
                                                const CameraConfiguration cam_parameters,
                                                const float3 init_pos,
                                                const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
                                                const Vec3fda translation)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= shading_buffer.cols || y >= shading_buffer.rows)
                {
                    return;
                }

                const Vec3fda pixel_position((x - cam_parameters.principal_x) / cam_parameters.focal_x,
                                             (y - cam_parameters.principal_y) / cam_parameters.focal_y,
                                             1.f);

                Vec3fda ray_direction = rotation * pixel_position;
                ray_direction.normalize();

                float ray_length = voxel_scale;

                Vec3fda grid = (translation + (ray_direction * ray_length)) / voxel_scale;

                float tsdf = get_tsdf(localVBA, hashTable, grid(0), grid(1), grid(2));
                float previous_tsdf;

                float result_x = 0;
                float result_y = 0;
                float result_z = 0;

                float normal_x = 0;
                float normal_y = 0;
                float normal_z = 0;

                float max_search_length = 9000; // 由于原始最大搜索长度为5166.66修改为9米

                while (ray_length < max_search_length)
                {
                    ray_length += truncation_distance * 0.5f;
                    grid = (translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale;

                    previous_tsdf = tsdf;

                    tsdf = get_tsdf(localVBA, hashTable, grid(0), grid(1), grid(2));

                    if (previous_tsdf < 0.f && tsdf > 0.f)
                    {
                        break;
                    }
                    if (previous_tsdf > 0.f && tsdf < 0.f)
                    {
                        const float t_star = ray_length - truncation_distance * 0.5f * tsdf / (tsdf - previous_tsdf);

                        const auto vertex = translation + ray_direction * t_star;

                        const Vec3fda location_in_grid = (vertex / voxel_scale);

                        Vec3fda normal, shifted;

                        shifted = location_in_grid;
                        shifted.x() += 1;
                        const float Fx1 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        shifted = location_in_grid;
                        shifted.x() -= 1;
                        const float Fx2 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        normal.x() = (Fx1 - Fx2);

                        shifted = location_in_grid;
                        shifted.y() += 1;
                        const float Fy1 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        shifted = location_in_grid;
                        shifted.y() -= 1;
                        const float Fy2 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        normal.y() = (Fy1 - Fy2);

                        shifted = location_in_grid;
                        shifted.z() += 1;
                        const float Fz1 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        shifted = location_in_grid;
                        shifted.z() -= 1;
                        const float Fz2 = interpolate_trilinearly(shifted, localVBA, hashTable);

                        normal.z() = (Fz1 - Fz2);

                        if (normal.norm() == 0)
                            break;

                        normal.normalize();
                        result_x = vertex.x();
                        result_y = vertex.y();
                        result_z = vertex.z();

                        normal_x = normal.x();
                        normal_y = normal.y();
                        normal_z = normal.z();

                        break;
                    }
                }

                if (result_x == 0 && result_y == 0 && result_z == 0)
                {
                    return;
                }

                if (normal_x == 0 && normal_y == 0 && normal_z == 0)
                {
                    return;
                }

                const float kd_x = 98. / 255;
                const float kd_y = 121. / 255;
                const float kd_z = 148. / 255;

                const float light_position_x = init_pos.x;
                const float light_position_y = init_pos.y;
                const float light_position_z = init_pos.z;

                const float eye_position_x = translation.x();
                const float eye_position_y = translation.y();
                const float eye_position_z = translation.z();
                const float light_intensity = 0.8;

                float eye_pose_direction_x = eye_position_x - result_x;
                float eye_pose_direction_y = eye_position_y - result_y;
                float eye_pose_direction_z = eye_position_z - result_z;
                float lens = sqrt(pow(eye_pose_direction_x, 2) +
                                  pow(eye_pose_direction_y, 2) +
                                  pow(eye_pose_direction_z, 2));
                eye_pose_direction_x /= lens;
                eye_pose_direction_y /= lens;
                eye_pose_direction_z /= lens;

                float light_direction_x = light_position_x - result_x;
                float light_direction_y = light_position_y - result_y;
                float light_direction_z = light_position_z - result_z;

                lens = sqrt(pow(light_direction_x, 2) +
                            pow(light_direction_y, 2) +
                            pow(light_direction_z, 2));
                light_direction_x /= lens;
                light_direction_y /= lens;
                light_direction_z /= lens;

                const float ambinent_light_x = 0.1;
                const float ambinent_light_y = 0.1;
                const float ambinent_light_z = 0.1;
                float light_cos = normal_x * light_direction_x + normal_y * light_direction_y + normal_z * light_direction_z;
                if (light_cos <= 0)
                {
                    light_cos = -light_cos;
                }

                float light_coffi = light_intensity * light_cos;
                float diffuse_light_x = kd_x * light_coffi;
                float diffuse_light_y = kd_y * light_coffi;
                float diffuse_light_z = kd_z * light_coffi;

                float h_x = light_direction_x + eye_pose_direction_x;
                float h_y = light_direction_y + eye_pose_direction_y;
                float h_z = light_direction_z + eye_pose_direction_z;
                lens = sqrt(pow(h_x, 2) + pow(h_y, 2) + pow(h_z, 2));
                h_x /= lens;
                h_y /= lens;
                h_z /= lens;
                float h_cos = normal_x * h_x + normal_y * h_y + normal_z * h_z;
                if (h_cos < 0)
                {
                    h_cos = -h_cos;
                }

                light_coffi = light_intensity * pow(h_cos, 10);

                float specular_light_x = 0.5f * light_coffi;
                float specular_light_y = 0.5f * light_coffi;
                float specular_light_z = 0.5f * light_coffi;
                shading_buffer.ptr(y)[x].x = (uchar)((ambinent_light_x + diffuse_light_x + specular_light_x) * 255);
                shading_buffer.ptr(y)[x].y = (uchar)((ambinent_light_y + diffuse_light_y + specular_light_y) * 255);
                shading_buffer.ptr(y)[x].z = (uchar)((ambinent_light_z + diffuse_light_z + specular_light_z) * 255);
                if (ambinent_light_x + diffuse_light_x + specular_light_x > 1)
                {
                    shading_buffer.ptr(y)[x].x = 255;
                }
                if (ambinent_light_y + diffuse_light_y + specular_light_y > 1)
                {
                    shading_buffer.ptr(y)[x].y = 255;
                }
                if (ambinent_light_z + diffuse_light_z + specular_light_z > 1)
                {
                    shading_buffer.ptr(y)[x].z = 255;
                }
            }

            void SurfacePrediction(ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene,
                                   const float &voxel_scale,
                                   GpuMat &shading_buffer,
                                   const float truncation_distance,
                                   const CameraConfiguration &cam_parameters,
                                   const float3 init_pos,
                                   cv::Mat &shading_img,
                                   const Eigen::Matrix4d &pose)
            {
                ITMVoxel_d *localVBA = scene->localVBA.GetVoxelBlocks();
                ITMHashEntry *hashTable = scene->index.GetEntries();

                dim3 threads(16, 16);
                dim3 blocks((shading_buffer.cols + threads.x - 1) / threads.x,
                            (shading_buffer.rows + threads.y - 1) / threads.y);

                cv::Scalar value = cv::Scalar(0, 0, 0);
                shading_buffer.setTo(value);

                raycast_tsdf_kernel<<<blocks, threads>>>(localVBA, hashTable, shading_buffer, voxel_scale, truncation_distance, cam_parameters, init_pos,
                                                         pose.block(0, 0, 3, 3).cast<float>(),
                                                         pose.block(0, 3, 3, 1).cast<float>());

                cudaError_t cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess)
                {
                    fprintf(stderr, "[Surface Prediction] CUDA error: %s\n", cudaGetErrorString(cudaStatus));
                }

                shading_buffer.download(shading_img);
            }
        }
    }
}