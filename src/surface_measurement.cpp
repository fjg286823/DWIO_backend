#include "DWIO.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"

#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudawarping.hpp>

#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace DWIO {
    namespace internal {

        namespace cuda {
            void compute_vertex_map(FrameData &frame_data, const float depth_cutoff_max,
                                    const float depth_cutoff_min, const CameraConfiguration cam_params);

            void compute_ground_vertex(FrameData &frame_data, const CameraConfiguration cam_params);

            void compute_normal_map(const GpuMat &vertex_map, GpuMat &normal_map);

            void compute_parallel_label(const GpuMat &vertex_map, const GpuMat &normal_map, GpuMat &parallel_label,
                                        const CameraConfiguration cam_params);
        }

        void SurfaceMeasurement(const cv::Mat_<cv::Vec3b> &color_frame,
                                const cv::Mat_<float> &depth_frame,
                                FrameData &frame_data,
                                const CameraConfiguration &camera_params,
                                const float depth_cutoff_max,
                                const float depth_cutoff_min) {

            frame_data.color_map.upload(color_frame);//cpu转到gpu中
            frame_data.depth_map.upload(depth_frame);
            cuda::compute_vertex_map(frame_data, depth_cutoff_max, depth_cutoff_min, camera_params);
            cuda::compute_ground_vertex(frame_data, camera_params);
            cuda::compute_normal_map(frame_data.vertex_map, frame_data.normal_map);
            cuda::compute_parallel_label(frame_data.vertex_map, frame_data.normal_map, frame_data.parallel_label, camera_params);

        }
    }
}