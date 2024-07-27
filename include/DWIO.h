//
// Created by baominjie on 2023/6/1.
//

#ifndef DWIO_DWIO_H
#define DWIO_DWIO_H

#include "INS.h"
#include "utility.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "../src/hash/ITMScene.h"
#include "../src/hash/ITMVoxelTypes.h"
#include "MemoryDeviceType.h"
#include "../src/cuda/include/ITMSwappingEngine_CUDA.h"
#include "../src/cuda/include/ITMSceneReconstructionEngine_CUDA.h"
#include "../src/hash/ITMRenderState_VH.h"
#include "../src/hash/ITMMesh.h"
#include "../src/cuda/include/ITMMeshingEngine_CUDA.h"
#include "ITMMeshingEngine_CPU.h"
#include "../src/hash/ITMMesh.h"
#include "../src/hash/Submap.h"
#include "../src/hash/ITMRepresentationAccess.h"
#include "Fusion.h"

namespace DWIO {

    struct KeyFrame {
        double time;
        Eigen::Matrix4d camera_pose;
        cv::Mat_<float> depth_map;
        cv::Mat_<cv::Vec3b> color_img;
        bool is_tracking_success{true};
    };

    class Pipeline {
    public:
        Pipeline(CameraConfiguration camera_config,
                 DataConfiguration data_config,
                 OptionConfiguration option_config);

        ~Pipeline() = default;

        std::vector<TimestampedPose> PoseInterpolation(const Eigen::Vector3d &pose_2d_last, const Eigen::Vector3d &pose_2d_now,
                                                       const double &pose_last_time, const double &pose_now_time,
                                                       std::deque<double> &interpolation_ratio_buffer, const double &init_y);

        bool ProcessFrameHash(const cv::Mat_<float> &depth_map, const cv::Mat_<cv::Vec3b> &color_img, cv::Mat &shaded_img,int& TotalTriangles);

        void SaveMap();       
        
        void SaveTrajectory(const KeyFrame &keyframe);

        void SaveTrajectory(const Eigen::Matrix4d CSM_pose, double time);
        
        void SaveTrajectoryInter(const Eigen::Matrix4d m_pose,double time);
        void export_ply(const std::string &filename, const Cloud &point_cloud);

        DWIO::Submap GetSubmap(DWIO::ITMScene<ITMVoxel_d, ITMVoxelBlockHash>* scene,Eigen::Matrix4d pose_,u_int32_t& submap_index);

        DWIO::submap get_submap(DWIO::ITMScene<ITMVoxel_d, ITMVoxelBlockHash>* scene,Eigen::Matrix4d pose_,u_int32_t& submap_index);

        void SaveGlobalMap(std::map<uint32_t,DWIO::Submap>&multi_submaps);
        void SaveGlobalMap(std::map<uint32_t,DWIO::submap>&submaps_);
        void SaveGlobalMapByVoxel(std::map<uint32_t,DWIO::Submap>&multi_submaps);
        void SaveGlobalMapByVoxel2(std::map<uint32_t,DWIO::submap>&submaps_);
        void SaveGlobalMap2(std::map<uint32_t,DWIO::Submap>&multi_submaps);

        void FuseSubmaps(ITMHashEntry* GlobalHashTable,ITMVoxel_d* GlobalVoxelData,DWIO::Submap& submap);
        void CheckGlobalMap(const ITMHashEntry* GlobalHashTable,int total);



    public:
        INS m_INS;

        int m_num_frame = 0;//局部子图中的帧数
        int global_frame_nums =0;//全局帧数

        Eigen::Matrix4d m_pose;
        Eigen::Matrix4d GlobalPose;

        std::deque<double> m_img_time_buffer;
        std::deque<cv::Mat> m_depth_img_buffer;
        std::deque<cv::Mat> m_color_img_buffer;

        std::deque<KeyFrame> m_keyframe_buffer;
        Eigen::Vector3d m_anchor_point;

        DWIO::ITMMesh *mesh;
        internal::FrameData m_frame_data;

    private:
        const CameraConfiguration m_camera_config;
        const DataConfiguration m_data_config;
        const OptionConfiguration m_option_config;

        internal::VolumeData m_volume_data;
        //internal::FrameData m_frame_data;
        internal::TransformCandidates m_candidates;
        internal::SearchData m_search_data;

        DWIO::ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene;

        DWIO::ITMSwappingEngine_CUDA<ITMVoxel_d> *swapEngine;
        DWIO::ITMSceneReconstructionEngine_CUDA<ITMVoxel_d> *sceneRecoEngine;
        DWIO::ITMRenderState_VH *renderState_vh;
        DWIO::ITMMeshingEngine_CUDA<ITMVoxel_d> *meshingEngine;
        DWIO::ITMMeshingEngine_CPU<ITMVoxel_d> *meshingEngineCpu;

        std::map<uint32_t,DWIO::Submap>multi_submaps;//先把子图放这里
        std::map<uint32_t,DWIO::submap>submaps_;
        uint32_t submap_index = 1;//0表示全局地图
        

    };

    namespace internal {
        void SurfaceMeasurement(const cv::Mat_<cv::Vec3b> &color_frame,
                                const cv::Mat_<float> &depth_map,
                                FrameData &frame_data,
                                const CameraConfiguration &camera_params,
                                const float depth_cutoff_max,
                                const float depth_cutoff_min);

        bool PoseEstimation(const TransformCandidates &candidates,
                            SearchData &search_data,
                            Eigen::Matrix4d &pose,
                            FrameData &frame_data,
                            const CameraConfiguration &cam_params,
                            const OptionConfiguration &option_config,
                            const ITMVoxel_d *voxelData,
                            const ITMHashEntry *hashTable,
                            float voxel_resolution,
                            Eigen::Matrix4d &CSM_pose);

        namespace cuda {
            void MapSegmentation(const cv::cuda::GpuMat vertex_map_current,
                                 const Eigen::Matrix4d &pose,
                                 const float voxel_resolution,
                                 int *volume_size,
                                 OptionConfiguration m_option_config);

            void SurfacePrediction(ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene,
                                   const float &voxel_scale,
                                   cv::cuda::GpuMat &shading_buffer,
                                   const float truncation_distance,
                                   const CameraConfiguration &cam_parameters,
                                   const float3 init_pos,
                                   cv::Mat &shaded_img,
                                   const Eigen::Matrix4d &pose);
            Cloud extract_points_hash(ITMScene<ITMVoxel_d, ITMVoxelBlockHash> *scene);
        }
    }
}

#endif // DWIO_DWIO_H
