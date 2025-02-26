//
// Created by baominjie on 2023/6/1.
//

#ifndef DWIO_DATA_H
#define DWIO_DATA_H

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#include <ctime>
#include <iostream>
#include <string>
#include <fstream>
#include <deque>
#include <pangolin/pangolin.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#else
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#endif

#include <math.h>
#include <map>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>



using cv::cuda::GpuMat;

#define POSE_INTERPOLATION
#ifdef POSE_INTERPOLATION
// #define TEMPORAL_CALIBRATION
#endif
#define USE_DEPTH_IMAGE
#define ANDERSON_ACCELERATION

struct TimestampedPose {
    unsigned long long time;
    Eigen::Matrix4d pose;
};

struct CartographerPose {
    int index;
    float pose_x;
    float pose_y;
    float pose_theta;
    unsigned long long laser_ts;
    unsigned long long send_ts;
    unsigned long long recv_ts;
    unsigned long long sync_ts;
};

struct Cloud {
    cv::Mat vertices;
    cv::Mat normals;
    cv::Mat color;
    int num_points;
};

namespace DWIO {

    struct CameraConfiguration {
        int image_width{640};
        int image_height{480};
        float focal_x{490.402};
        float focal_y{490.113};
        float principal_x{317.857};
        float principal_y{241.343};
        float depth_scale{1.0};
        float max_parallel_cosine{0.};

        explicit CameraConfiguration(const std::string &camera_file) {
            // std::cout << "[Main] Read camera configure file!" << std::endl;
            cv::FileStorage camera_setting(camera_file, cv::FileStorage::READ);
            image_width = camera_setting["Camera.width"];
            image_height = camera_setting["Camera.height"];
            focal_x = camera_setting["Camera.fx"];
            focal_y = camera_setting["Camera.fy"];
            principal_x = camera_setting["Camera.cx"];
            principal_y = camera_setting["Camera.cy"];
            depth_scale = camera_setting["Camera.scale"];
            max_parallel_cosine = camera_setting["max_parallel_cosine"];
            camera_setting.release();
        }
    };

    struct DataConfiguration {
        int3 volume_size{make_int3(1500, 150, 1500)};
        float voxel_resolution{15.f};
        float3 init_position{(float) (volume_size.x) / 2 * voxel_resolution,
                             (float) (volume_size.x) / 2 * voxel_resolution,
                             (float) (volume_size.x) / 2 * voxel_resolution};
        float truncation_distance{120.f};
        float depth_cutoff_distance_max{2000.f};
        float depth_cutoff_distance_min{500.f};
        std::string result_path{"./"};
        std::string datasets_path{"./"};
        int point_buffer_size{3 * 2000000};

        //粒子滤波！
        int max_iteration {20};
        std::string PST_path {"~"};
        float scaling_coefficient1 {0.12};
        float scaling_coefficient2 {0.12};
        float init_fitness {0.5};
        float momentum {0.9};

        explicit DataConfiguration(const std::string &data_file) {
            // std::cout << "[Main] Read datasets configure file!" << std::endl;
            cv::FileStorage data_setting(data_file, cv::FileStorage::READ);
            depth_cutoff_distance_max = data_setting["depth_cutoff_distance_max"];
            depth_cutoff_distance_min = data_setting["depth_cutoff_distance_min"];
            voxel_resolution = data_setting["voxel_resolution"];
            truncation_distance = data_setting["truncated_size"];
            int voxel_size_x = data_setting["voxel_size_x"];
            int voxel_size_y = data_setting["voxel_size_y"];
            int voxel_size_z = data_setting["voxel_size_z"];
            volume_size = make_int3(voxel_size_x, voxel_size_y, voxel_size_z);
            float init_x = data_setting["init_x"];
            float init_y = data_setting["init_y"];
            float init_z = data_setting["init_z"];
            float init_position_x = (float) (volume_size.x) / 2 * voxel_resolution - init_x;
            float init_position_y = (float) (volume_size.y) / 2 * voxel_resolution - init_y;
            float init_position_z = (float) (volume_size.z) / 2 * voxel_resolution - init_z;
            init_position = make_float3(init_position_x, init_position_y, init_position_z);
            data_setting["datasets_path"] >> datasets_path;
            
            data_setting["PST_path"]>>PST_path;
            max_iteration=data_setting["max_iteration"];
            scaling_coefficient1=data_setting["scaling_coefficient1"];
            scaling_coefficient2=data_setting["scaling_coefficient2"];
            init_fitness=data_setting["init_fitness"];
            momentum=data_setting["momentum"];
            data_setting.release();
        }
    };

    struct OptionConfiguration {
        int3 trans_window{3, 1, 3};
        int3 rot_window{20, 1, 1};
        int num_candidates{0};
        float rotation_resolution{0.02f};
        float2 se3_converge{0.1, 0.1};
        float weight_translation{1e4};
        float weight_rotation{1e4};
        int min_CSM_count{1000};
        double max_CSM_score{2e4};

        explicit OptionConfiguration(const std::string &option_file) {
            // std::cout << "[Main] Read option file!" << std::endl;
            cv::FileStorage option_setting(option_file, cv::FileStorage::READ);

            rotation_resolution = option_setting["rotation_resolution"];

            int trans_window_x = option_setting["trans_window_x"];
            int trans_window_y = option_setting["trans_window_y"];
            int trans_window_z = option_setting["trans_window_z"];
            trans_window = make_int3(trans_window_x, trans_window_y, trans_window_z);
            int rot_window_yaw = option_setting["rot_window_yaw"];
            int rot_window_pitch = option_setting["rot_window_pitch"];
            int rot_window_roll = option_setting["rot_window_roll"];
            rot_window = make_int3(rot_window_yaw, rot_window_pitch, rot_window_roll);
            num_candidates = (2 * trans_window_x + 1) *
                             (2 * trans_window_y + 1) *
                             (2 * trans_window_z + 1) *
                             (2 * rot_window_yaw + 1) *
                             (2 * rot_window_pitch + 1) *
                             (2 * rot_window_roll + 1);

            float min_delta_rou = option_setting["min_delta_rou"];
            float min_delta_phi = option_setting["min_delta_phi"];
            se3_converge = make_float2(min_delta_rou, min_delta_phi);
            weight_translation = option_setting["weight_translation"];
            weight_rotation = option_setting["weight_rotation"];
            min_CSM_count = option_setting["min_CSM_count"];
            max_CSM_score = option_setting["max_CSM_score"];
            option_setting.release();
        }
    };

    namespace internal {

        struct FrameData {
            GpuMat depth_map;
            GpuMat depth_temp;
            GpuMat parallel_label;
            GpuMat color_map;
            GpuMat vertex_map;
            GpuMat ground_vertex;
            GpuMat non_ground_vertex;
            GpuMat normal_map;
            GpuMat shading_buffer;

            cv::Mat host_vertex_map;

            explicit FrameData(const int image_height, const int image_width) {
                depth_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC1);
                depth_temp = cv::cuda::createContinuous(image_height, image_width, CV_32FC1);
                parallel_label = cv::cuda::createContinuous(image_height, image_width, CV_32FC1);
                color_map = cv::cuda::createContinuous(image_height, image_width, CV_8UC3);
                vertex_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                ground_vertex = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                non_ground_vertex = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                normal_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                shading_buffer = cv::cuda::createContinuous(image_height, image_width, CV_8UC3);

                host_vertex_map = cv::Mat::zeros(image_height, image_width, CV_32FC3);
            }
        };

        struct VolumeData {
            GpuMat tsdf_volume;
            GpuMat weight_volume;
            GpuMat color_volume;
            int3 volume_size;
            float voxel_scale;

            VolumeData(const int3 _volume_size, const float _voxel_scale) : tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z / 1000, _volume_size.x/100, CV_16SC1)),//减小一点
                                                                            weight_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z / 1000,
                                                                                                                     _volume_size.x/100, CV_16SC1)),
                                                                            color_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z / 1000,
                                                                                                                    _volume_size.x/100, CV_8UC3)),
                                                                            volume_size(_volume_size), voxel_scale(_voxel_scale) {
                tsdf_volume.setTo(32767);
                weight_volume.setTo(0);
                color_volume.setTo(0);
            }

        };

        struct TransformCandidates {
            GpuMat q_gpu;
            cv::Mat q_cpu;

            TransformCandidates(const OptionConfiguration option_config, float voxel_resolution) {
                q_gpu = cv::cuda::createContinuous(option_config.num_candidates, 7, CV_32FC1);
                q_cpu = cv::Mat::zeros(option_config.num_candidates, 7, CV_32FC1);
                int candidate_index = 0;
                for (int trans_x = -option_config.trans_window.x;
                     trans_x <= option_config.trans_window.x; ++trans_x) {
                    for (int trans_y = -option_config.trans_window.y;
                         trans_y <= option_config.trans_window.y; ++trans_y) {
                        for (int trans_z = -option_config.trans_window.z;
                             trans_z <= option_config.trans_window.z; ++trans_z) {
                            for (int rot_yaw = -option_config.rot_window.x;
                                 rot_yaw <= option_config.rot_window.x; ++rot_yaw) {
                                for (int rot_pitch = -option_config.rot_window.y;
                                     rot_pitch <= option_config.rot_window.y; ++rot_pitch) {
                                    for (int rot_roll = -option_config.rot_window.z;
                                         rot_roll <= option_config.rot_window.z; ++rot_roll) {
                                        q_cpu.ptr<float>(candidate_index)[0] = (float) trans_x * voxel_resolution;
                                        q_cpu.ptr<float>(candidate_index)[1] = (float) trans_y * voxel_resolution;
                                        q_cpu.ptr<float>(candidate_index)[2] = (float) trans_z * voxel_resolution;

                                        Eigen::Quaternionf d_q;
                                        Eigen::AngleAxisf d_p(Eigen::AngleAxisf((float)rot_pitch *
                                        option_config.rotation_resolution,
                                                Eigen::Vector3f::UnitX()));
                                        Eigen::AngleAxisf d_y(Eigen::AngleAxisf((float)rot_yaw *
                                        option_config.rotation_resolution,
                                                Eigen::Vector3f::UnitY()));
                                        Eigen::AngleAxisf d_r(Eigen::AngleAxisf((float)rot_roll *
                                        option_config.rotation_resolution,
                                                Eigen::Vector3f::UnitZ()));
                                        d_q = d_r * d_y * d_p;
                                        q_cpu.ptr<float>(candidate_index)[3] = d_q.w();
                                        q_cpu.ptr<float>(candidate_index)[4] = d_q.x();
                                        q_cpu.ptr<float>(candidate_index)[5] = d_q.y();
                                        q_cpu.ptr<float>(candidate_index)[6] = d_q.z();

                                        ++candidate_index;
                                    }
                                }
                            }
                        }
                    }
                }
                q_gpu.upload(q_cpu);
            }
        };

        struct SearchData {
            GpuMat gpu_search_count;
            cv::Mat search_count;
            GpuMat gpu_search_value;
            cv::Mat search_value;

            GpuMat gpu_sum_A;
            cv::Mat sum_A;
            GpuMat gpu_sum_b;
            cv::Mat sum_b;
            GpuMat gpu_GN_count;
            cv::Mat GN_count;
            GpuMat gpu_GN_value;
            cv::Mat GN_value;

            explicit SearchData(const OptionConfiguration option_config) {
                search_count = cv::Mat::zeros(option_config.num_candidates, 1, CV_32FC1);
                gpu_search_count = cv::cuda::createContinuous(option_config.num_candidates, 1, CV_32FC1);
                search_value = cv::Mat::zeros(option_config.num_candidates, 1, CV_32FC1);
                gpu_search_value = cv::cuda::createContinuous(option_config.num_candidates, 1, CV_32FC1);

                sum_A = cv::Mat::zeros(36, 1, CV_32FC1);
                gpu_sum_A = cv::cuda::createContinuous(36, 1, CV_32FC1);
                sum_b = cv::Mat::zeros(6, 1, CV_32FC1);
                gpu_sum_b = cv::cuda::createContinuous(6, 1, CV_32FC1);
                GN_count = cv::Mat::zeros(1, 1, CV_32FC1);
                gpu_GN_count = cv::cuda::createContinuous(1, 1, CV_32FC1);
                GN_value = cv::Mat::zeros(1, 1, CV_32FC1);
                gpu_GN_value = cv::cuda::createContinuous(1, 1, CV_32FC1);
            }
        };

        struct QuaternionData{
            std::vector<GpuMat> q;
            std::vector<cv::Mat> q_trans;
            int num=20;

  
            QuaternionData(std::vector<int> particle_level, std::string PST_path)
            {
                q.resize(60);
                q_trans.resize(60);
                for (int i=0;i<num;i++){
                    q_trans[i]=cv::Mat(particle_level[0],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[0], 6, CV_32FC1);
                    q_trans[i]=cv::imread(PST_path+"pst_10240_"+std::to_string(i)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);

                }
                for (int i=num;i<num*2;i++){
                    q_trans[i]=cv::Mat(particle_level[1],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[1], 6, CV_32FC1);
                    q_trans[i]=cv::imread(PST_path+"pst_3072_"+std::to_string(i-20)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);

                }
                for (int i=num*2;i<num*3;i++){
                    q_trans[i]=cv::Mat(particle_level[2],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[2], 6, CV_32FC1);
                    q_trans[i]=cv::imread(PST_path+"pst_1024_"+std::to_string(i-40)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);

                }
                std::cout << "read particle" << std::endl;

            }
            
        };

        struct ParticleSearchData{
            std::vector<GpuMat> gpu_search_count;
            std::vector<cv::Mat> search_count;
            std::vector<GpuMat> gpu_search_value;
            std::vector<cv::Mat> search_value;

            ParticleSearchData(std::vector<int> particle_level)
            {
                gpu_search_count.resize(3);
                search_count.resize(3);
                gpu_search_value.resize(3);
                search_value.resize(3);
                search_count[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_count[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_count[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);
                gpu_search_count[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_count[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_count[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);
                search_value[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_value[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_value[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);
                gpu_search_value[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_value[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_value[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);
            }
            

        };

        struct CloudData
        {
            GpuMat vertices;
            GpuMat normals;
            GpuMat color;

            cv::Mat host_vertices;
            cv::Mat host_normals;
            cv::Mat host_color;

            int *point_num;
            int host_point_num;

            explicit CloudData(const int max_number) : vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                                                       normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                                                       color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
                                                       host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
            {
                vertices.setTo(0.f);
                normals.setTo(0.f);
                color.setTo(0.f);

                cudaMalloc(&point_num, sizeof(int));
                cudaMemset(point_num, 0, sizeof(int));
            }

            CloudData(const CloudData &) = delete;

            CloudData &operator=(const CloudData &data) = delete;

            void download()
            {
                vertices.download(host_vertices);
                normals.download(host_normals);
                color.download(host_color);
                cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
            }
        };


    }
}

#endif // DWIO_DATA_H