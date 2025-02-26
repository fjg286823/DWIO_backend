#include "DWIO.h"
#include "common.h"
#include <opencv2/core/eigen.hpp>
#include <sys/time.h>

#include "AA.h"
#include "../../hash/ITMVoxelBlockHash.h"
#include "../../hash/ITMVoxelTypes.h"

#include <iostream>
#include <fstream>

using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
using Matrix3frm = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

namespace DWIO {
    namespace internal {
        namespace cuda {
            void CSM(const ITMVoxel_d *voxelData,
                     const ITMHashEntry *hashTable,
                     const TransformCandidates &candidates,
                     SearchData &search_data,
                     float voxel_scale,
                     const Eigen::Matrix3d &rotation_current,
                     const Matf31da &translation_current,
                     const cv::cuda::GpuMat &vertex_map_current,
                     const cv::cuda::GpuMat &parallel_label,
                     const Eigen::Matrix3d &rotation_previous_inv,
                     const Matf31da &translation_previous,
                     const CameraConfiguration &cam_params,
                     const int num_candidate,
                     const int level,
                     const int level_index);

            bool particle_evaluation(const ITMVoxel_d *voxelData, const ITMHashEntry *hashTable,
                                     const QuaternionData &quaterinons, ParticleSearchData &search_data, const Eigen::Matrix3d &rotation_current, const Matf31da &translation_current,
                                     const cv::cuda::GpuMat &vertex_map_current, const cv::cuda::GpuMat &normal_map_current,
                                     const Eigen::Matrix3d &rotation_previous_inv, const Matf31da &translation_previous,
                                     const CameraConfiguration &cam_params, const int particle_index, const int particle_size,
                                     const Matf61da &search_size, const int level, const int level_index,
                                     Eigen::Matrix<double, 7, 1> &mean_transform, float *min_tsdf,float voxel_scale);

            void gauss_newton(const ITMVoxel_d *voxelData,
                              const ITMHashEntry *hashTable,
                              SearchData &search_data,
                              const Eigen::MatrixXd &so3_j_left,
                              float voxel_scale,
                              const Eigen::Matrix3d &rotation_current,
                              const Matf31da &translation_current,
                              const cv::cuda::GpuMat &vertex_map_current,
                              const cv::cuda::GpuMat &parallel_label);


        }

        bool PoseEstimation(const TransformCandidates &candidates,
                            SearchData &search_data,
                            Eigen::Matrix4d &pose,
                            FrameData &frame_data,
                            const CameraConfiguration &cam_params,
                            const OptionConfiguration &option_config,
                            const ITMVoxel_d *voxelData,
                            const ITMHashEntry *hashTable,
                            float voxel_resolution,
                            Eigen::Matrix4d &CSM_pose) {
            Eigen::Matrix3d current_global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3d current_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix3d previous_global_rotation_inverse(current_global_rotation.inverse());
            Eigen::Vector3d previous_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix<double, 3, 1> t_odom = current_global_translation;
            Eigen::Matrix<double, 3, 1> phi_odom = LogSO3(current_global_rotation);
            Eigen::Matrix3d R_odom = current_global_rotation;

            Eigen::Matrix<double, 7, 1> transform = Eigen::Matrix<double, 7, 1>::Zero();

            double min_score, score, score_2d_pose, count_2d_pose;
            int min_count, count;
            std::cout << "[CSM] " << option_config.num_candidates << " candidates";
            cuda::CSM(voxelData, hashTable, candidates, search_data, voxel_resolution, current_global_rotation,
                      current_global_translation, frame_data.non_ground_vertex,
                      frame_data.parallel_label, previous_global_rotation_inverse,
                      previous_global_translation, cam_params, option_config.num_candidates, 4, 4);

            min_score = (double) search_data.search_value.ptr<int>(option_config.num_candidates / 2)[0] /
                        (double) search_data.search_count.ptr<int>(option_config.num_candidates / 2)[0];
            min_count = search_data.search_count.ptr<int>(option_config.num_candidates / 2)[0];
            score_2d_pose = min_score;
            count_2d_pose = min_count;

            for (int i = 0; i < option_config.num_candidates; ++i) {
                score = (double) search_data.search_value.ptr<int>(i)[0] / (double) search_data.search_count.ptr<int>(i)[0];
                count = search_data.search_count.ptr<int>(i)[0];
                if (score < min_score) {
                    transform(0, 0) = (double) candidates.q_cpu.ptr<float>(i)[0];
                    transform(1, 0) = (double) candidates.q_cpu.ptr<float>(i)[1];
                    transform(2, 0) = (double) candidates.q_cpu.ptr<float>(i)[2];
                    transform(3, 0) = (double) candidates.q_cpu.ptr<float>(i)[3];
                    transform(4, 0) = (double) candidates.q_cpu.ptr<float>(i)[4];
                    transform(5, 0) = (double) candidates.q_cpu.ptr<float>(i)[5];
                    transform(6, 0) = (double) candidates.q_cpu.ptr<float>(i)[6];

                    min_score = score;
                    min_count = count;
                }
            }

            if (min_count < option_config.min_CSM_count) {
                std::cout << " tracking failed! Score: " << min_score << " Count: " << min_count << std::endl;
                return false;
            } else {
                std::cout << " tracking success! Score: " << min_score << " Count: " << min_count << std::endl;
            }

            double qw = transform(3, 0);
            double qx = transform(4, 0);
            double qy = transform(5, 0);
            double qz = transform(6, 0);

            auto d_camera_translation = transform.head<3>();
            Eigen::Matrix3d d_camera_rotation;
            d_camera_rotation << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                    2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy);

            current_global_translation = current_global_translation + d_camera_translation;
            current_global_rotation = d_camera_rotation * current_global_rotation;

            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;
            bool flag =true;
            // if (abs(min_score - score_2d_pose) > 19000) {
            //     std::cout << "[CSM] Bad 2D pose, 2D pose score: " << score_2d_pose << " << CSM score: " << min_score << std::endl;
            //     //flag = false;
            //     return true;
            // }

            // For debug.
            CSM_pose = pose;

            //这里应该返回false;
            if (min_score > option_config.max_CSM_score)
                return true;

            Eigen::Matrix<double, 3, 1> t_CSM = current_global_translation;
            Eigen::Matrix<double, 3, 1> phi_CSM = LogSO3(current_global_rotation);

            float r_depth_last = 32767;
            float r_depth = 0;

            int n_GN_iteration = 0;

            Eigen::Matrix<double, 3, 1> t_GN = current_global_translation;
            Eigen::Matrix<double, 3, 1> phi_GN = LogSO3(current_global_rotation);
            Eigen::Matrix<double, 6, 1> xi_GN;
            xi_GN.head(3) = t_GN;
            xi_GN.tail(3) = phi_GN;

            AA anderson_;
            anderson_.init(20, 6, xi_GN);

            for (n_GN_iteration = 0; n_GN_iteration < 20; n_GN_iteration++) {
                t_GN = current_global_translation;
                phi_GN = LogSO3(current_global_rotation);

                Eigen::MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
                Eigen::MatrixXf b = Eigen::Matrix<float, 6, 1>::Zero();
                // Depth factor;
                Eigen::Matrix<double, 3, 3> so3_j_left = ComputeSO3JLeft(phi_GN);
                cuda::gauss_newton(voxelData, hashTable, search_data, so3_j_left, voxel_resolution, current_global_rotation,
                                   current_global_translation, frame_data.vertex_map, frame_data.parallel_label);
                MyCV2Eigen(search_data.sum_A, A);
                cv2eigen(search_data.sum_b, b);
                int GN_count = search_data.GN_count.ptr<int>(0)[0];
                r_depth = search_data.GN_value.ptr<float>(0)[0] / (float) GN_count;
                A = A / GN_count;
                b = b / GN_count;
                //if(flag){
                    // Translation factor.
                    Eigen::Matrix<double, 6, 3> J_translation = Eigen::Matrix<double, 6, 3>::Zero();
                    J_translation.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
                    Eigen::Matrix<double, 3, 1> r_translation = t_GN - t_odom;
                    A = A + option_config.weight_translation * (J_translation * J_translation.transpose()).cast<float>();
                    b = b + option_config.weight_translation * (J_translation * r_translation).cast<float>();
                    // Rotation factor.
                    Eigen::Matrix3d r_R = current_global_rotation.transpose() * R_odom;
                    Eigen::Matrix<double, 3, 1> r_phi = LogSO3(r_R);
                    Eigen::Matrix<double, 6, 3> J_rotation = Eigen::Matrix<double, 6, 3>::Zero();
                    J_rotation.block(3, 0, 3, 3) = (-ComputeSO3InverseJRight(r_phi) * R_odom.transpose()).transpose();
                
                    A = A + option_config.weight_rotation * (J_rotation * J_rotation.transpose()).cast<float>();
                    b = b + option_config.weight_rotation * (J_rotation * r_phi).cast<float>();
               // }
                Eigen::Matrix<double, 6, 1> d_xi = (A.ldlt().solve(-b)).cast<double>();
                Eigen::Matrix<double, 3, 1> d_t = d_xi.head(3);
                Eigen::Matrix<double, 3, 1> d_phi = d_xi.tail(3);

                if (isnan(d_xi[0])) {
                    std::cout << "[G-N] Iteration is terminated：Delta state is nan! " << std::endl;
                    break;
                }
                if (n_GN_iteration > 0 && r_depth >= r_depth_last) {
                    std::cout << "[G-N] Iteration is terminated：Depth residual increase! " << std::endl;
                    break;
                }
                if (d_t.norm() < option_config.se3_converge.x &&
                    d_phi.norm() < option_config.se3_converge.y) {
                    std::cout << "[G-N] Iteration is terminated：Converge! Delta translation: " << d_t.norm();
                    std::cout << " Delta rotation: " << d_phi.norm() << std::endl;
                    break;
                }

                r_depth_last = r_depth;
                current_global_rotation = ExpSO3(d_phi) * current_global_rotation;
                current_global_translation = current_global_translation + d_t;
                pose.block(0, 0, 3, 3) = current_global_rotation;
                pose.block(0, 3, 3, 1) = current_global_translation;

                // Anderson acceleration
                xi_GN.head(3) = current_global_translation;
                xi_GN.tail(3) = LogSO3(current_global_rotation);
                Eigen::Matrix<double, Eigen::Dynamic, 1> AA_xi = anderson_.compute(xi_GN);
                Eigen::Matrix<double, 3, 1> AA_t = AA_xi.block(0, 0, 3, 1);
                Eigen::Matrix<double, 3, 1> AA_phi = AA_xi.block(3, 0, 3, 1);
                current_global_translation = AA_t;
                current_global_rotation = ExpSO3(AA_phi);
                std::cout << "[G-N] Iteration: " << n_GN_iteration;
                std::cout << " Depth residual: " << r_depth << " Count: " << GN_count << std::endl;
            }

            return true;
        }




        void update_seach_size(const float tsdf, const float scaling_coefficient,Matf61da& search_size, Eigen::Matrix<double, 7, 1>& mean_transform )
        {
            
            double s_tx=fabs(mean_transform(0,0))+1e-3;
            double s_ty=fabs(mean_transform(1,0))+1e-3;
            double s_tz=fabs(mean_transform(2,0))+1e-3; 

            double s_qx=fabs(mean_transform(4,0))+1e-3; 
            double s_qy=fabs(mean_transform(5,0))+1e-3;
            double s_qz=fabs(mean_transform(6,0))+1e-3;

            double trans_norm=sqrt(s_tx*s_tx+s_ty*s_ty+s_tz*s_tz+s_qx*s_qx+s_qy*s_qy+s_qz*s_qz);


            double normal_tx=s_tx/trans_norm;
            double normal_ty=s_ty/trans_norm;
            double normal_tz=s_tz/trans_norm;
            double normal_qx=s_qx/trans_norm;
            double normal_qy=s_qy/trans_norm;
            double normal_qz=s_qz/trans_norm;

            search_size(3,0) = scaling_coefficient * tsdf*normal_qx+1e-3;
            search_size(4,0) = scaling_coefficient * tsdf*normal_qy+1e-3;
            search_size(5,0) = scaling_coefficient * tsdf*normal_qz+1e-3;  
            search_size(0,0) = scaling_coefficient * tsdf*normal_tx+1e-3;
            search_size(1,0) = scaling_coefficient * tsdf*normal_ty+1e-3;
            search_size(2,0) = scaling_coefficient * tsdf*normal_tz+1e-3;
        }

        bool ParticlePoseEstimation(const QuaternionData& quaternions,
                    ParticleSearchData &particle_search_data,
                    SearchData &search_data,
                    Eigen::Matrix4d &pose,
                    FrameData &frame_data,
                    const CameraConfiguration &cam_params,
                    const OptionConfiguration &option_config,
                    const DataConfiguration &controller_config,
                    const std::vector<int> particle_level,
                    float * iter_tsdf,
                    bool * previous_frame_success,
                    Matf61da& initialize_search_size,
                    const ITMVoxel_d *voxelData,
                    const ITMHashEntry *hashTable,
                    float voxel_resolution) 
        {

            Eigen::Matrix3d current_global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3d current_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix3d previous_global_rotation_inverse(current_global_rotation.inverse());
            Eigen::Vector3d previous_global_translation = pose.block(0, 3, 3, 1);
            Eigen::Matrix<double, 3, 1> t_odom = current_global_translation;
            Eigen::Matrix3d R_odom = current_global_rotation;

            float beta=controller_config.momentum;
            Matf61da previous_search_size;

            Matf61da search_size;
            float lens= controller_config.scaling_coefficient1*(*iter_tsdf);
            search_size<< lens, lens, lens, lens, lens, lens;

            int particle_index[20] ={0,1+20,2+40,3,4+20,5+40,6+0,7+20,8+40,
                                    9+0,10+20,11+40,12+0,13+20,14+40,
                                    15+0,16+20,17+40,18+0,19+20};
            int level[20] = {32,16,8,32,16,8,32,16,8,32,16,8,32,16,8,32,16,8,32,16};

            int count_particle=0;
            int level_index=5;
            bool success=true;
            bool previous_success=true;

            int count=0;
            int count_success=0;
            float min_tsdf;

            double qx;
            double qy;
            double qz;

            while(true){

                Eigen::Matrix<double, 7, 1> mean_transform=Eigen::Matrix<double, 7, 1>::Zero();
  
                if(count==18/*controller_config.max_iteration*/){
                    break;
                }

                if (!success){
                    count_particle=0;
                }
                success=cuda::particle_evaluation(voxelData,hashTable,quaternions,particle_search_data,current_global_rotation, current_global_translation,
                                    frame_data.vertex_map, frame_data.normal_map,
                                    previous_global_rotation_inverse, previous_global_translation,
                                    cam_params,particle_index[count_particle],particle_level[particle_index[count_particle]/20],
                                    search_size,level[count_particle],level_index,
                                    mean_transform,&min_tsdf,voxel_resolution); 


                if (count==0 && !success)
                {
                    *iter_tsdf=min_tsdf;
                }
            
                qx=mean_transform(4,0);
                qy=mean_transform(5,0);
                qz=mean_transform(6,0);

                if (success){
                    if (count_particle<19){
                        ++count_particle;
                    }
                    ++count_success;

                    auto camera_translation_incremental = mean_transform.head<3>();
                    double qw=mean_transform(3,0);
                    Eigen::Matrix3d camera_rotation_incremental;

                    camera_rotation_incremental << 1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw),
                                                    2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw),
                                                    2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy);

                    current_global_translation =  current_global_translation + camera_translation_incremental*1000;
                    current_global_rotation = camera_rotation_incremental * current_global_rotation;
          
                }
                

                level_index+=5;
                level_index=level_index%level[count_particle];

                update_seach_size(min_tsdf,controller_config.scaling_coefficient2,search_size,mean_transform);

                if (previous_success&&success){
                    search_size(0,0)=beta*search_size(0,0)+(1-beta)*previous_search_size(0,0);
                    search_size(1,0)=beta*search_size(1,0)+(1-beta)*previous_search_size(1,0);
                    search_size(2,0)=beta*search_size(2,0)+(1-beta)*previous_search_size(2,0);
                    search_size(3,0)=beta*search_size(3,0)+(1-beta)*previous_search_size(3,0);
                    search_size(4,0)=beta*search_size(4,0)+(1-beta)*previous_search_size(4,0);
                    search_size(5,0)=beta*search_size(5,0)+(1-beta)*previous_search_size(5,0);  

                    previous_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                    search_size(3,0),search_size(4,0),search_size(5,0);

                }else if(success){

                    previous_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                    search_size(3,0),search_size(4,0),search_size(5,0);

                }

                if(success){
                    previous_success=true;
                }else{
                    previous_success=false;
                }

                if(count==0){
                    if (success){
                        initialize_search_size<<search_size(0,0),search_size(1,0),search_size(2,0),
                        search_size(3,0),search_size(4,0),search_size(5,0);        
                        *previous_frame_success=true;
                    }else{
                        *previous_frame_success=false;
                    }
                }
                ++count;
            }


            if (count_success==0 ){
                std::cout << "-------------search failed!--------------" << std::endl;
                return false;
            }
            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;

            // 执行高斯牛顿！
             int n_GN_iteration = 0;
             float r_depth_last = 32767;
             float r_depth = 0;

             Eigen::Matrix<double, 3, 1> t_GN = current_global_translation;
             Eigen::Matrix<double, 3, 1> phi_GN = LogSO3(current_global_rotation);
             Eigen::Matrix<double, 6, 1> xi_GN;
             xi_GN.head(3) = t_GN;
             xi_GN.tail(3) = phi_GN;

             AA anderson_;
             anderson_.init(20, 6, xi_GN);

             for (n_GN_iteration = 0; n_GN_iteration < 20; n_GN_iteration++) {
                 t_GN = current_global_translation;
                 phi_GN = LogSO3(current_global_rotation);

                 Eigen::MatrixXf A = Eigen::Matrix<float, 6, 6>::Zero();
                 Eigen::MatrixXf b = Eigen::Matrix<float, 6, 1>::Zero();
                 // Depth factor;
                 Eigen::Matrix<double, 3, 3> so3_j_left = ComputeSO3JLeft(phi_GN);
                 cuda::gauss_newton(voxelData, hashTable, search_data, so3_j_left, voxel_resolution, current_global_rotation,
                                    current_global_translation, frame_data.vertex_map, frame_data.parallel_label);
                 MyCV2Eigen(search_data.sum_A, A);
                 cv2eigen(search_data.sum_b, b);
                 int GN_count = search_data.GN_count.ptr<int>(0)[0];
                 r_depth = search_data.GN_value.ptr<float>(0)[0] / (float) GN_count;
                 A = A / GN_count;
                 b = b / GN_count;

                 // Translation factor.
                 Eigen::Matrix<double, 6, 3> J_translation = Eigen::Matrix<double, 6, 3>::Zero();
                 J_translation.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
                 Eigen::Matrix<double, 3, 1> r_translation = t_GN - t_odom;
                 A = A + option_config.weight_translation * (J_translation * J_translation.transpose()).cast<float>();
                 b = b + option_config.weight_translation * (J_translation * r_translation).cast<float>();
                 // Rotation factor.
                 Eigen::Matrix3d r_R = current_global_rotation.transpose() * R_odom;
                 Eigen::Matrix<double, 3, 1> r_phi = LogSO3(r_R);
                 Eigen::Matrix<double, 6, 3> J_rotation = Eigen::Matrix<double, 6, 3>::Zero();
                 J_rotation.block(3, 0, 3, 3) = (-ComputeSO3InverseJRight(r_phi) * R_odom.transpose()).transpose();

                 A = A + option_config.weight_rotation * (J_rotation * J_rotation.transpose()).cast<float>();
                 b = b + option_config.weight_rotation * (J_rotation * r_phi).cast<float>();

                 Eigen::Matrix<double, 6, 1> d_xi = (A.ldlt().solve(-b)).cast<double>();
                 Eigen::Matrix<double, 3, 1> d_t = d_xi.head(3);
                 Eigen::Matrix<double, 3, 1> d_phi = d_xi.tail(3);

                 if (isnan(d_xi[0])) {
                     std::cout << "[G-N] Iteration is terminated：Delta state is nan! " << std::endl;
                     break;
                 }
                 if (n_GN_iteration > 0 && r_depth >= r_depth_last) {
                     std::cout << "[G-N] Iteration is terminated：Depth residual increase! " << std::endl;
                     break;
                 }
                 if (d_t.norm() < option_config.se3_converge.x &&
                     d_phi.norm() < option_config.se3_converge.y) {
                     std::cout << "[G-N] Iteration is terminated：Converge! Delta translation: " << d_t.norm();
                     std::cout << " Delta rotation: " << d_phi.norm() << std::endl;
                     break;
                 }

                 r_depth_last = r_depth;
                 current_global_rotation = ExpSO3(d_phi) * current_global_rotation;
                 current_global_translation = current_global_translation + d_t;
                 pose.block(0, 0, 3, 3) = current_global_rotation;
                 pose.block(0, 3, 3, 1) = current_global_translation;

                 // Anderson acceleration
                //  xi_GN.head(3) = current_global_translation;
                //  xi_GN.tail(3) = LogSO3(current_global_rotation);
                //  Eigen::Matrix<double, Eigen::Dynamic, 1> AA_xi = anderson_.compute(xi_GN);
                //  Eigen::Matrix<double, 3, 1> AA_t = AA_xi.block(0, 0, 3, 1);
                //  Eigen::Matrix<double, 3, 1> AA_phi = AA_xi.block(3, 0, 3, 1);
                //  current_global_translation = AA_t;
                //  current_global_rotation = ExpSO3(AA_phi);
                 std::cout << "[G-N] Iteration: " << n_GN_iteration;
                 std::cout << " Depth residual: " << r_depth << " Count: " << GN_count << std::endl;
             }


            return true;
        }
    }
}