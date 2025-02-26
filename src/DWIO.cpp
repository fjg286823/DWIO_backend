//
// Created by baominjie on 2023/6/1.
//

#include "../include/DWIO.h"

using Vector4f = Eigen::Matrix<float, 4, 1, Eigen::DontAlign>;

std::ofstream file_submap("../submap_poses.txt", std::ios::app);

namespace DWIO {
    Pipeline::Pipeline(const CameraConfiguration camera_config,
                       const DataConfiguration data_config,
                       const OptionConfiguration option_config)
            : m_camera_config(camera_config), m_data_config(data_config), m_option_config(option_config),
              m_volume_data(data_config.volume_size, data_config.voxel_resolution),
              m_frame_data(camera_config.image_height, camera_config.image_width),
              m_candidates(option_config, data_config.voxel_resolution), m_search_data(option_config),
                /*PST(particle_level,data_config.PST_path),search_data(particle_level),*/iter_tsdf{data_config.init_fitness}
    {

        PST = new internal::QuaternionData(particle_level, data_config.PST_path);
        search_data = new internal::ParticleSearchData(particle_level);
        m_pose.setIdentity();
        // m_pose(0, 3) = m_data_config.init_position.x;
        // m_pose(1, 3) = m_data_config.init_position.y;
        // m_pose(2, 3) = m_data_config.init_position.z;
        m_pose = m_pose * m_INS.extrinsic_camera_odom;
        GlobalPose = m_pose;
        //这段初始化时间测试一下需要多久!
        auto start = std::chrono::high_resolution_clock::now();
        scene = new ITMScene<ITMVoxel_d, ITMVoxelBlockHash>(true, MEMORYDEVICE_CUDA,m_data_config.voxel_resolution);
        swapEngine = new ITMSwappingEngine_CUDA<ITMVoxel_d>();
        sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<ITMVoxel_d>();
        sceneRecoEngine->ResetScene(scene);

        renderState_vh = new ITMRenderState_VH(scene->index.noTotalEntries, MEMORYDEVICE_CUDA);
        meshingEngine = new ITMMeshingEngine_CUDA<ITMVoxel_d>();
        meshingEngineCpu = new ITMMeshingEngine_CPU<ITMVoxel_d>();

        //配置log
        mylogger = spdlog::basic_logger_mt("spdlog", "../system.log");
        mylogger->set_pattern("[%n][%Y-%m-%d %H:%M:%S.%e] [%l] [%t]  %v");
  		mylogger->set_level(spdlog::level::debug);
  		spdlog::flush_every(std::chrono::seconds(5));
  		mylogger->flush_on(spdlog::level::debug);
        if(mylogger){
            std::cout << "mylogger " << std::endl;
        }

        //后端优化
        backend = new BackendOptimazation();
        std::thread backendThread(&BackendOptimazation::Run,backend);
        backendThread.detach();


        //新的子图结果
        DWIO::LocalMap *new_map = new DWIO::LocalMap(m_pose,m_data_config.voxel_resolution);
        sceneRecoEngine->ResetScene(new_map->scene);
        activateMaps.push_back(new_map);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        //粒子滤波

        std::cout << "initial taken: " << duration.count() << " seconds." << std::endl;
    }

    std::vector<TimestampedPose> Pipeline::PoseInterpolation(const Eigen::Vector3d &pose_2d_last,
                                                             const Eigen::Vector3d &pose_2d_now,
                                                             const double &pose_last_time,
                                                             const double &pose_now_time,
                                                             std::deque<double> &interpolation_ratio_buffer,
                                                             const double &init_y) {
        Eigen::AngleAxisd pitch_last(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd yaw_last(Eigen::AngleAxisd(-pose_2d_last(2), Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd roll_last(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R_last;
        R_last = roll_last * yaw_last * pitch_last;
        m_INS.m_LiDAR_pose_last.block(0, 0, 3, 3) = R_last;
        m_INS.m_LiDAR_pose_last(0, 3) = pose_2d_last(1);
        m_INS.m_LiDAR_pose_last(1, 3) = init_y;
        m_INS.m_LiDAR_pose_last(2, 3) = pose_2d_last(0);
        m_INS.m_LiDAR_pose_time_last = pose_last_time;

        Eigen::AngleAxisd pitch_now(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd yaw_now(Eigen::AngleAxisd(-pose_2d_now(2), Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd roll_now(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R_now;
        R_now = roll_now * yaw_now * pitch_now;
        m_INS.m_LiDAR_pose_now.block(0, 0, 3, 3) = R_now;
        m_INS.m_LiDAR_pose_now(0, 3) = pose_2d_now(1);
        m_INS.m_LiDAR_pose_now(1, 3) = init_y;
        m_INS.m_LiDAR_pose_now(2, 3) = pose_2d_now(0);
        m_INS.m_LiDAR_pose_time_now = pose_now_time;

        std::vector<TimestampedPose> pose_interpolation_buffer;
        for (const auto &ratio: interpolation_ratio_buffer) {
            TimestampedPose timestamped_pose;
            timestamped_pose.pose = InterpolationMainifold(m_INS.m_LiDAR_pose_last, m_INS.m_LiDAR_pose_now, ratio);
            timestamped_pose.time = (unsigned long long) ((pose_last_time + ratio * (pose_now_time - pose_last_time)) * 1e6);
            pose_interpolation_buffer.push_back(timestamped_pose);
            interpolation_ratio_buffer.pop_front();
        }
        return pose_interpolation_buffer;
    }


    bool Pipeline::ProcessFrameHash(const cv::Mat_<float> &depth_map,
                                    const cv::Mat_<cv::Vec3b> &color_img,
                                    cv::Mat &shaded_img,int& TotalTriangles) {
        KeyFrame keyframe;
        keyframe.color_img = color_img;
        keyframe.depth_map = depth_map;
        keyframe.time = m_img_time_buffer.front();

        auto start = std::chrono::high_resolution_clock::now();

        internal::SurfaceMeasurement(color_img,
                                     depth_map,
                                     m_frame_data,
                                     m_camera_config,
                                     m_data_config.depth_cutoff_distance_max,
                                     m_data_config.depth_cutoff_distance_min);

        m_INS.GetInitialPose(m_pose, m_data_config.init_position.y);
        Eigen::Matrix4d interpolatePose = m_pose;
        GlobalPose = m_pose;
        if(m_num_frame==0)
        {
            scene->initial_pose = m_pose;
        }
        m_pose = scene->initial_pose.inverse() * m_pose;//转到子图坐标系了

        int submap_size[] = {0, 0, 0, 0, 0, 0};
        internal::cuda::MapSegmentation(m_frame_data.vertex_map,
                                        m_pose, m_data_config.voxel_resolution,
                                        submap_size,
                                        m_option_config);

        Eigen::Matrix<float, 4, 1, Eigen::DontAlign> camera_intrinsic;
        camera_intrinsic(0, 0) = m_camera_config.focal_x;
        camera_intrinsic(1, 0) = m_camera_config.focal_y;
        camera_intrinsic(2, 0) = m_camera_config.principal_x;
        camera_intrinsic(3, 0) = m_camera_config.principal_y;

        sceneRecoEngine->AllocateScene(scene,
                                       m_frame_data.depth_map,
                                       m_pose,
                                       renderState_vh,
                                       camera_intrinsic,
                                       m_data_config.truncation_distance,
                                       submap_size,
                                       false,
                                       false);
        //这里子图对应的block又被转移进来吗？
        swapEngine->IntegrateGlobalIntoLocal(scene, renderState_vh, false);

        Eigen::Matrix4d CSM_pose;
        // if (m_num_frame > 0) {
        //     keyframe.is_tracking_success = internal::PoseEstimation(m_candidates,
        //                                                             m_search_data,
        //                                                             m_pose,
        //                                                             m_frame_data,
        //                                                             m_camera_config,
        //                                                             m_option_config,
        //                                                             scene->localVBA.GetVoxelBlocks(),
        //                                                             scene->index.GetEntries(),
        //                                                             m_data_config.voxel_resolution,
        //                                                             CSM_pose);
        // }
        keyframe.camera_pose = scene->initial_pose * m_pose;
        GlobalPose = keyframe.camera_pose;

        m_keyframe_buffer.push_back(keyframe);
        if (m_keyframe_buffer.size() > 100)
            m_keyframe_buffer.pop_front();

        m_num_frame++;
        global_frame_nums++;
        std::cout<<"Frame: ["<<global_frame_nums<<"]"<<std::endl;
        
        if (keyframe.is_tracking_success) {

            // if (global_frame_nums == 1)
            // {
            //     SaveTrajectory(keyframe.camera_pose, keyframe.time);
            //     SaveTrajectory(keyframe);
            //     SaveTrajectoryInter(interpolatePose,keyframe.time);
            // }
            // else
            // {
            //     CSM_pose = scene->initial_pose*CSM_pose;
            //     SaveTrajectory(CSM_pose, keyframe.time);
            //     SaveTrajectory(keyframe);
            //     SaveTrajectoryInter(interpolatePose,keyframe.time);
            // }



            std::cout<<"分配深度图！"<<std::endl;
            sceneRecoEngine->AllocateSceneFromDepth(scene,
                                                    m_frame_data.depth_map,
                                                    m_pose,
                                                    renderState_vh,
                                                    camera_intrinsic,
                                                    m_data_config.truncation_distance,submap_size);
            //更新地图
            sceneRecoEngine->IntegrateIntoScene(scene,
                                                m_frame_data.depth_map,
                                                m_frame_data.color_map,
                                                m_pose.inverse(),
                                                renderState_vh,
                                                camera_intrinsic,
                                                m_data_config.truncation_distance);
            //融合cpu-gpu体素数据
            swapEngine->IntegrateGlobalIntoLocal(scene, renderState_vh, true);
            //转出到cpu
            swapEngine->SaveToGlobalMemory(scene, renderState_vh);
            m_frame_data.color_map.upload(color_img);
            m_frame_data.depth_map.upload(depth_map);
            //渲染可视化图片
            internal::cuda::SurfacePrediction(scene,
                                              m_volume_data.voxel_scale,
                                              m_frame_data.shading_buffer,
                                              m_data_config.truncation_distance,
                                              m_camera_config,
                                              m_data_config.init_position,
                                              shaded_img,
                                              m_pose);

            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> duration = end - start;
            // std::cout << "定位代码执行时间：" << duration.count() << " 毫秒" << std::endl;

            bool NewSubmap = sceneRecoEngine->showHashTableAndVoxelAllocCondition(scene,renderState_vh);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "定位代码执行时间：" << duration.count() << " 毫秒" << std::endl;

            if(NewSubmap)//应该让新子图与旧子图一起更新几帧之后再完全使用新子图才对
            {
                auto start2= std::chrono::high_resolution_clock::now();
                std::cout<<"**************创建子图！********************"<<std::endl;

                //复制scene的数据到子图结构中,需要先把所有数据转到cpu,这里能否变成先放到另一个空间上，然后让一个线程单独来进行子图copy
                //这个动作能不能用gpu先分配好给子图，然后再将子图数据转移到cpu上？
                swapEngine->MoveVoxelToGlobalMemorey(scene,renderState_vh);
#ifdef SUBMAP
                DWIO::Submap submap = GetSubmap(scene,scene->initial_pose,submap_index);//应该是全局位姿
#else
                DWIO::submap* new_submap = get_submap(scene,scene->initial_pose,submap_index);//这样一次需要的时间也太久了！
#endif
                // scene->initial_pose = GlobalPose;//还待处理的是需要给新子图同步更新几帧，然后再完全使用新子图，不然很容易
                //[问题]: 这里我是否不应该清空scene,而是重新生成一个scene这样速度应该会快点，但是这里也没有多慢？
                //清理scene的调用Reset函数，同时把体素数据给清空了
                sceneRecoEngine->ResetScene(scene);
                scene->globalCache->ResetVoxelBlocks();
                m_num_frame = 0;//保证每个子图的第一帧直接建图，不定位
                //将子图插入到后端中（写一个函数实现）目前直接在GetSubmap中实现了


                auto end2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration2 = end2 - start2;
                std::cout << "子图创建消耗时间：" << duration2.count() << " 毫秒" << std::endl;

            }
        }

        return keyframe.is_tracking_success;
    }


    bool Pipeline::ProcessFrameMutil(const cv::Mat_<float> &depth_map, const cv::Mat_<cv::Vec3b> &color_img, cv::Mat &shaded_img){
        KeyFrame keyframe;  
        keyframe.color_img = color_img;
        keyframe.depth_map = depth_map;

        internal::SurfaceMeasurement(color_img,
                                     depth_map,
                                     m_frame_data,
                                     m_camera_config,
                                     m_data_config.depth_cutoff_distance_max,
                                     m_data_config.depth_cutoff_distance_min);

        m_INS.GetInitialPose(m_pose, m_data_config.init_position.y);
        
        if(activateMaps.empty()){
            std::cout << "error ,no initial map!" << std::endl;
            exit(-1);
        }
        auto current_map = activateMaps.front();
        m_pose = current_map->estimatedGlobalPose.inverse() * m_pose;//转到子图坐标系了


        int submap_size[] = {0, 0, 0, 0, 0, 0};
        internal::cuda::MapSegmentation(m_frame_data.vertex_map,
                                        m_pose, m_data_config.voxel_resolution,
                                        submap_size,
                                        m_option_config);

        Eigen::Vector4f camera_intrinsic;
        camera_intrinsic(0, 0) = m_camera_config.focal_x;
        camera_intrinsic(1, 0) = m_camera_config.focal_y;
        camera_intrinsic(2, 0) = m_camera_config.principal_x;
        camera_intrinsic(3, 0) = m_camera_config.principal_y;

        sceneRecoEngine->AllocateScene(current_map->scene,
                                       m_frame_data.depth_map,
                                       m_pose,
                                       current_map->renderState,
                                       camera_intrinsic,
                                       m_data_config.truncation_distance,
                                       submap_size,
                                       false,
                                       false);
        swapEngine->IntegrateGlobalIntoLocal(current_map->scene, current_map->renderState, false);

        Eigen::Matrix4d CSM_pose;
        if (m_num_frame > 0) {
            keyframe.is_tracking_success = internal::PoseEstimation(m_candidates,
                                                                    m_search_data,
                                                                    m_pose,
                                                                    m_frame_data,
                                                                    m_camera_config,
                                                                    m_option_config,
                                                                    current_map->scene->localVBA.GetVoxelBlocks(),
                                                                    current_map->scene->index.GetEntries(),
                                                                    m_data_config.voxel_resolution,
                                                                    CSM_pose);

            // keyframe.is_tracking_success = internal::ParticlePoseEstimation(*PST,*search_data,m_search_data,m_pose,
            //                                                         m_frame_data,m_camera_config,m_option_config,m_data_config,
            //                                                         particle_level,&iter_tsdf,&previous_frame_success,initialize_search_size,
            //                                                         current_map->scene->localVBA.GetVoxelBlocks(),
            //                                                         current_map->scene->index.GetEntries(),m_data_config.voxel_resolution);
        }
        keyframe.camera_pose = current_map->estimatedGlobalPose * m_pose;
        GlobalPose = keyframe.camera_pose;

        m_keyframe_buffer.push_back(keyframe);
        if (m_keyframe_buffer.size() > 100)
            m_keyframe_buffer.pop_front();

        m_num_frame++;
        global_frame_nums++;
        std::cout<<"Frame: ["<<global_frame_nums<<"]"<<std::endl;
        
        if (keyframe.is_tracking_success) {
            Eigen::Matrix4d pose_inMap;
            
            for (auto map : activateMaps) //更新所有的活跃子图，尤其新的子图是刚建立的所以不用转出操作！
            {
                pose_inMap = map->estimatedGlobalPose.inverse()*GlobalPose;
                sceneRecoEngine->AllocateSceneFromDepth(map->scene,
                                        m_frame_data.depth_map,
                                        pose_inMap,
                                        map->renderState,
                                        camera_intrinsic,
                                        m_data_config.truncation_distance,submap_size);
                sceneRecoEngine->IntegrateIntoScene(map->scene,
                                        m_frame_data.depth_map,
                                        m_frame_data.color_map,
                                        pose_inMap.inverse(),
                                        map->renderState,
                                        camera_intrinsic,
                                        m_data_config.truncation_distance);
            }
            
            //融合cpu-gpu体素数据
            swapEngine->IntegrateGlobalIntoLocal(current_map->scene, current_map->renderState, true);
            //转出到cpu
            // swapEngine->SaveToGlobalMemory(current_map->scene, current_map->renderState);
            m_frame_data.color_map.upload(color_img);
            m_frame_data.depth_map.upload(depth_map);
            //渲染可视化图片
            internal::cuda::SurfacePrediction(current_map->scene,
                                              m_volume_data.voxel_scale,
                                              m_frame_data.shading_buffer,
                                              m_data_config.truncation_distance,
                                              m_camera_config,
                                              m_data_config.init_position,
                                              shaded_img,
                                              m_pose);


            //接下来就是写局部子图的连续跟踪几帧以及创新新子图情况，旧子图保存就是连续跟踪几帧。这个时候怎么提取数据又是件事！

            bool NewSubmap = false;
            if (activateMaps.size() == 1) // 先简单用这个逻辑来避免，明天再想怎么优雅的解决这个问题！
                NewSubmap = sceneRecoEngine->showHashTableAndVoxelAllocCondition(current_map->scene,current_map->renderState);
            if(NewSubmap){
                LocalMap *new_map = new LocalMap(GlobalPose,m_data_config.voxel_resolution);
                sceneRecoEngine->ResetScene(new_map->scene);
                activateMaps.push_back(new_map);
                current_map->generate_newmap = true;
            }
            
            if(current_map->generate_newmap){
                if(current_map->initial_nums==0){//该从当前子图里面提取子图并且释放对应的内存
            
                    swapEngine->MoveVoxelToGlobalMemorey(current_map->scene,current_map->renderState);
                    DWIO::submap* new_submap = get_submap(current_map->scene,current_map->estimatedGlobalPose,submap_index);//这样一次需要的时间也太久了
            
                    m_num_frame = 0;
                    activateMaps.pop_front();
                    delete current_map;
                }
                current_map->initial_nums--;
            
            }
        }

        return keyframe.is_tracking_success;


    }
    void Pipeline::SaveTrajectoryInter(const Eigen::Matrix4d m_pose,double time)
    {
        std::string filename = m_data_config.datasets_path + "trajectory_interpolate.bin";
        std::fstream trajectory_file;
        trajectory_file.open(filename, std::ios::out | std::ios::binary | std::ios::app);
        if (trajectory_file.is_open())
        {
            TimestampedPose pose_out;
            pose_out.time = (unsigned long long)(time * 1e6);
            pose_out.pose = m_pose;
            trajectory_file.write(reinterpret_cast<char *>(&pose_out), sizeof(TimestampedPose));
        }
        trajectory_file.close();
    }

    void Pipeline::SaveTrajectory(const Eigen::Matrix4d CSM_pose, double time)
    {
        std::string filename = m_data_config.datasets_path + "trajectory_CSM.bin";
        std::fstream trajectory_file;
        trajectory_file.open(filename, std::ios::out | std::ios::binary | std::ios::app);
        if (trajectory_file.is_open())
        {
            TimestampedPose pose_out;
            pose_out.time = (unsigned long long)(time * 1e6);
            pose_out.pose = CSM_pose;
            trajectory_file.write(reinterpret_cast<char *>(&pose_out), sizeof(TimestampedPose));
        }
        trajectory_file.close();
    }

    void Pipeline::SaveTrajectory(const KeyFrame &keyframe)
    {
        std::string filename = m_data_config.datasets_path + "trajectory.bin";
        std::fstream trajectory_file;
        trajectory_file.open(filename, std::ios::out | std::ios::binary | std::ios::app);
        if (trajectory_file.is_open())
        {
            TimestampedPose pose_out;
            pose_out.time = (unsigned long long)(keyframe.time * 1e6);
            pose_out.pose = keyframe.camera_pose;
            trajectory_file.write(reinterpret_cast<char *>(&pose_out), sizeof(TimestampedPose));
        }
        trajectory_file.close();
    }


    void Pipeline::export_ply(const std::string &filename, const Cloud &point_cloud) {
        std::ofstream file_out{filename};
        if (!file_out.is_open())
            return;

        // std::cout << "[Main] Start saving point clouds! " << std::endl;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " ";
            file_out << normal.x << " " << normal.y << " " << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " " << static_cast<int>(color.z) << std::endl;
        }

        // std::cout << "[Main] Finish saving point clouds: " << point_cloud.num_points << std::endl;
    }

    void Pipeline::SaveMap() {
        std::cout << "[Main] Start save map!" << std::endl;
        //gpu
        // Vector3s *blockMapPos;
        // int noTotalEntries = scene->index.noTotalEntries;
        // cudaMalloc((void **) &blockMapPos, noTotalEntries * sizeof(Vector3s));
        // cudaMemset(blockMapPos, 0, sizeof(Vector3s) * noTotalEntries);
        // sceneRecoEngine->computeMapBlock(scene, blockMapPos);
        // int *NeedToMoveIn = (int *) malloc(sizeof(int));
        // NeedToMoveIn[0] = 0;
        // sceneRecoEngine->SwapAllBlocks(scene, renderState_vh, NeedToMoveIn);
        // int MoveInNum = (int) ceil((float) NeedToMoveIn[0] / SDF_TRANSFER_BLOCK_NUM);
        // for (int i = 0; i < MoveInNum; i++) {
        //     std::cout<<"["<<i<<"]"<<std::endl;
        //     swapEngine->TransferGlobalMap(scene);
        // }
        //  std::cout<<"start get points"<<std::endl;
        //ITMMesh *mesh = new ITMMesh(MEMORYDEVICE_CUDA);
        // int TotalTriangles =0;
        // meshingEngine->MeshScene(mesh, scene,TotalTriangles);//因为里面的findneight()没改
        // mesh->saveBinPly("scene_bin.ply");  

        //ply
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        // Cloud local_cloud = internal::cuda::extract_points_hash(scene);
        // std::cout<<"local points num:"<<local_cloud.num_points<<std::endl;
        // export_ply(m_data_config.datasets_path+"scene.ply",local_cloud);

        //cpu - 原来的生成地图
        // sceneRecoEngine->showHashTableAndVoxelAllocCondition(scene,renderState_vh);
        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);

        // swapEngine->MoveVoxelToGlobalMemorey(scene,renderState_vh);//把gpu体素转到cpu
        // sceneRecoEngine->showHashTableAndVoxelAllocCondition(scene,renderState_vh);
        // scene->index.GpuToCpuHashData();//复制一份hash表到cpu，
        // ITMHashEntry *hashTable = scene->index.GetEntriesCpu();
        // std::cout<<"cpu map generate"<<std::endl;
        // meshingEngineCpu->MeshScene(mesh_cpu, scene);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");

#ifdef SUBMAP
        //SaveGlobalMap(multi_submaps);
        SaveGlobalMapByVoxel(multi_submaps);
#else
        backend->doOptimazation();
        std::map<uint32_t, DWIO::submap *> &multi_submap = backend->get_submaps();
        //这里还需要把当前子图的数据提取成一个子图加入进来！
        save_global_map(multi_submap);

#endif
        //回收内存操作：
        std::cout<<"free memory!"<<std::endl;
        multi_submaps.clear();
        submaps_.clear();
        std::cout<<"free done"<<std::endl;
        
    }

    DWIO::Submap Pipeline::GetSubmap(DWIO::ITMScene<ITMVoxel_d, ITMVoxelBlockHash>* scene,Eigen::Matrix4d pose_,u_int32_t& submap_index)
    {
        DWIO::Submap submap(pose_,submap_index,SDF_BUCKET_NUM , SDF_EXCESS_LIST_SIZE);
        scene->globalCache->CopyVoxelBlocks(submap.storedVoxelBlocks_submap,SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE);
        scene->index.CopyHashEntries(submap.hashEntries_submap);
        multi_submaps[submap_index] = submap;

        //使用赋值的体素和hash表生成地图，用于验证
        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        // meshingEngineCpu->MeshScene(mesh_cpu,submap.storedVoxelBlocks_submap,submap.hashEntries_submap->GetData(MEMORYDEVICE_CPU),SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE,m_data_config.voxel_resolution);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + std::to_string(submap.submap_id)+ "scene.ply");
        // std::cout<<"submap save!"<<std::endl;
        submap_index++;
        return submap;
    }

    //这样的结果是超级耗时！
    DWIO::submap* Pipeline::get_submap(DWIO::ITMScene<ITMVoxel_d, ITMVoxelBlockHash>* scene,Eigen::Matrix4d pose_,u_int32_t& submap_index)
    {
        //生成一张子图，并把hash表和体素数据转移过去！
        DWIO::submap* new_submap = new DWIO::submap(pose_,submap_index,SDF_BUCKET_NUM,SDF_EXCESS_LIST_SIZE);
        scene->index.CopyHashEntries(new_submap->hashEntries_submap);
        new_submap->genereate_blocks(scene->globalCache->GetVoxelData());


        ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        meshingEngineCpu->MeshScene(mesh_cpu ,new_submap->blocks_, new_submap->hashEntries_submap->GetData(MEMORYDEVICE_CPU),SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE,
                                    m_data_config.voxel_resolution,new_submap->submap_pose.cast<float>());
        pcl::PointCloud<PointType>::Ptr submap_points(new pcl::PointCloud<PointType>);
        mesh_cpu->get_submap_points(submap_points);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + std::to_string(new_submap->submap_id)+ "scene.ply");
        // std::string filename = m_data_config.datasets_path + std::to_string(new_submap->submap_id) + "scene.stl";
        // mesh_cpu->WriteSTL(filename.c_str());
        delete mesh_cpu;
        std::cout << "11111" << std::endl;
        // backend->Insert_submap(new_submap, submap_points, submap_index);
        backend->Insert_submap_2(new_submap, submap_points, submap_index);
        submap_index++;
        // backend->Notify();


        //保存位姿！
        if (!file_submap.is_open()) {
            std::cerr << "无法打开文件 poses.txt" << std::endl;
            exit(-1);
        }
        Eigen::Quaterniond q(new_submap->local_rotation);
        q.normalized();
        file_submap << new_submap->local_translation.x() << " " << new_submap->local_translation.y() << " " << new_submap->local_translation.z() << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " <<q.w() << std::endl;
        return new_submap;
    }


    /*void Pipeline::SaveGlobalMapByVoxel(std::map<uint32_t,DWIO::Submap>&multi_submaps)
    {
        //生成一个全局地图，并开辟新的hash函数
        DWIO::Submap GlobalMap(Eigen::Matrix4d::Identity(),0,MAP_BUCKET_NUM,MAP_EXCESS_LIST_SIZE);
        ITMHashEntry* GlobalHashTable = GlobalMap.hashEntries_submap->GetData(MEMORYDEVICE_CPU);
        ITMVoxel_d* GlobalVoxelData = GlobalMap.storedVoxelBlocks_submap;
        int nums =0;

        for( auto& it : multi_submaps)//遍历每个子图
        {
            auto& submap = it.second;

            //只保存一张子图看看效果
            if(submap.submap_id >1)
            {
                std::cout<<"only fusion "<<nums<<" submap"<<std::endl;
                break;
            }
            nums++;
            const ITMHashEntry* hashEntries = submap.hashEntries_submap->GetData(MEMORYDEVICE_CPU);//得到子图的hash表
            CheckGlobalMap(hashEntries,submap.noTotalEntries);
            //应该再封装一个子函数去实现!
            for(int i=0;i<submap.noTotalEntries;i++)//遍历子图的hash表
            {
                
                if(hashEntries[i].ptr<-1) continue;
                for (int z = 0; z < SDF_BLOCK_SIZE; z++){
                    for (int y = 0; y < SDF_BLOCK_SIZE; y++){
                        for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                            Vector3i submapPos;
                            //体素坐标
                            submapPos.x() = hashEntries[i].pos.x * SDF_BLOCK_SIZE;
                            submapPos.y() = hashEntries[i].pos.y * SDF_BLOCK_SIZE;
                            submapPos.z() = hashEntries[i].pos.z * SDF_BLOCK_SIZE;
                            Vector4f submapPosf;//该block中每个小体素的世界坐标          
                            submapPosf(0) = (float)(submapPos.x() + x + 0.5) * m_data_config.voxel_resolution;
                            submapPosf(1) = (float)(submapPos.y() + y + 0.5) * m_data_config.voxel_resolution; 
                            submapPosf(2) = (float)(submapPos.z() + z + 0.5) * m_data_config.voxel_resolution;  
                            submapPosf(3) = 1.0;
                            Vector4f globalPosef;
                            Vector3i globalPose;//在全局坐标系下的block坐标
                            Vector3s globalBlockPose;
                            globalPosef =  submap.submap_pose.cast<float>() * submapPosf;//转成float再乘
                            globalPose.x() =  std::round(globalPosef(0) / m_data_config.voxel_resolution);
                            globalPose.y() =  std::round(globalPosef(1) / m_data_config.voxel_resolution);
                            globalPose.z() =  std::round(globalPosef(2) / m_data_config.voxel_resolution);  
                            std::cout<<"1"<<std::endl; 
                            int global_linearIdx = pointToVoxelBlockPosCpu(globalPose, globalBlockPose);

                            //我再换个方法，就是找到global对应的block块后，遍历global的block块中的小体素然后再从submap中找到对应的体素赋值看看会怎么样！
                            
                            std::cout<<"2"<<std::endl;
                            //在融合过程中，不仅需要体素数据融合，还要同时创建全局hash表中对应的hash条目
                            int globalHashIndex = hashIndexGlobal(globalBlockPose);
                            std::cout<<"3:index "<<globalHashIndex<<std::endl;
                            ITMHashEntry hashEntry = GlobalHashTable[globalHashIndex];

                            bool isFound =false;
                            bool isExtra =false;

                            if(IS_EQUAL3(hashEntry.pos, globalBlockPose) && hashEntry.ptr >= -1)
                            {
                                isFound =true;
                            }
                            if(!isFound)//是否在额外链表中
                            {
                                if(hashEntry.ptr >= -1){

                                    while(hashEntry.offset >= 1){
                                        globalHashIndex = MAP_BUCKET_NUM + hashEntry.offset - 1;
                                        hashEntry = GlobalHashTable[globalHashIndex];
                                        if(IS_EQUAL3(hashEntry.pos, globalBlockPose))
                                        {
                                            isFound = true;
                                            break;
                                        }
                                    }
                                    isExtra =true;//用来表示是否是在额外列表区域没找到
                                }
                            }
                            std::cout<<"4"<<std::endl;
                            if(isFound)//找到了,且对应的索引为hashIdx
                            {
                                std::cout<<"found !!!"<<std::endl;
                                //根据索引取体素，然后融合global和submap
                                ITMVoxel_d* submap_voxel = submap.GetVoxel(i);
                                ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);
                                            
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[global_linearIdx],128);

                            }
                            else{//没找到
                                //创建hash条目，这里还涉及到额外链表的分配啊
                                if(isExtra)//hash条目要创建在额外列表的地方
                                {
                                    ITMHashEntry hashEntry_temp{};
                                    hashEntry_temp.pos.x = globalBlockPose.x();
                                    hashEntry_temp.pos.y = globalBlockPose.y();
                                    hashEntry_temp.pos.z = globalBlockPose.z();
                                    hashEntry_temp.ptr = -1;

                                    int exlOffset = GlobalMap.GetExtraListPos();
                                    if(exlOffset < 0)//如果额外链表不够了就跳过这个block的处理
                                        continue;
                                    
                                    GlobalHashTable[globalHashIndex].offset = exlOffset + 1; 
                                    GlobalHashTable[MAP_BUCKET_NUM + exlOffset] = hashEntry_temp; 
                                    globalHashIndex = MAP_BUCKET_NUM + exlOffset;//为了后面的融合赋值！
                                    //std::cout<<"generate a hashEntry"<<std::endl;
                                }
                                else{
                                    //顺序部分插入
                                    ITMHashEntry hashEntry_temp{};
                                    hashEntry_temp.pos.x = globalBlockPose.x();
                                    hashEntry_temp.pos.y = globalBlockPose.y();
                                    hashEntry_temp.pos.z = globalBlockPose.z();
                                    hashEntry_temp.ptr = -1;
                                    hashEntry_temp.offset = 0;

                                    GlobalHashTable[globalHashIndex] = hashEntry_temp;
                                    //std::cout<<"generate a hashEntry"<<std::endl;
                                }
                                std::cout<<"fusion"<<std::endl;
                                ITMVoxel_d* submap_voxel = submap.GetVoxel(i);
                                ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);                 
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[global_linearIdx],128);
                                std::cout<<"5"<<std::endl;
                            }
                        }
                    }
                }
                
            }
        }
        //检查全局地图
        // CheckGlobalMap(GlobalHashTable,GlobalMap.noTotalEntries);
        // std::cout<<"generate map"<<std::endl;
        // //在这里生成地图！
        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        // meshingEngineCpu->MeshScene_global(mesh_cpu,GlobalVoxelData,GlobalHashTable,GlobalMap.noTotalEntries,m_data_config.voxel_resolution);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");        
    }*/

    void Pipeline::save_global_map(std::map<uint32_t,DWIO::submap*>&submaps_)
    {
        //在这里生成地图！
        ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        meshingEngineCpu->MeshScene_global(mesh_cpu, submaps_, m_data_config.voxel_resolution);
        // meshingEngineCpu->MeshScene_global_Box(mesh_cpu, submaps_, m_data_config.voxel_resolution);
        // meshingEngineCpu->MeshScene_global_hash(mesh_cpu, submaps_, m_data_config.voxel_resolution);
        mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");  
        // mesh_cpu->WriteSTL(m_data_config.datasets_path + "scene.ply"); 
        std::cout<<"save map successfully!"<<std::endl;         
    }

    void Pipeline::get_last_submap(){
        std::cout << "get last submap!" << std::endl;
        auto current_map = activateMaps.front();
        swapEngine->MoveVoxelToGlobalMemorey(current_map->scene,current_map->renderState);
        get_submap(current_map->scene,current_map->estimatedGlobalPose,submap_index);
        //睡眠一会等待后端处理！
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "get over" << std::endl;
    }

    void Pipeline::FuseSubmaps(ITMHashEntry* GlobalHashTable,ITMVoxel_d* GlobalVoxelData,DWIO::Submap& submap)
    {

    }
    void Pipeline::CheckGlobalMap(const ITMHashEntry* GlobalHashTable,int total)
    {
        int nums =0;
        for(int i = 0; i < total; i++)
        {
            if(GlobalHashTable[i].ptr>=-1)
                nums++;
        }
        std::cout<<"-------------------------------"<<std::endl;
        std::cout<<"global map block nums: "<<nums<<std::endl;
        std::cout<<"-------------------------------"<<std::endl;
    }

}