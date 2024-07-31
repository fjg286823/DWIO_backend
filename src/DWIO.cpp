//
// Created by baominjie on 2023/6/1.
//

#include "../include/DWIO.h"

namespace DWIO {
    Pipeline::Pipeline(const CameraConfiguration camera_config,
                       const DataConfiguration data_config,
                       const OptionConfiguration option_config)
            : m_camera_config(camera_config), m_data_config(data_config), m_option_config(option_config),
              m_volume_data(data_config.volume_size, data_config.voxel_resolution),
              m_frame_data(camera_config.image_height, camera_config.image_width),
              m_candidates(option_config, data_config.voxel_resolution), m_search_data(option_config) {
        m_pose.setIdentity();
        m_pose(0, 3) = m_data_config.init_position.x;
        m_pose(1, 3) = m_data_config.init_position.y;
        m_pose(2, 3) = m_data_config.init_position.z;
        m_pose = m_pose * m_INS.extrinsic_camera_odom;
        GlobalPose = m_pose;
        scene = new ITMScene<ITMVoxel_d, ITMVoxelBlockHash>(true, MEMORYDEVICE_CUDA);
        swapEngine = new ITMSwappingEngine_CUDA<ITMVoxel_d>();
        sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<ITMVoxel_d>();
        sceneRecoEngine->ResetScene(scene);
        renderState_vh = new ITMRenderState_VH(scene->index.noTotalEntries, MEMORYDEVICE_CUDA);
        meshingEngine = new ITMMeshingEngine_CUDA<ITMVoxel_d>();
        meshingEngineCpu = new ITMMeshingEngine_CPU<ITMVoxel_d>();
        mesh = new ITMMesh(MEMORYDEVICE_CUDA);
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

        Vector4f camera_intrinsic;
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

        swapEngine->IntegrateGlobalIntoLocal(scene, renderState_vh, false);

        Eigen::Matrix4d CSM_pose;
        if (m_num_frame > 0) {
            keyframe.is_tracking_success = internal::PoseEstimation(m_candidates,
                                                                    m_search_data,
                                                                    m_pose,
                                                                    m_frame_data,
                                                                    m_camera_config,
                                                                    m_option_config,
                                                                    scene->localVBA.GetVoxelBlocks(),
                                                                    scene->index.GetEntries(),
                                                                    m_data_config.voxel_resolution,
                                                                    CSM_pose);
        }
        keyframe.camera_pose = scene->initial_pose * m_pose;
        GlobalPose = keyframe.camera_pose;
        // std::cout<<"global pose :"<<std::endl;
        // std::cout<<GlobalPose<<std::endl;

        m_keyframe_buffer.push_back(keyframe);
        if (m_keyframe_buffer.size() > 100)
            m_keyframe_buffer.pop_front();

        m_num_frame++;
        global_frame_nums++;
        std::cout<<"Frame: ["<<global_frame_nums<<"]"<<std::endl;
        
        if (keyframe.is_tracking_success) {

            if (global_frame_nums == 1)
            {
                SaveTrajectory(keyframe.camera_pose, keyframe.time);
                SaveTrajectory(keyframe);
                SaveTrajectoryInter(interpolatePose,keyframe.time);
            }
            else
            {
                CSM_pose = scene->initial_pose*CSM_pose;
                SaveTrajectory(CSM_pose, keyframe.time);
                SaveTrajectory(keyframe);
                SaveTrajectoryInter(interpolatePose,keyframe.time);
            }
            //很可能是重新创建子图后这里有问题了
            std::cout<<"分配深度图！"<<std::endl;
            sceneRecoEngine->AllocateSceneFromDepth(scene,
                                                    m_frame_data.depth_map,
                                                    m_pose,
                                                    renderState_vh,
                                                    camera_intrinsic,
                                                    m_data_config.truncation_distance,submap_size);

            sceneRecoEngine->IntegrateIntoScene(scene,
                                                m_frame_data.depth_map,
                                                m_frame_data.color_map,
                                                m_pose.inverse(),
                                                renderState_vh,
                                                camera_intrinsic,
                                                m_data_config.truncation_distance);

            swapEngine->IntegrateGlobalIntoLocal(scene, renderState_vh, true);

            swapEngine->SaveToGlobalMemory(scene, renderState_vh);
            m_frame_data.color_map.upload(color_img);
            m_frame_data.depth_map.upload(depth_map);

            internal::cuda::SurfacePrediction(scene,
                                              m_volume_data.voxel_scale,
                                              m_frame_data.shading_buffer,
                                              m_data_config.truncation_distance,
                                              m_camera_config,
                                              m_data_config.init_position,
                                              shaded_img,
                                              m_pose);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "定位代码执行时间：" << duration.count() << " 毫秒" << std::endl;

            bool NewSubmap = sceneRecoEngine->showHashTableAndVoxelAllocCondition(scene,renderState_vh);
            if(NewSubmap)
            {
                std::cout<<"**************创建子图！********************"<<std::endl;
                //复制scene的数据到子图结构中,需要先把所有数据转到cpu
                swapEngine->MoveVoxelToGlobalMemorey(scene,renderState_vh);
#ifdef SUBMAP
                DWIO::Submap submap = GetSubmap(scene,scene->initial_pose,submap_index);//应该是全局位姿
#else
                DWIO::submap* new_submap = get_submap(scene,scene->initial_pose,submap_index);//这样一次需要的时间也太久了！
#endif
                
                //清理scene的调用Reset函数，同时把体素数据给清空了
                sceneRecoEngine->ResetScene(scene);
                scene->globalCache->ResetVoxelBlocks();
                m_num_frame = 0;//保证每个子图的第一帧直接建图，不定位
                //将子图插入到后端中（写一个函数实现）目前直接在GetSubmap中实现了
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
        SaveGlobalMap(submaps_);
        //SaveGlobalMapByVoxel2(submaps_);
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
        DWIO::submap* new_submap = new DWIO::submap(pose_,submap_index,SDF_BUCKET_NUM,SDF_EXCESS_LIST_SIZE);
        scene->index.CopyHashEntries(new_submap->hashEntries_submap);
        new_submap->genereate_blocks(scene->globalCache->GetVoxelData());
        submaps_[submap_index] = new_submap;

        //将体素转成子图结构的体素数据，还需要生成子图验证一下，生成子图的代码没变过，那么就是生成submap有问题，是用new搞的？
        ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        meshingEngineCpu->MeshScene(mesh_cpu ,new_submap->blocks_, new_submap->hashEntries_submap->GetData(MEMORYDEVICE_CPU),SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE,
                                    m_data_config.voxel_resolution,new_submap->submap_pose.cast<float>());
        mesh_cpu->saveBinPly(m_data_config.datasets_path + std::to_string(new_submap->submap_id)+ "scene.ply");


        // DWIO::submap new_submap(pose_,submap_index,SDF_BUCKET_NUM,SDF_EXCESS_LIST_SIZE);
        // scene->index.CopyHashEntries(new_submap.hashEntries_submap);
        // new_submap.genereate_blocks(scene->globalCache->GetVoxelData());
        // submaps_[submap_index] = new_submap;

        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        // meshingEngineCpu->MeshScene(mesh_cpu ,new_submap.blocks_, new_submap.hashEntries_submap->GetData(MEMORYDEVICE_CPU),SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE,
        //                             m_data_config.voxel_resolution,new_submap.submap_pose.cast<float>());
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + std::to_string(new_submap.submap_id)+ "scene.ply");
        submap_index++;
        std::cout<<"get submap()"<<std::endl;
        return new_submap;
    }

    void Pipeline::SaveGlobalMap(std::map<uint32_t,DWIO::Submap>&multi_submaps)
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
                Vector3i submapPos;
                submapPos.x() = hashEntries[i].pos.x * SDF_BLOCK_SIZE;
                submapPos.y() = hashEntries[i].pos.y * SDF_BLOCK_SIZE;
                submapPos.z() = hashEntries[i].pos.z * SDF_BLOCK_SIZE;
                Vector4f submapPosf;
                submapPosf(0) = (float)(submapPos.x() + SDF_BLOCK_SIZE/2) * m_data_config.voxel_resolution;
                submapPosf(1) = (float)(submapPos.y() + SDF_BLOCK_SIZE/2) * m_data_config.voxel_resolution; 
                submapPosf(2) = (float)(submapPos.z() + SDF_BLOCK_SIZE/2) * m_data_config.voxel_resolution;  
                submapPosf(3) = 1.0;
                Vector4f globalPosef;
                globalPosef =  submap.submap_pose.cast<float>() * submapPosf;//转成float再乘
                Vector3i globalPose;//在全局坐标系下的block坐标
                globalPose.x() =  std::round(globalPosef(0) / SDF_BLOCK_SIZE/m_data_config.voxel_resolution);
                globalPose.y() =  std::round(globalPosef(1) / SDF_BLOCK_SIZE/m_data_config.voxel_resolution);
                globalPose.z() =  std::round(globalPosef(2) / SDF_BLOCK_SIZE/m_data_config.voxel_resolution);          

                //在融合过程中，不仅需要体素数据融合，还要同时创建全局hash表中对应的hash条目
                int globalHashIndex = hashIndexGlobal(globalPose);
                //std::cout<<"1"<<std::endl;
                //std::cout<<"hashindex: "<<globalHashIndex<<std::endl;
                ITMHashEntry hashEntry = GlobalHashTable[globalHashIndex];
                //std::cout<<"2"<<std::endl;
                bool isFound =false;
                bool isExtra =false;

                if(IS_EQUAL3(hashEntry.pos, globalPose) && hashEntry.ptr >= -1)
                {
                    isFound =true;
                }
                if(!isFound)//是否在额外链表中
                {
                    if(hashEntry.ptr >= -1){

                        while(hashEntry.offset >= 1){
                            globalHashIndex = MAP_BUCKET_NUM + hashEntry.offset - 1;
                            hashEntry = GlobalHashTable[globalHashIndex];
                            if(IS_EQUAL3(hashEntry.pos, globalPose))
                            {
                                isFound = true;
                                break;
                            }
                        }
                        isExtra =true;//用来表示是否是在额外列表区域没找到
                    }
                }
                //std::cout<<"3"<<std::endl;
                if(isFound)//找到了,且对应的索引为hashIdx
                {
                    std::cout<<"found !!!"<<std::endl;
                    //根据索引取体素，然后融合global和submap
                    ITMVoxel_d* submap_voxel = submap.GetVoxel(i);
                    ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);
                    for (int z = 0; z < SDF_BLOCK_SIZE; z++){
                        for (int y = 0; y < SDF_BLOCK_SIZE; y++){
                            for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                                
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[locId],128);
                            }
                        }
                    }           

                }
                else{//没找到

                    //创建hash条目，这里还涉及到额外链表的分配啊
                    if(isExtra)//hash条目要创建在额外列表的地方
                    {
                        ITMHashEntry hashEntry_temp{};
                        hashEntry_temp.pos.x = globalPose.x();
                        hashEntry_temp.pos.y = globalPose.y();
                        hashEntry_temp.pos.z = globalPose.z();
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
                        hashEntry_temp.pos.x = globalPose.x();
                        hashEntry_temp.pos.y = globalPose.y();
                        hashEntry_temp.pos.z = globalPose.z();
                        hashEntry_temp.ptr = -1;
                        hashEntry_temp.offset = 0;

                        GlobalHashTable[globalHashIndex] = hashEntry_temp;
                        //std::cout<<"generate a hashEntry"<<std::endl;
                    }
                    //std::cout<<"4"<<std::endl;
                    ITMVoxel_d* submap_voxel = submap.GetVoxel(i);
                    ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);
                    for (int z = 0; z < SDF_BLOCK_SIZE; z++){
                        for (int y = 0; y < SDF_BLOCK_SIZE; y++){
                            for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                                
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[locId],128);
                            }
                        }
                    } 
                }
                
            }
        }

        // //检查全局地图
        // CheckGlobalMap(GlobalHashTable,GlobalMap.noTotalEntries);
        // //在这里生成地图！
        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        // std::cout<<"2"<<std::endl;
        // meshingEngineCpu->MeshScene_global(mesh_cpu,GlobalVoxelData,GlobalHashTable,GlobalMap.noTotalEntries,m_data_config.voxel_resolution);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");
        // std::cout<<"save map successfully!"<<std::endl;
    }

    void Pipeline::SaveGlobalMapByVoxel(std::map<uint32_t,DWIO::Submap>&multi_submaps)
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
    }

    
    void Pipeline::SaveGlobalMap(std::map<uint32_t,DWIO::submap*>&submaps_)
    {
        DWIO::Submap GlobalMap(Eigen::Matrix4d::Identity(),0,SDF_BUCKET_NUM,SDF_EXCESS_LIST_SIZE);
        ITMHashEntry* GlobalHashTable = GlobalMap.hashEntries_submap->GetData(MEMORYDEVICE_CPU);
        ITMVoxel_d* GlobalVoxelData = GlobalMap.storedVoxelBlocks_submap;    
        int nums =0;
        for( auto& it : submaps_)
        {
            auto& submap = it.second;
            // if(submap.submap_id >1)
            // {
            //     std::cout<<"only fusion "<<nums<<" submap"<<std::endl;
            //     break;
            // }
            nums++;
            const ITMHashEntry* hashEntries = submap->hashEntries_submap->GetData(MEMORYDEVICE_CPU);
            CheckGlobalMap(hashEntries,submap->noTotalEntries);
            for(int i=0;i<submap->noTotalEntries;i++)
            {
                if(hashEntries[i].ptr<-1) continue;
                Vector3i submapPos;
                submapPos.x() = hashEntries[i].pos.x * SDF_BLOCK_SIZE;
                submapPos.y() = hashEntries[i].pos.y * SDF_BLOCK_SIZE;
                submapPos.z() = hashEntries[i].pos.z * SDF_BLOCK_SIZE;
                //这里求global block pos有问题，如果submapPose为负数就不是我想要的解决了，还是直接使用搭配的函数吧
                Vector4f submapPosf;//回来试试啥都不加看看
                submapPosf(0) = (float)(submapPos.x() + (submapPos.x()>0 ? SDF_BLOCK_SIZE/2: (-SDF_BLOCK_SIZE/2))) * m_data_config.voxel_resolution;
                submapPosf(1) = (float)(submapPos.y() + (submapPos.y()>0 ? SDF_BLOCK_SIZE/2: (-SDF_BLOCK_SIZE/2))) * m_data_config.voxel_resolution; 
                submapPosf(2) = (float)(submapPos.z() + (submapPos.z()>0 ? SDF_BLOCK_SIZE/2: (-SDF_BLOCK_SIZE/2))) * m_data_config.voxel_resolution;  
                submapPosf(3) = 1.0;
                Vector4f globalPosef;
                globalPosef =  submap->submap_pose.cast<float>() * submapPosf;//转成float再乘
                Vector3i globalPose;//在全局坐标系下的block坐标,我草之前怎么生成成功地图的！
                globalPose.x() =  std::round(globalPosef(0) /m_data_config.voxel_resolution/ SDF_BLOCK_SIZE);
                globalPose.y() =  std::round(globalPosef(1) /m_data_config.voxel_resolution/ SDF_BLOCK_SIZE);
                globalPose.z() =  std::round(globalPosef(2) /m_data_config.voxel_resolution/ SDF_BLOCK_SIZE);          

                //在融合过程中，不仅需要体素数据融合，还要同时创建全局hash表中对应的hash条目
                int globalHashIndex = hashIndex(globalPose);
                ITMHashEntry hashEntry = GlobalHashTable[globalHashIndex];

                bool isFound =false;
                bool isExtra =false;

                if(IS_EQUAL3(hashEntry.pos, globalPose) && hashEntry.ptr >= -1)
                {
                    isFound =true;
                }
                if(!isFound)//是否在额外链表中
                {
                    if(hashEntry.ptr >= -1){

                        while(hashEntry.offset >= 1){
                            globalHashIndex = SDF_BUCKET_NUM + hashEntry.offset - 1;
                            hashEntry = GlobalHashTable[globalHashIndex];
                            if(IS_EQUAL3(hashEntry.pos, globalPose))
                            {
                                isFound = true;
                                break;
                            }
                        }
                        isExtra =true;//用来表示是否是在额外列表区域没找到
                    }
                }

                if(isFound)//找到了,且对应的索引为hashIdx
                {
                    std::cout<<"found !!!"<<std::endl;
                    //根据索引取体素，然后融合global和submap
                    ITMVoxel_d* submap_voxel = submap->GetVoxel(i);
                    ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);
                    for (int z = 0; z < SDF_BLOCK_SIZE; z++){
                        for (int y = 0; y < SDF_BLOCK_SIZE; y++){
                            for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                                
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[locId],128);
                            }
                        }
                    }           

                }
                else{//没找到
                    //std::cout<<"no found !!!"<<std::endl;
                    if(isExtra)//hash条目要创建在额外列表的地方
                    {
                        ITMHashEntry hashEntry_temp{};
                        hashEntry_temp.pos.x = globalPose.x();
                        hashEntry_temp.pos.y = globalPose.y();
                        hashEntry_temp.pos.z = globalPose.z();
                        hashEntry_temp.ptr = -1;

                        int exlOffset = GlobalMap.GetExtraListPos();
                        if(exlOffset < 0)//如果额外链表不够了就跳过这个block的处理
                            continue;
                        
                        GlobalHashTable[globalHashIndex].offset = exlOffset + 1; 
                        GlobalHashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry_temp; 
                        globalHashIndex = SDF_BUCKET_NUM + exlOffset;//为了后面的融合赋值！
                        //std::cout<<"generate a hashEntry"<<std::endl;
                    }
                    else{
                        //顺序部分插入
                        ITMHashEntry hashEntry_temp{};
                        hashEntry_temp.pos.x = globalPose.x();
                        hashEntry_temp.pos.y = globalPose.y();
                        hashEntry_temp.pos.z = globalPose.z();
                        hashEntry_temp.ptr = -1;
                        hashEntry_temp.offset = 0;

                        GlobalHashTable[globalHashIndex] = hashEntry_temp;
                        //std::cout<<"generate a hashEntry"<<std::endl;
                    }
                    //std::cout<<"generate voxel"<<std::endl;
                    ITMVoxel_d* submap_voxel = submap->GetVoxel(i);
                    ITMVoxel_d* global_voxel = GlobalMap.GetVoxel(globalHashIndex);
                    for (int z = 0; z < SDF_BLOCK_SIZE; z++){
                        for (int y = 0; y < SDF_BLOCK_SIZE; y++){
                            for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                            
                                int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
                                compute(submap_voxel[locId],global_voxel[locId],128);
                            }
                        }
                    } 
                }


            }

        }
        // CheckGlobalMap(GlobalHashTable,GlobalMap.noTotalEntries);
        // ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        // meshingEngineCpu->MeshScene_global(mesh_cpu,GlobalVoxelData,GlobalHashTable,GlobalMap.noTotalEntries,m_data_config.voxel_resolution);
        // mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");  

    }

    void Pipeline::SaveGlobalMapByVoxel2(std::map<uint32_t,DWIO::submap>&submaps_)
    {
   
        for( auto& it : submaps_)//遍历每个子图
        {
            auto& submap = it.second;


    

        }

        //在这里生成地图！
        ITMMesh *mesh_cpu = new ITMMesh(MEMORYDEVICE_CPU);
        mesh_cpu->saveBinPly(m_data_config.datasets_path + "scene.ply");  
        std::cout<<"save map successfully!"<<std::endl;         
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