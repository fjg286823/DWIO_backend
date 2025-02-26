#include "../include/backend.h"


//生成子图触发通知
void BackendOptimazation::Notify() {  
    {  
        std::lock_guard<std::mutex> lk(run_mtx_);
        is_ready_ = true; 
    } 
    cond_.notify_all();
}


void BackendOptimazation::WaitForNotify() {  
    std::unique_lock<std::mutex> lock(run_mtx_);  
    //wait()把锁run_mtx的控制权交出去,并将当前线程设置为等待状态,直到被唤醒,重新拿回锁的控制权,被唤醒后会执行lambda表达式,返回true,则返回,返回false则继续等待  
    cond_.wait(lock, [this] { return is_ready_ ; });
    is_ready_ = false;
}

void BackendOptimazation::Run(){
    
    std::cout << "backend thread run! " << std::endl;
    while (true)
    {
        WaitForNotify();
        mtxPosegraph.lock();
        //添加odometry因子
        addOdomFactor();

        // 执行优化
        mylogger->debug("后端执行优化 : graph size {0}",graph_index);
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        if(loop_flag_){
            std::cout<<"loop update!"<<std::endl;
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
    
        mylogger->debug("update over");
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();
        mtxPosegraph.unlock();
        isamCurrentEstimate = isam->calculateEstimate();
        //更新一下上一帧位姿
        last_submap_pose = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

        //更新子图对应位姿！
        correctPose();
        loop_flag_ = false;
        
    } 
}

void BackendOptimazation::addOdomFactor()
{

    auto submap = submaps_[submaps_.size()-1];
    int index = submaps_.size()-1;
    if (index == 0)
    {
        // 添加先验
        // 旋转矩阵用Rot3用四元数构建（w,x,y,z）,
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(submap->local_rotation, submap->local_translation), priorNoise));
        // 变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(submap->local_rotation, submap->local_translation));
        //last_submap_pose = trans2gtsamPose(submap->local_rotation, submap->local_translation);
        graph_index = 0;
    }
    else{
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = last_submap_pose;
        gtsam::Pose3 poseTo = trans2gtsamPose(submap->local_rotation, submap->local_translation);
        //这里还需要看下这个里程计添加的是否正确！索引和poseFrome，已检查没问题！
        gtSAMgraph.add(BetweenFactor<Pose3>(graph_index,index, poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate.insert(index, poseTo);
        graph_index = index;
    }
    
}

//这个更新位姿需要再想一下是否每次都更新还是只有回环的时候才更新！
// void BackendOptimazation::correctPose()
// {
//     if(submaps_.empty())
//         return ;

//     int numPoses = isamCurrentEstimate.size();
//     if(numPoses!=submaps_.size()){
//         std::cout << "error: submaps.size() != gtsam size()" << std::endl;
//         return;
//     }

//     for (int i = 0; i < numPoses;i++){
//         auto submap = submaps_[i];

//         submap->local_translation.x() = isamCurrentEstimate.at<Pose3>(i).translation().x();
//         submap->local_translation.y() = isamCurrentEstimate.at<Pose3>(i).translation().y();
//         submap->local_translation.z() = isamCurrentEstimate.at<Pose3>(i).translation().z();
//         double roll,pitch,yaw;
//         roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
//         pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
//         yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

//         submap->local_rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
//         Eigen::Quaterniond q_orin(submap->local_rotation);
//         q_orin.normalized();
//         mylogger->debug("第 {0}子图位姿：{1} {2} {3} rotation: {4} {5} {6} {7}",i,submap->local_translation.x(),submap->local_translation.y(),submap->local_translation.z(),q_orin.x(),q_orin.y(),q_orin.z(),q_orin.w());
//     }
//     mylogger->debug("----------------update over---------------");
// }

void BackendOptimazation::correctPose()
{
    std::cout<<"矫正位姿"<<std::endl;

    int numPoses = isamCurrentEstimate.size();
    for (int i = 0; i < numPoses;i++){
        Poses_buf[i].p.x() = isamCurrentEstimate.at<Pose3>(i).translation().x();
        Poses_buf[i].p.y() = isamCurrentEstimate.at<Pose3>(i).translation().y();
        Poses_buf[i].p.z() = isamCurrentEstimate.at<Pose3>(i).translation().z();
        //这段转换需要看下写的对不对，检查了下应该没问题！
        // gtsam::Quaternion q_gtsam = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion();
        double roll,pitch,yaw;
        roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
        pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
        yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())  // 绕 Z 轴旋转
                            * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())   // 绕 Y 轴旋转
                            * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()); // 绕 X 轴旋转
        Poses_buf[i].q = q.normalized();
    }
}

gtsam::Pose3 BackendOptimazation::trans2gtsamPose(Eigen::Matrix3d rotation,Eigen::Vector3d translation)

{
    Eigen::Quaterniond q(rotation);
    gtsam::Rot3 R_gtsam(q.w(), q.x(), q.y(), q.z());
    //这个位移有待商榷，得再看下！就是正常的x,y,z
    gtsam::Point3 P_gtsam(translation.x(),translation.y(),translation.z());
    gtsam::Pose3 pose(R_gtsam, P_gtsam);
    return pose;
}

void BackendOptimazation::Insert_submap_2(DWIO::submap *submap,pcl::PointCloud<PointType>::Ptr submap_points, uint32_t submap_index){
    submaps_[submap_index] = submap;
    std::cout<<"Insert_submap"<<std::endl;
    //体素滤波后保存
    sor.setInputCloud(submap_points);
    sor.setLeafSize(0.03f, 0.03f, 0.03f);
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    sor.filter(*cloud_filtered);
    mtx_buf.lock();
    submaps_clouds_.push_back(cloud_filtered);
    mtx_buf.unlock();

    // 全部变换到lidar坐标系下！
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix << 1, 0, 0, 0, 0, 1, 0, 1, 0;
    Eigen::Matrix3d rotation_old = submap->local_rotation;
    Eigen::Matrix3d rotation_new = rotationMatrix*rotation_old*rotationMatrix.transpose();
    Eigen::Quaterniond q_new(rotation_new);
    Pose pose;
    pose.q = q_new.normalized();
    pose.p = rotationMatrix*submap->local_translation/1000;
    Poses_buf.push_back(pose);
    std::cout << "Pose_buf size: " << Poses_buf.size() << std::endl;
}

void BackendOptimazation::points_2_vertexs(Vertices& vertexs,pcl::PointCloud<PointType>::Ptr points){
    vertexs.resize(3,points->points.size());
    for (int i = 0; i < points->points.size();i++){
        vertexs(0,i) = points->points[i].x;
        vertexs(1,i) = points->points[i].y;
        vertexs(2,i) = points->points[i].z;
    }
    
}

bool BackendOptimazation::fast_robust_icp(int _loop_kf_idx, int _curr_kf_idx ,Eigen::Affine3f& correctionLidarFrame){
    pcl::PointCloud<PointType>::Ptr source_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr target_cloud(new pcl::PointCloud<PointType>());
    source_cloud = submaps_clouds_[_curr_kf_idx];
    target_cloud = submaps_clouds_[_loop_kf_idx];

    Vertices vertices_source;
    Vertices vertices_target ;
    points_2_vertexs(vertices_source,source_cloud);
    points_2_vertexs(vertices_target,target_cloud);   
    // std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;
    // std::cout << "target: " << vertices_target.rows() << " x " << vertices_target.cols() << std::endl;
    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
    // std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;
    double time;
    ICP::Parameters pars; 

    ///--- Execute registration
    std::cout << "begin registration..." << std::endl;
    DWIO::FRICP<3> fricp;
    double begin_reg = omp_get_wtime();
    double converge_rmse = 0;

    //     pars.use_init = true;
    //     pars.init_trans = init_trans;

    pars.f = ICP::WELSCH;
    pars.use_AA = true;
    double fitness = fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
    if(fitness>0.2){
        return false;
    }
    DWIO::MatrixXX res_trans = pars.res_trans;   
	std::cout << "Registration done!" << std::endl;
    double end_reg = omp_get_wtime();
    time = end_reg - begin_reg;
    std::cout<<"spend time = "<<time<<std::endl;
    vertices_source = scale * vertices_source; 


    res_trans.block(0,3,3,1) *= scale;
    Eigen::Affine3d res_T;
    res_T.linear() = res_trans.block(0,0,3,3);
    res_T.translation() = res_trans.block(0,3,3,1);

    Eigen::Matrix3d rotation_pose = res_trans.block(0,0,3,3);
    Eigen::Vector3d euler_angles = rotation_pose.eulerAngles(0,1,2);
    std::cout<<"res: "<<std::endl;
    std::cout<<res_trans<<std::endl;
    std::cout << "roll: " << euler_angles.x()/3.1415*180 << " pitch: " << euler_angles.y()/3.1415*180 << " yaw: " << euler_angles.z()/3.1415*180 
    << " x: " << res_trans(0,3) << " y: " << res_trans(1,3) << " z: " << res_trans(2,3) << std::endl;
    correctionLidarFrame = res_T.cast<float>();
    return true;
}

void BackendOptimazation::doOptimazation(){
    if(Poses_buf.size()<5){
        return;
    }
    std::cout << "do optimazation!" << std::endl;

    float x, y, z, roll, pitch, yaw;
    Eigen::Matrix3d rotation_pose;
    Eigen::Vector3d euler_angles;
    scManager.makeAndSaveScancontextAndKeys(*submaps_clouds_[0]);
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5).finished());
    gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(Poses_buf[0].q.toRotationMatrix(), Poses_buf[0].p), priorNoise));
    initialEstimate.insert(0, trans2gtsamPose(Poses_buf[0].q.toRotationMatrix(), Poses_buf[0].p));
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();    
    gtSAMgraph.resize(0);
    initialEstimate.clear();
    isamCurrentEstimate = isam->calculateEstimate();
    correctPose();

    //添加里程计
    for (int i = 1; i < 20;i++){
        scManager.makeAndSaveScancontextAndKeys(*submaps_clouds_[i]);
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-4).finished());
        gtsam::Pose3 poseFrom = trans2gtsamPose(Poses_buf[i-1].q.toRotationMatrix(), Poses_buf[i-1].p);
        gtsam::Pose3 poseTo = trans2gtsamPose(Poses_buf[i].q.toRotationMatrix(), Poses_buf[i].p);
        gtSAMgraph.add(BetweenFactor<Pose3>(i-1,i, poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(i, poseTo);

        //执行回环检测！
        bool loop_flag = false;
        auto detectResult = scManager.detectLoopClosureID();
        int SCclosestHistoryFrameID = detectResult.first;
        if(SCclosestHistoryFrameID != -1){
            loop_flag = true;
            const int prev_node_idx = SCclosestHistoryFrameID; // 这是找到的回环索引！
            const int curr_node_idx = i;
            cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;
            Eigen::Affine3f correctionLidarFrame;
            if(fast_robust_icp(prev_node_idx, curr_node_idx,correctionLidarFrame)){
                pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
                std::cout <<"Loop constraint roll: "<<roll/3.1415*180<<" pitch: "<<pitch/3.1415*180 <<" yaw: " << yaw/3.1415*180 << " ,translation x: " << x << " y: " << y << " z: " << z << std::endl;
                Eigen::Matrix3d rotation_pose = Poses_buf[curr_node_idx].q.toRotationMatrix();
                Eigen::Vector3d euler_angles = rotation_pose.eulerAngles(0,1,2);
                Eigen::Vector3d trans = Poses_buf[curr_node_idx].p;
                Eigen::Affine3f tWrong = pcl::getTransformation(trans.x(), trans.y(), trans.z(), euler_angles.x(), euler_angles.y(), euler_angles.z());
                Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
                pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                //利用先前位姿构造！
                rotation_pose = Poses_buf[prev_node_idx].q.toRotationMatrix();
                euler_angles = rotation_pose.eulerAngles(0,1,2);
                trans = Poses_buf[prev_node_idx].p;
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(euler_angles.x(), euler_angles.y(), euler_angles.z()), Point3(trans.x(), trans.y(), trans.z()));
                gtsam::Pose3 relative_pose = poseFrom.between(poseTo);
                std::cout << "Relative Pose: " << relative_pose << std::endl;
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(curr_node_idx, prev_node_idx, relative_pose, robustLoopNoise));
            }
        }
    }
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    gtSAMgraph.resize(0);
    initialEstimate.clear();
    isamCurrentEstimate = isam->calculateEstimate();
    correctPose();

    //将所有位姿转回到相机坐标系下！
    for (int i = 0; i < Poses_buf.size();i++){
        Eigen::Matrix3d rotationMatrix;
        rotationMatrix << 1, 0, 0, 0, 0, 1, 0, 1, 0;
        // rotationMatrix = rotationMatrix.transpose();
        Eigen::Matrix3d rotation_old = Poses_buf[i].q.toRotationMatrix();
        Eigen::Matrix3d rotation_new = rotationMatrix*rotation_old*rotationMatrix.transpose();
        Eigen::Quaterniond q_new(rotation_new);
        Pose pose;
        pose.q = q_new.normalized();
        pose.p = rotationMatrix*Poses_buf[i].p*1000;
        submaps_[i]->local_translation = pose.p;
        submaps_[i]->local_rotation = pose.q.toRotationMatrix();
    }
}


void BackendOptimazation::Insert_submap(DWIO::submap *submap,pcl::PointCloud<PointType>::Ptr submap_points, uint32_t submap_index)
{
    submaps_[submap_index] = submap;
    std::cout<<"Insert_submap"<<std::endl;
    //体素滤波后保存
    sor.setInputCloud(submap_points);
    sor.setLeafSize(0.03f, 0.03f, 0.03f);
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    sor.filter(*cloud_filtered);
    std::cout << "上锁前" << std::endl;
    mtx_buf.lock();
    std::cout << "上锁后" << std::endl;
    submaps_clouds_.push_back(cloud_filtered);
    loop_is_ready_ = true;
    mtx_buf.unlock();
    std::cout << "唤醒" << std::endl;
    loop_cond_.notify_all();

    std::cout << "notify loop detection" << std::endl;
    
    pcl::io::savePLYFileASCII( "../result/"+ std::to_string(submap_index)+".ply", *cloud_filtered);
    std::cout << "save ply" << std::endl;
}

std::map<uint32_t, DWIO::submap *>& BackendOptimazation::get_submaps()
{
    return submaps_;
}

void BackendOptimazation::LoopDetection() {

    std::cout<<"LoopDetection run!"<<std::endl;

    while(!stopLoopDetect) {

        {
            std::unique_lock<std::mutex> lock(mtx_buf);  
            loop_cond_.wait(lock, [this] { return loop_is_ready_ ; });
            loop_is_ready_ = false;
        }
        std::cout << "do loop detection" << std::endl;
        mtx_buf.lock();
        auto cloud = submaps_clouds_.back();

        scManager.makeAndSaveScancontextAndKeys(*cloud);
        auto detectResult = scManager.detectLoopClosureID();
        int SCclosestHistoryFrameID = detectResult.first;
        if( SCclosestHistoryFrameID != -1 ) {
            const int prev_node_idx = SCclosestHistoryFrameID; //这是找到的回环索引！
            const int curr_node_idx = submaps_clouds_.size()-1;

            auto it = loopIndexContainer.find(curr_node_idx);
            if(it != loopIndexContainer.end()){
                mtx_buf.unlock();
                continue;
            }

            // std::cout << "Loop detected! - between" << prev_node_idx << " and " << curr_node_idx <<"submap cloud size: "<<submaps_clouds_.size()<< std::endl;
            mylogger->debug("Loop detected! - between {0} and {1} ", prev_node_idx, curr_node_idx);

            pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
            cureKeyframeCloud = submaps_clouds_[curr_node_idx];
            targetKeyframeCloud = submaps_clouds_[prev_node_idx];
            mtx_buf.unlock();
            
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(0.1);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);
            icp.setInputSource(cureKeyframeCloud);
            icp.setInputTarget(targetKeyframeCloud);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            float loopFitnessScoreThreshold = 0.2;
            if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
                mylogger->debug("[SC loop] ICP fitness test failed ({0} > {1}). Add this SC loop.  ",icp.getFitnessScore(),loopFitnessScoreThreshold);
                continue;
            } else {
                mylogger->debug("[SC loop] ICP fitness test passed ({0} < {1}). Add this SC loop.  ",icp.getFitnessScore(),loopFitnessScoreThreshold);
            }

            loopIndexContainer[curr_node_idx] = prev_node_idx;

            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f correctionLidarFrame;
            correctionLidarFrame = icp.getFinalTransformation();
            Eigen::Matrix3f rotationMatrix;
            rotationMatrix << 1, 0, 0, 0, 0, 1, 0, 1, 0;
            
            correctionLidarFrame = rotationMatrix.transpose() * correctionLidarFrame * rotationMatrix; // 这么写可能有问题，可能是乘反了！
            correctionLidarFrame.translation() = correctionLidarFrame.translation() * 1000;

            Eigen::Vector3d euler_angles = submaps_[curr_node_idx]->local_rotation.eulerAngles(0,1,2);
            Eigen::Vector3d trans = submaps_[curr_node_idx]->local_translation;
            Eigen::Affine3f tWrong = pcl::getTransformation(trans.x(), trans.y(), trans.z(), euler_angles.x(), euler_angles.y(), euler_angles.z());

            pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
            // std::cout <<"Loop constraint roll: "<<roll<<" pitch: "<<pitch <<" yaw: " << yaw << " ,x: " << x << " y: " << y << " z: " << z << std::endl;
            mylogger->debug("Loop constraint roll: ({0} pitch: {1} yaw: {2}; x: {3}, y: {4}, z: {5}. ", roll,pitch,yaw,x,y,z);
            Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
            pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));

            
            
            //利用先前位姿构造！
            euler_angles = submaps_[prev_node_idx]->local_rotation.eulerAngles(0,1,2);
            trans = submaps_[prev_node_idx]->local_translation;
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(euler_angles.x(), euler_angles.y(), euler_angles.z()), Point3(trans.x(), trans.y(), trans.z()));
            gtsam::Pose3 relative_pose = poseFrom.between(poseTo);
            mtxPosegraph.lock();
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(curr_node_idx, prev_node_idx, relative_pose, robustLoopNoise));
            loop_flag_ = true;
            mtxPosegraph.unlock();  
            // std::cout<<"loop constraint: "<<x * 1000<<" "<< y * 1000<<" "<< z * 1000<<" rotation:"<< roll<<" " <<pitch<<" "<< yaw<<std::endl;
            // mylogger->debug("loop constraint :trans:{0},{1},{2} , rotation: {3} {4} {5}", x * 1000, y * 1000, z * 1000, roll, pitch, yaw);
        }else{
            mtx_buf.unlock();
        }

    }


}