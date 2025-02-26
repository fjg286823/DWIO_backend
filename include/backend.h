#pragma once


#include "../src/cuda/include/common.h"

#include "../src/hash/Submap.h"

#include <mutex>
#include <thread>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

//log
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

//pcl

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/ply_io.h>

//SC
#include "Scancontext.h"
#include "../fricp/FRICP.h"

using namespace gtsam;

struct Pose {
    Eigen::Vector3d p; // 位姿的平移部分
    Eigen::Quaterniond q; // 位姿的旋转部分
};

typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

class BackendOptimazation{
public:

    BackendOptimazation(){
        is_ready_ = false;
        loop_flag_ = false;
        loop_is_ready_ =false;
        // 启动一个线程运行Run函数，不能用thread一个非静态成员函数，而是在类外来调用这个函数
        // std::thread optimazation(Run);
        // optimazation.detach();
        graph_index = -1;
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;//当因子图中的某个变量的线性化误差超过这个阈值时，就会重新线性化这个变量1
        parameters.relinearizeSkip = 1;//表示每次优化后跳过多少次不进行重新线性化1。设置为1，那么每次优化后都会进行重新线性化
        // parameters.factorization = ISAM2Params::QR;//为了数值稳定!
        isam = new ISAM2(parameters);
        double loopNoiseScore = 1e-4; // constant is ok...
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
        robustLoopNoise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1), gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

        //ScanContext
        double scDistThres, scMaximumRadius;
        scDistThres = 0.5;
        scMaximumRadius = 20;
        scManager.setSCdistThres(scDistThres);
        scManager.setMaximumRadius(scMaximumRadius);

        mylogger = spdlog::get("spdlog");
        if(mylogger){
            std::cout<<"backend mylogger"<<std::endl;
        }

        //loop detection
        stopLoopDetect = false;
        LoopDetect = std::thread(&BackendOptimazation::LoopDetection, this);
    }

    ~BackendOptimazation(){
        delete isam;
        submaps_.clear();
        stopLoopDetect = true; // 通知线程停止
        if (LoopDetect.joinable()) {
            LoopDetect.join(); // 等待线程结束
        }
    }

    void LoopDetection();

    void addOdomFactor();

    gtsam::Pose3 trans2gtsamPose(Eigen::Matrix3d rotation,Eigen::Vector3d translation);

    void correctPose();

    void Run();
    void Notify();
    void WaitForNotify();
    void Insert_submap(DWIO::submap *submap,pcl::PointCloud<PointType>::Ptr submap_points, uint32_t submap_index);
    void Insert_submap_2(DWIO::submap *submap,pcl::PointCloud<PointType>::Ptr submap_points, uint32_t submap_index);
    std::map<uint32_t, DWIO::submap *> &get_submaps();
    std::vector<Pose>Poses_buf;
    void points_2_vertexs(Vertices& vertexs,pcl::PointCloud<PointType>::Ptr points);
    bool fast_robust_icp(int _loop_kf_idx, int _curr_kf_idx ,Eigen::Affine3f& correctionLidarFrame);
    void doOptimazation();
private:
    //子图应该放在这里
    std::map<uint32_t,DWIO::submap*>submaps_;
    std::vector<pcl::PointCloud<PointType>::Ptr> submaps_clouds_;
    //后端优化用到的因子图

    //线程异同步
    bool is_ready_;
    std::mutex run_mtx_;
    std::condition_variable cond_;
    // gtsam
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
    std::mutex mtxPosegraph;

    gtsam::Pose3 last_submap_pose;
    int graph_index;

    std::shared_ptr<spdlog::logger> mylogger;

    //loop flag
    bool loop_flag_;

    //体素滤波
    pcl::VoxelGrid<PointType> sor;

    //SC
    SCManager scManager;
    std::map<int, int> loopIndexContainer;//存放回环对！
    std::mutex mtx_buf;
    std::thread LoopDetect; // 工作线程
    std::queue<pcl::PointCloud<PointType>::Ptr> cloud_temp_buf;
    std::atomic<bool> stopLoopDetect;
    bool loop_is_ready_;
    std::condition_variable loop_cond_;
};