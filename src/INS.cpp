//
// Created by baominjie on 23-7-6.
//
#include "INS.h"

namespace DWIO {

    void INS::GetInitialPose(Eigen::Matrix4d &pose, const double &init_y) {
#ifndef POSE_INTERPOLATION
        m_LiDAR_pose.setIdentity();
        Eigen::AngleAxisd pitch(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd yaw(Eigen::AngleAxisd(-m_odom_pose(2), Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd roll(Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R;
        R = roll * yaw * pitch;
        m_LiDAR_pose.block(0, 0, 3, 3) = R;
        m_LiDAR_pose(0, 3) = m_odom_pose(1);
        m_LiDAR_pose(1, 3) = init_y;
        m_LiDAR_pose(2, 3) = m_odom_pose(0);
#endif
        pose = m_LiDAR_pose * extrinsic_camera_odom;//这里应该需要处理反乘上当前子图的位姿
    }

}
