//
// Created by baominjie on 23-7-6.
//

#ifndef DWIO_INS_H
#define DWIO_INS_H

#include "data.h"
#include "utility.hpp"

#define ODOM_BUFFER_SIZE 100

namespace DWIO {
    class INS {
    public:
        INS() {
            //扫地机的外参
            // double angle = 0 * M_PI / 180.0;

            // extrinsic_camera_odom.setIdentity();//-26.;105.
            // extrinsic_camera_odom(0, 3) = -26.;//-120./26
            // extrinsic_camera_odom(1, 3) = 171.;
            // extrinsic_camera_odom(2, 3) = 105.;//413.

            // extrinsic_camera_odom(0,0) = std::cos(angle);
            // extrinsic_camera_odom(0,2) = std::sin(angle);
            // extrinsic_camera_odom(2,0) = -std::sin(angle);
            // extrinsic_camera_odom(2,2) = std::cos(angle); 

            //小车外参
            extrinsic_camera_odom.setIdentity();//-26.;105.
            extrinsic_camera_odom(0, 3) = -26.;//-120./26
            extrinsic_camera_odom(1, 3) = 302.21;
            extrinsic_camera_odom(2, 3) = 164.;//413.

            extrinsic_camera_odom_inv = extrinsic_camera_odom.inverse();
        };

        ~INS() = default;

        void GetInitialPose(Eigen::Matrix4d &pose, const double &init_y);

        Eigen::Vector3d m_odom_pose;

        Eigen::Matrix4d m_LiDAR_pose;
        double m_LiDAR_pose_time{};
        Eigen::Matrix4d m_LiDAR_pose_last;
        double m_LiDAR_pose_time_last{};
        Eigen::Matrix4d m_LiDAR_pose_now;
        double m_LiDAR_pose_time_now{};

        Eigen::Matrix4d extrinsic_camera_odom;
        Eigen::Matrix4d extrinsic_camera_odom_inv;
    };
}

#endif //DWIO_INS_H
