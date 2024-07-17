//
// Created by baominjie on 2023/6/1.
//

#ifndef DWIO_DATA_LOADER_H
#define DWIO_DATA_LOADER_H

#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "DWIO.h"

class data_loader {
public:
    explicit data_loader(const std::string &file_path);

    ~data_loader();

    void GetNextFrame(const std::string &file_path, cv::Mat &color_img, cv::Mat &depth_map);

    void GetNextPose(CartographerPose &pose_2d);

    bool FrameHasMore();

    bool PoseHasMore();

    double m_color_img_time{};
    double m_depth_map_time{};

private:
    std::ifstream m_frame_file;
    std::ifstream m_pose_file;

    std::string m_color_img_name;
    std::string m_depth_map_name;

    unsigned long long m_odom_time_bias = 0;
};

#endif //DWIO_DATA_LOADER_H
