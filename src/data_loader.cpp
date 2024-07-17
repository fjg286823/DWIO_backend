//
// Created by baominjie on 2023/6/1.
//

#include "data_loader.h"

data_loader::data_loader(const std::string &file_path) {
    std::string frame_filename = file_path + "associations.txt";
    m_frame_file.open(frame_filename, std::fstream::in);
    std::string pose_filename = file_path + "time_pose.bin";
    m_pose_file.open(pose_filename, std::ios::in | std::ios::binary);
    if (!m_frame_file.is_open() || !m_pose_file.is_open()) {
        std::cout << "[Data loader] File open failed!" << std::endl;
        exit(0);
    }
}

data_loader::~data_loader() {
    m_frame_file.close();
    m_pose_file.close();
}

void data_loader::GetNextFrame(const std::string &file_path, cv::Mat &color_img, cv::Mat &depth_map) {
    std::string line_data, read_data;
    getline(m_frame_file, line_data);
    std::stringstream input_string_stream(line_data);
    input_string_stream >> read_data;
    m_color_img_time = std::stod(read_data.substr(0, read_data.length() - 4)) / 1e6;
    input_string_stream >> read_data;
    m_color_img_name = file_path + read_data;
    input_string_stream >> read_data;
    m_depth_map_time = std::stod(read_data.substr(0, read_data.length() - 4)) / 1e6;
    input_string_stream >> read_data;
    m_depth_map_name = file_path + read_data;
    depth_map = cv::imread(m_depth_map_name, -1);
    if (depth_map.empty()) {
        std::cout << "[Data loader] Depth image open failed!" << std::endl;
        depth_map = cv::Mat::zeros(depth_map.rows, depth_map.cols, CV_16UC1);
    }
    depth_map.convertTo(depth_map, CV_16UC1, 1.0);
    color_img = cv::imread(m_color_img_name);
    if (color_img.empty()) {
        color_img = cv::Mat::zeros(depth_map.rows, depth_map.cols, CV_8UC3);
    }
}

void data_loader::GetNextPose(CartographerPose &pose_2d) {
    m_pose_file.read(reinterpret_cast<char *>(&pose_2d), sizeof(CartographerPose));
    pose_2d.recv_ts = pose_2d.recv_ts - m_odom_time_bias;
    pose_2d.sync_ts = pose_2d.sync_ts - m_odom_time_bias;
}

bool data_loader::FrameHasMore() {
    return (m_frame_file.peek() != EOF);
}

bool data_loader::PoseHasMore() {
    return (m_pose_file.peek() != EOF);
}
