#include "DWIO.h"
#include "data_loader.h"
#include <pangolin/pangolin.h>
#include <csignal>

std::atomic<bool> stop(false);

void sigint_handler(int sig) {
    if (sig == SIGINT) {
        std::cout << "[Main] DWIO terminated!\n";
        stop = true;
    }
}

int main() {
    signal(SIGINT, sigint_handler);

    const std::string data_file("../sample_data.yaml");
    const std::string option_file("../option.yaml");
    const std::string camera_file("../Astra1.yaml");

    const DWIO::CameraConfiguration camera_config(camera_file);
    const DWIO::DataConfiguration data_config(data_file);
    const DWIO::OptionConfiguration option_config(option_file);

    pangolin::View shaded_cam;
    pangolin::View depth_cam;
    pangolin::View color_cam;

    pangolin::GlTexture image_texture;
    pangolin::GlTexture shaded_texture;
    pangolin::GlTexture depth_texture;

    pangolin::CreateWindowAndBind("DWIO", 2880, 1440);
    color_cam = pangolin::Display("color_cam").SetAspect((float) camera_config.image_width / (float) camera_config.image_height);
    shaded_cam = pangolin::Display("shaded_cam").SetAspect((float) camera_config.image_width / (float) camera_config.image_height);
    depth_cam = pangolin::Display("depth_cam").SetAspect((float) camera_config.image_width / (float) camera_config.image_height);


    glEnable(GL_DEPTH_TEST);

    // 创建一个摄像机
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(600,800,200,200,300,400,0.1,10000),
        pangolin::ModelViewLookAt(0,-10,10, 0,0,0, pangolin::AxisZ)
    );
    const int UI_WIDTH = 180;
    pangolin::View& d_cam = pangolin::CreateDisplay()
        //.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::Display("window")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(shaded_cam)
            .AddDisplay(color_cam)
            .AddDisplay(depth_cam)
            .AddDisplay(d_cam);
            

    image_texture = pangolin::GlTexture(camera_config.image_width,
                                        camera_config.image_height,
                                        GL_RGB,
                                        false,
                                        0,
                                        GL_RGB,
                                        GL_UNSIGNED_BYTE);
    shaded_texture = pangolin::GlTexture(camera_config.image_width,
                                         camera_config.image_height,
                                         GL_RGB,
                                         false,
                                         0,
                                         GL_RGB,
                                         GL_UNSIGNED_BYTE);
    depth_texture = pangolin::GlTexture(camera_config.image_width,
                                        camera_config.image_height,
                                        GL_LUMINANCE,
                                        false,
                                        0,
                                        GL_LUMINANCE,
                                        GL_UNSIGNED_BYTE);

    cv::Mat color_img;
    cv::Mat depth_map;
    cv::Mat shaded_img(camera_config.image_height, camera_config.image_width, CV_8UC3);

    CartographerPose cartographer_pose{};
    double pose_last_time = 0, pose_now_time = 0;
    std::deque<CartographerPose> pose_2d_buffer;
    std::deque<double> interpolation_ratio_buffer;

    // std::cout << "initial particle" << std::endl;
    // std::vector<int> particle_level = {10240, 3072, 1024};
    // DWIO::internal::QuaternionData PST(particle_level,data_config.PST_path);
    // DWIO::internal::ParticleSearchData search_data(particle_level);

    DWIO::Pipeline pipeline{camera_config, data_config, option_config};

    data_loader loader(data_config.datasets_path);
    int loop_count=0;
    int TotalTriangles =0;
    bool is_tracking_success =false;
    while (loader.PoseHasMore() && (!stop)) {

        loader.GetNextPose(cartographer_pose);
        while (loader.FrameHasMore() && pipeline.m_img_time_buffer.size() < 500) {
            loader.GetNextFrame(data_config.datasets_path, color_img, depth_map);
            pipeline.m_color_img_buffer.push_back(color_img);
            pipeline.m_depth_img_buffer.push_back(depth_map);
            pipeline.m_img_time_buffer.push_back(loader.m_depth_map_time);
        }

        pose_2d_buffer.push_back(cartographer_pose);
        if (pose_2d_buffer.size() > 2)
            pose_2d_buffer.pop_front();

        if (pose_2d_buffer.size() == 1) {
            pipeline.m_anchor_point = pipeline.m_pose.block(0, 3, 3, 1);
            pose_last_time = (double) (cartographer_pose.recv_ts) / 1e6;
            pose_now_time = (double) (cartographer_pose.recv_ts) / 1e6;
        } else {
            pose_now_time = pose_last_time + (double) (pose_2d_buffer[1].laser_ts - pose_2d_buffer[0].laser_ts) / 1e6;
            for (const auto &img_time: pipeline.m_img_time_buffer) {
                if (img_time >= pose_last_time && img_time < pose_now_time) {
                    interpolation_ratio_buffer.push_back((img_time - pose_last_time) / (pose_now_time - pose_last_time));
                }
                if (img_time >= pose_now_time)
                    break;
            }
        }

        std::vector<TimestampedPose> pose_interpolation_buffer = pipeline.PoseInterpolation(
                Eigen::Vector3d
                        (pose_2d_buffer[0].pose_x * 1000 ,
                         -pose_2d_buffer[0].pose_y * 1000 ,
                         pose_2d_buffer[0].pose_theta),
                Eigen::Vector3d
                        (pose_2d_buffer[1].pose_x * 1000 ,
                         -pose_2d_buffer[1].pose_y * 1000 ,
                         pose_2d_buffer[1].pose_theta),
                pose_last_time,
                pose_now_time,
                interpolation_ratio_buffer,
                0.0);
        

        // loop_count++;
        // if(loop_count==200)
        //     break;
        for (const auto &timestamped_pose: pose_interpolation_buffer) {
            while (!pipeline.m_img_time_buffer.empty()) {
                if ((double) timestamped_pose.time * 1e-6 > pipeline.m_img_time_buffer.front()) {
                    pipeline.m_img_time_buffer.pop_front();
                    pipeline.m_depth_img_buffer.pop_front();
                    pipeline.m_color_img_buffer.pop_front();
                } else
                    break;
            }
            pipeline.m_INS.m_LiDAR_pose_time = (double) timestamped_pose.time * 1e-6;
            pipeline.m_INS.m_LiDAR_pose = timestamped_pose.pose;
            if (!pipeline.m_depth_img_buffer.empty()) {

                // is_tracking_success = pipeline.ProcessFrameHash(pipeline.m_depth_img_buffer.front(),
                //                           pipeline.m_color_img_buffer.front(),
                //                           shaded_img,TotalTriangles);
                is_tracking_success = pipeline.ProcessFrameMutil(pipeline.m_depth_img_buffer.front(),
                                          pipeline.m_color_img_buffer.front(),
                                          shaded_img);

                std::cout<<"-----------------------------------------------------"<<std::endl;
            } else
                break;

            glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT); 
            color_cam.Activate();
            image_texture.Upload(pipeline.m_color_img_buffer.front().data, GL_BGR, GL_UNSIGNED_BYTE);
            image_texture.RenderToViewportFlipY();
            depth_cam.Activate();
            pipeline.m_depth_img_buffer.front().convertTo(pipeline.m_depth_img_buffer.front(), CV_8U, 256 / 10000.0);
            depth_texture.Upload(pipeline.m_depth_img_buffer.front().data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
            depth_texture.RenderToViewportFlipY();
            shaded_cam.Activate();
            shaded_texture.Upload(shaded_img.data, GL_BGR, GL_UNSIGNED_BYTE);
            shaded_texture.RenderToViewportFlipY();

            pangolin::FinishFrame();

        }
        pose_last_time = pose_now_time;

        
    }
    //添加一个函数对最后一个子图进行提取和回环检测
    pipeline.get_last_submap();
    pipeline.SaveMap();

    return 0;
}
