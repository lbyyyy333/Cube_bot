#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/opencv.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "image_process.hpp"

class ImageProcessNode : public rclcpp::Node
{
public:
  explicit ImageProcessNode(const std::string & name)
  : Node(name)
  {
    RCLCPP_INFO(this->get_logger(), "%s has been started.", name.c_str());
    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "OriginalImage", 10, std::bind(&ImageProcessNode::image_callback, this, std::placeholders::_1));
    finish_pub_ = this->create_publisher<std_msgs::msg::String>("massege", 10);
    cube_result_pub_ = this->create_publisher<std_msgs::msg::String>("cube_result", 10);
  }

private:
  void publish_cube_result_if_ready()
  {
    if (!image_processor_.allFacesFilled(classified_faces_)) {
      return;
    }

    const std::string cube_string = image_processor_.cubeToString(cube_);
    if (cube_string == last_published_cube_result_) {
      return;
    }

    std_msgs::msg::String cube_result_msg;
    cube_result_msg.data = cube_string;
    cube_result_pub_->publish(cube_result_msg);
    last_published_cube_result_ = cube_string;

    std_msgs::msg::String finish_msg;
    finish_msg.data = "FINISH";
    finish_pub_->publish(finish_msg);

    RCLCPP_INFO(this->get_logger(), "All faces have been classified.");
    RCLCPP_INFO(this->get_logger(), "Published cube_result: %s", cube_string.c_str());
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const cv::Mat image = image_processor_.msgImageToMat(*msg);
    if (image.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received an empty image.");
      return;
    }

    image_processor_.initPointsIfNeeded(points_);

    cv::Mat display = image.clone();
    image_processor_.drawGrid(display, points_);
    cv::imshow("display", display);
    cv::setMouseCallback("display", ImageProcess::mouseCallback, &mouse_context_);

    const cv::Mat warped1 =
      image_processor_.extractQuadROI(image, {points_[0], points_[1], points_[3], points_[4]});
    const cv::Mat warped2 =
      image_processor_.extractQuadROI(image, {points_[1], points_[2], points_[4], points_[5]});
    const auto grid1 = image_processor_.Spit_to_9(warped1);
    const auto grid2 = image_processor_.Spit_to_9(warped2);

    if (cv::waitKey(10) == 'x') {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          color_grid_[1][i * 3 + j] = image_processor_.extractDominantColor(grid1[i][j], 3, false);
          color_grid_[2][i * 3 + j] = image_processor_.extractDominantColor(grid2[i][j], 3, false);
        }
      }

      for (int i = 1; i <= 2; i++) {
        const int bucket_index =
          image_processor_.storeFaceInBucket(i, color_grid_, classified_faces_, cube_);
        if (bucket_index == 0) {
          RCLCPP_WARN(this->get_logger(), "Face %d center color is not matched to W/R/G/Y/O/B.", i);
          continue;
        }

        RCLCPP_INFO(
          this->get_logger(),
          "Face %d stored in bucket %d and synced to cube (1:white/U 2:red/R 3:green/F 4:yellow/D 5:orange/L 6:blue/B).",
          i,
          bucket_index);
      }

      if (image_processor_.allFacesFilled(classified_faces_)) {
        RCLCPP_INFO(
          this->get_logger(),
          "Current cube string: %s",
          image_processor_.cubeToString(cube_).c_str());
      }
      publish_cube_result_if_ready();
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr finish_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cube_result_pub_;
  std::vector<std::vector<Color>> cube_{6, std::vector<Color>(9)};
  std::vector<std::vector<cv::Vec3b>> color_grid_{3, std::vector<cv::Vec3b>(9)};
  std::vector<std::vector<cv::Vec3b>> classified_faces_{7, std::vector<cv::Vec3b>(9)};
  std::string last_published_cube_result_;
  ImageProcess image_processor_;
  std::vector<cv::Point2f> points_;
  int dragging_point_index_{-1};
  ImageProcess::MouseContext mouse_context_{&points_, &dragging_point_index_};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageProcessNode>("image_process_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
