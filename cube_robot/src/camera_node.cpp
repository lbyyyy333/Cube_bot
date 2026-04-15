#include <iostream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

sensor_msgs::msg::Image Mat_to_ImageMsg(const cv::Mat & image)
{
  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = rclcpp::Clock().now();
  cv_image.header.frame_id = "camera_frame";
  cv_image.encoding = "bgr8";
  cv_image.image = image;
  return *cv_image.toImageMsg();
}

class Camera_Node : public rclcpp::Node
{
private:
  cv::VideoCapture cap_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr done_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr string_pub_;

  int photo_count_{0};
  int move_count_{0};
  bool waiting_done_{false};

  static constexpr int kTargetPhotos = 3;
  static constexpr int kTargetMoves = 2;

  bool capture_once()
  {
    cv::Mat frame;
    cap_ >> frame;
    if (frame.empty()) {
      RCLCPP_WARN(this->get_logger(), "Empty frame captured.");
      return false;
    }

    image_pub_->publish(Mat_to_ImageMsg(frame));
    ++photo_count_;
    RCLCPP_INFO(this->get_logger(), "Captured photo %d/%d.", photo_count_, kTargetPhotos);
    return true;
  }

  void send_rotate_command()
  {
    std_msgs::msg::String cmd;
    cmd.data = "CAPTURED";
    string_pub_->publish(cmd);
    ++move_count_;
    waiting_done_ = true;
    RCLCPP_INFO(this->get_logger(), "Sent rotate command %d/%d, waiting DONE.", move_count_, kTargetMoves);
  }

  void capture_and_step()
  {
    if (photo_count_ >= kTargetPhotos) {
      RCLCPP_INFO(this->get_logger(), "Sequence already finished.");
      return;
    }

    if (!capture_once()) {
      return;
    }

    if (move_count_ < kTargetMoves) {
      send_rotate_command();
    } else {
      waiting_done_ = false;
      RCLCPP_INFO(this->get_logger(), "Sequence complete: 3 photos and 2 moves done.");
    }
  }

  void done_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg->data != "DONE") {
      return;
    }

    if (!waiting_done_) {
      RCLCPP_WARN(this->get_logger(), "Received DONE but not waiting for it, ignoring.");
      return;
    }

    waiting_done_ = false;
    capture_and_step();
  }

public:
  explicit Camera_Node(const std::string & name)
  : Node(name)
  {
    cap_.open(0);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open camera.");
      throw std::runtime_error("Failed to open camera.");
    }

    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("OriginalImage", 10);
    string_pub_ = this->create_publisher<std_msgs::msg::String>("massege", 10);
    done_sub_ = this->create_subscription<std_msgs::msg::String>(
      "motion_done", 10, std::bind(&Camera_Node::done_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "%s has been started.", name.c_str());
    capture_and_step();
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Camera_Node>("camera_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
