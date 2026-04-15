#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "serial_driver/serial_port.hpp"

class SerialBridgeNode : public rclcpp::Node
{
private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr serial_rx_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr motion_done_pub_;

  drivers::common::IoContext ctx_;
  std::shared_ptr<drivers::serial_driver::SerialPort> serial_port_;

  std::thread rx_thread_;
  std::atomic<bool> running_{false};
  std::string rx_accumulator_;

  void ASCII_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "接受到了 '%s'", msg->data.c_str());
    if (!serial_port_ || !serial_port_->is_open()) {
      RCLCPP_WARN(this->get_logger(), "串口未打开，跳过发送");
      return;
    }

    const std::string & data_to_send = msg->data;
    const std::vector<uint8_t> data_uint8(data_to_send.begin(), data_to_send.end());
    try {
      serial_port_->send(data_uint8);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "串口发送失败: %s", e.what());
    }
  }

  void handle_line(const std::string & line)
  {
    if (line.empty()) {
      return;
    }

    std_msgs::msg::String rx_msg;
    rx_msg.data = line;
    serial_rx_pub_->publish(rx_msg);
    RCLCPP_INFO(this->get_logger(), "串口接收: '%s'", line.c_str());

    if (line == "DONE") {
      std_msgs::msg::String done_msg;
      done_msg.data = line;
      motion_done_pub_->publish(done_msg);
      RCLCPP_INFO(this->get_logger(), "已发布动作完成信号到 motion_done");
    }
  }

  void receive_loop()
  {
    std::vector<uint8_t> buffer(2048, 0U);

    while (rclcpp::ok() && running_) {
      try {
        const size_t n = serial_port_->receive(buffer);
        if (n == 0) {
          continue;
        }

        for (size_t i = 0; i < n; ++i) {
          const char ch = static_cast<char>(buffer[i]);
          if (ch == '\n') {
            handle_line(rx_accumulator_);
            rx_accumulator_.clear();
          } else if (ch != '\r') {
            rx_accumulator_.push_back(ch);
          }
        }
      } catch (const std::exception & e) {
        if (running_) {
          RCLCPP_ERROR(this->get_logger(), "串口接收失败: %s", e.what());
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
    }
  }

public:
  explicit SerialBridgeNode(const std::string & name)
  : Node(name), ctx_(1)
  {
    const drivers::serial_driver::SerialPortConfig config(
      115200U,
      drivers::serial_driver::FlowControl::NONE,
      drivers::serial_driver::Parity::NONE,
      drivers::serial_driver::StopBits::ONE);

    serial_port_ = std::make_shared<drivers::serial_driver::SerialPort>(ctx_, "/dev/ttyUSB0", config);
    try {
      serial_port_->open();
      RCLCPP_INFO(this->get_logger(), "串口已打开，开始等待话题数据与下位机反馈");
    } catch (const std::exception & e) {
      RCLCPP_FATAL(this->get_logger(), "打开串口失败: %s", e.what());
      throw;
    }

    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "cube_result", 10, std::bind(&SerialBridgeNode::ASCII_callback, this, std::placeholders::_1));

    serial_rx_pub_ = this->create_publisher<std_msgs::msg::String>("serial_rx", 10);
    motion_done_pub_ = this->create_publisher<std_msgs::msg::String>("motion_done", 10);

    running_ = true;
    rx_thread_ = std::thread(&SerialBridgeNode::receive_loop, this);
  }

  ~SerialBridgeNode() override
  {
    running_ = false;

    if (serial_port_ && serial_port_->is_open()) {
      try {
        serial_port_->close();
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "关闭串口失败: %s", e.what());
      }
    }

    if (rx_thread_.joinable()) {
      rx_thread_.join();
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SerialBridgeNode>("serial_bridge_node");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
