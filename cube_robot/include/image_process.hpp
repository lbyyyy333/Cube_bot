#ifndef CUBE_ROBOT_IMAGE_PROCESS_HPP_
#define CUBE_ROBOT_IMAGE_PROCESS_HPP_

#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include <vector>

enum Color
{
  U,
  R,
  F,
  D,
  L,
  B,
};

class ImageProcess
{
public:
  struct MouseContext
  {
    std::vector<cv::Point2f> * points;
    int * dragging_point_index;
  };

  cv::Mat msgImageToMat(const sensor_msgs::msg::Image & msg) const;
  std::vector<std::vector<cv::Mat>> Spit_to_9(const cv::Mat & roi);
  cv::Vec3b extractDominantColor(const cv::Mat & src, int K, bool show = true);
  void savePoints(const std::vector<cv::Point2f> & pts, const std::string & filename);
  bool loadPoints(std::vector<cv::Point2f> & pts, const std::string & filename);
  cv::Mat extractQuadROI(const cv::Mat & img, const std::vector<cv::Point2f> & src_pts);
  void drawGrid(cv::Mat & display, const std::vector<cv::Point2f> & points) const;
  bool allFacesFilled(const std::vector<std::vector<cv::Vec3b>> & classified_faces) const;
  void initPointsIfNeeded(
    std::vector<cv::Point2f> & points, const std::string & filename = "cube_pts.txt");
  int classifyColor(const cv::Vec3b & bgr_color) const;
  bool bucketToColor(int bucket_index, Color & color) const;
  char colorToChar(Color color) const;
  std::string cubeToString(const std::vector<std::vector<Color>> & cube) const;
  void syncBucketToCube(
    const std::vector<std::vector<cv::Vec3b>> & classified_faces,
    std::vector<std::vector<Color>> & cube,
    int bucket_index) const;
  int storeFaceInBucket(
    int face_slot,
    const std::vector<std::vector<cv::Vec3b>> & color_grid,
    std::vector<std::vector<cv::Vec3b>> & classified_faces,
    std::vector<std::vector<Color>> & cube) const;
  static void mouseCallback(int event, int x, int y, int flags, void * userdata);
};

#endif  // CUBE_ROBOT_IMAGE_PROCESS_HPP_
