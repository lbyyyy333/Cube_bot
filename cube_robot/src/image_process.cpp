#include "image_process.hpp"

#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

cv::Mat ImageProcess::msgImageToMat(const sensor_msgs::msg::Image & msg) const
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("image_process"), "cv_bridge exception: %s", e.what());
    return cv::Mat();
  }
  return cv_ptr->image;
}

std::vector<std::vector<cv::Mat>> ImageProcess::Spit_to_9(const cv::Mat & roi)
{
  const int cell_w = roi.cols / 3;
  const int cell_h = roi.rows / 3;

  std::vector<std::vector<cv::Mat>> grid(3, std::vector<cv::Mat>(3));

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      const cv::Rect cell_rect(j * cell_w, i * cell_h, cell_w, cell_h);
      grid[i][j] = roi(cell_rect).clone();
    }
  }

  return grid;
}

cv::Vec3b ImageProcess::extractDominantColor(const cv::Mat & src, int K, bool show)
{
  cv::Mat data;
  src.convertTo(data, CV_32F);
  data = data.reshape(1, src.rows * src.cols);

  cv::Mat labels;
  cv::Mat centers;
  const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);

  cv::kmeans(data, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

  std::vector<int> counts(K, 0);
  for (int i = 0; i < labels.rows; i++) {
    const int label = labels.at<int>(i);
    counts[label]++;
  }

  const int dominant_idx = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());

  cv::Vec3b dominant_color;
  dominant_color[0] = static_cast<uchar>(centers.at<float>(dominant_idx, 0));
  dominant_color[1] = static_cast<uchar>(centers.at<float>(dominant_idx, 1));
  dominant_color[2] = static_cast<uchar>(centers.at<float>(dominant_idx, 2));

  if (show) {
    cv::Mat result(src.size(), src.type());

    for (int i = 0; i < labels.rows; i++) {
      const int label = labels.at<int>(i);
      cv::Vec3b color;

      color[0] = static_cast<uchar>(centers.at<float>(label, 0));
      color[1] = static_cast<uchar>(centers.at<float>(label, 1));
      color[2] = static_cast<uchar>(centers.at<float>(label, 2));

      result.at<cv::Vec3b>(i / src.cols, i % src.cols) = color;
    }

    cv::imshow("KMeans Result", result);
    if (cv::waitKey(1) == 'q') {
      cv::destroyWindow("KMeans Result");
    }
  }

  return dominant_color;
}

void ImageProcess::savePoints(const std::vector<cv::Point2f> & pts, const std::string & filename)
{
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    std::cout << "无法打开文件保存点" << std::endl;
    return;
  }

  for (const auto & p : pts) {
    ofs << p.x << " " << p.y << std::endl;
  }

  ofs.close();
  std::cout << "点已保存到 " << filename << std::endl;
}

bool ImageProcess::loadPoints(std::vector<cv::Point2f> & pts, const std::string & filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cout << "没有找到配置文件，使用默认点" << std::endl;
    return false;
  }

  pts.clear();

  float x = 0.0F;
  float y = 0.0F;
  while (ifs >> x >> y) {
    pts.push_back(cv::Point2f(x, y));
  }

  ifs.close();

  if (pts.size() != 6) {
    std::cout << "点数量错误，使用默认点" << std::endl;
    return false;
  }

  std::cout << "成功加载点" << std::endl;
  return true;
}

cv::Mat ImageProcess::extractQuadROI(const cv::Mat & img, const std::vector<cv::Point2f> & src_pts)
{
  const int width = 200;
  const int height = 200;

  const std::vector<cv::Point2f> dst_pts = {
    {0, 0},
    {width - 1, 0},
    {0, height - 1},
    {width - 1, height - 1}
  };

  const cv::Mat transform = cv::getPerspectiveTransform(src_pts, dst_pts);

  cv::Mat warped;
  cv::warpPerspective(img, warped, transform, cv::Size(width, height));

  return warped;
}

void ImageProcess::drawGrid(cv::Mat & display, const std::vector<cv::Point2f> & points) const
{
  if (points.size() != 6) {
    return;
  }

  for (const auto & p : points) {
    cv::circle(display, p, 5, cv::Scalar(0, 255, 0), -1);
  }

  cv::line(display, points[0], points[1], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[1], points[2], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[3], points[4], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[4], points[5], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[0], points[3], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[1], points[4], cv::Scalar(255, 0, 0), 2);
  cv::line(display, points[2], points[5], cv::Scalar(255, 0, 0), 2);
}

bool ImageProcess::allFacesFilled(const std::vector<std::vector<cv::Vec3b>> & classified_faces) const
{
  for (int face = 1; face <= 6; ++face) {
    for (const auto & color : classified_faces[face]) {
      if (color == cv::Vec3b(0, 0, 0)) {
        return false;
      }
    }
  }

  return true;
}

void ImageProcess::initPointsIfNeeded(std::vector<cv::Point2f> & points, const std::string & filename)
{
  if (!points.empty()) {
    return;
  }

  if (!loadPoints(points, filename)) {
    points = {
      cv::Point2f(200, 100),
      cv::Point2f(300, 90),
      cv::Point2f(400, 100),
      cv::Point2f(200, 300),
      cv::Point2f(300, 310),
      cv::Point2f(400, 300)
    };
  }
}

int ImageProcess::classifyColor(const cv::Vec3b & bgr_color) const
{
  const int b = bgr_color[0];
  const int g = bgr_color[1];
  const int r = bgr_color[2];

  if (
    b >= 150 && g >= 150 && r >= 150 &&
    std::abs(b - g) <= 40 &&
    std::abs(b - r) <= 40 &&
    std::abs(g - r) <= 40)
  {
    return 1;
  }

  if (r >= 150 && g <= 110 && b <= 110 && r >= g + 40 && r >= b + 40) {
    return 2;
  }

  if (g >= 140 && r <= 140 && b <= 120 && g >= r + 20 && g >= b + 20) {
    return 3;
  }

  if (r >= 150 && g >= 150 && b <= 110) {
    return 4;
  }

  if (r >= 180 && g >= 80 && g <= 180 && b <= 90 && r >= g + 20) {
    return 5;
  }

  if (b >= 140 && g <= 170 && r <= 130 && b >= g + 20 && b >= r + 20) {
    return 6;
  }

  return 0;
}

bool ImageProcess::bucketToColor(int bucket_index, Color & color) const
{
  switch (bucket_index) {
    case 1:
      color = U;
      return true;
    case 2:
      color = R;
      return true;
    case 3:
      color = F;
      return true;
    case 4:
      color = D;
      return true;
    case 5:
      color = L;
      return true;
    case 6:
      color = B;
      return true;
    default:
      return false;
  }
}

char ImageProcess::colorToChar(Color color) const
{
  switch (color) {
    case U:
      return 'U';
    case R:
      return 'R';
    case F:
      return 'F';
    case D:
      return 'D';
    case L:
      return 'L';
    case B:
      return 'B';
    default:
      return 'X';
  }
}

std::string ImageProcess::cubeToString(const std::vector<std::vector<Color>> & cube) const
{
  std::string cube_string;
  cube_string.reserve(54);

  for (const auto & face : cube) {
    for (const auto & color : face) {
      cube_string.push_back(colorToChar(color));
    }
  }

  return cube_string;
}

void ImageProcess::syncBucketToCube(
  const std::vector<std::vector<cv::Vec3b>> & classified_faces,
  std::vector<std::vector<Color>> & cube,
  int bucket_index) const
{
  if (bucket_index < 1 || bucket_index > 6) {
    return;
  }

  for (size_t sticker_index = 0; sticker_index < classified_faces[bucket_index].size(); ++sticker_index) {
    const int sticker_bucket = classifyColor(classified_faces[bucket_index][sticker_index]);
    Color sticker_color {};
    if (!bucketToColor(sticker_bucket, sticker_color)) {
      continue;
    }

    cube[bucket_index - 1][sticker_index] = sticker_color;
  }
}

int ImageProcess::storeFaceInBucket(
  int face_slot,
  const std::vector<std::vector<cv::Vec3b>> & color_grid,
  std::vector<std::vector<cv::Vec3b>> & classified_faces,
  std::vector<std::vector<Color>> & cube) const
{
  if (face_slot < 0 || static_cast<size_t>(face_slot) >= color_grid.size()) {
    return 0;
  }

  const int bucket_index = classifyColor(color_grid[face_slot][4]);
  if (bucket_index == 0) {
    return 0;
  }

  classified_faces[bucket_index] = color_grid[face_slot];
  syncBucketToCube(classified_faces, cube, bucket_index);
  return bucket_index;
}

void ImageProcess::mouseCallback(int event, int x, int y, int /*flags*/, void * userdata)
{
  auto * context = static_cast<MouseContext *>(userdata);
  if (context == nullptr || context->points == nullptr || context->dragging_point_index == nullptr) {
    return;
  }

  auto & points = *(context->points);
  auto & dragging_point_index = *(context->dragging_point_index);

  if (event == cv::EVENT_LBUTTONDOWN) {
    for (size_t i = 0; i < points.size(); i++) {
      if (cv::norm(points[i] - cv::Point2f(static_cast<float>(x), static_cast<float>(y))) < 5.0) {
        dragging_point_index = static_cast<int>(i);
        break;
      }
    }
  } else if (event == cv::EVENT_MOUSEMOVE) {
    if (dragging_point_index != -1) {
      points[dragging_point_index] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }
  } else if (event == cv::EVENT_LBUTTONUP) {
    dragging_point_index = -1;
  }
}
