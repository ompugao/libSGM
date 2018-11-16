/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sl/Camera.hpp>

#include <libsgm.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
template <class... Args>
static std::string format_string(const char* fmt, Args... args)
{
	const int BUF_SIZE = 1024;
	char buf[BUF_SIZE];
	std::snprintf(buf, BUF_SIZE, fmt, args...);
	return std::string(buf);
}

struct device_buffer {
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

int main(int argc, char* argv[]) {	
	
	const int disp_size = 128;
	
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION_HD720;
	sl::ERROR_CODE err = zed.open(initParameters);
	if (err != sl::SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}
	const int width = static_cast<int>(zed.getResolution().width);
	const int height = static_cast<int>(zed.getResolution().height);

	sl::Mat d_zed_image_l(zed.getResolution(), sl::MAT_TYPE_8U_C1, sl::MEM_GPU);
	sl::Mat d_zed_image_r(zed.getResolution(), sl::MAT_TYPE_8U_C1, sl::MEM_GPU);

  sl::CalibrationParameters calib_param = zed.getCameraInformation(zed.getResolution()).calibration_parameters; // calibration_paramters_raw


  cv::Mat cameraMatrixLeft(3,3,CV_64F), distCoeffsLeft(5,1,CV_64F);
  cv::Mat cameraMatrixRight(3,3,CV_64F), distCoeffsRight(5,1,CV_64F);
  cv::Size imageSize;
  imageSize.width = width;
  imageSize.height = height;
  cameraMatrixLeft.at<double>(0,0) = calib_param.left_cam.fx;
  cameraMatrixLeft.at<double>(0,2) = calib_param.left_cam.cx;
  cameraMatrixLeft.at<double>(1,1) = calib_param.left_cam.fy;
  cameraMatrixLeft.at<double>(1,2) = calib_param.left_cam.cy;
  cameraMatrixLeft.at<double>(2,2) = 1;
  cameraMatrixRight.at<double>(0,0) = calib_param.right_cam.fx;
  cameraMatrixRight.at<double>(0,2) = calib_param.right_cam.cx;
  cameraMatrixRight.at<double>(1,1) = calib_param.right_cam.fy;
  cameraMatrixRight.at<double>(1,2) = calib_param.right_cam.cy;
  cameraMatrixRight.at<double>(2,2) = 1;
  for (int i = 0; i < 5; i++) {
    distCoeffsLeft.at<double>(i) = calib_param.left_cam.disto[i];
    distCoeffsRight.at<double>(i) = calib_param.right_cam.disto[i];
  }
  sl::Rotation rot;
  rot.setRotationVector(calib_param.R);
  cv::Mat R(3, 3, CV_64F), T(3, 1, CV_64F);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R.at<double>(i, j) = rot(i, j);
    }
  }
  for (int i = 0; i < 3; i++) {
    T.at<double>(i) = calib_param.T[i];
  }
  std::cout << cameraMatrixLeft << std::endl;
  std::cout << cameraMatrixRight << std::endl;
  std::cout << distCoeffsLeft << std::endl;
  std::cout << distCoeffsRight << std::endl;
  std::cout << R << std::endl;
  std::cout << T << std::endl;

  cv::Mat R1, R2, P1, P2, Q;
  cv::Rect validRoi[2];
  cv::stereoRectify(cameraMatrixLeft, distCoeffsLeft,
        cameraMatrixRight, distCoeffsRight,
        imageSize, R, T, R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
  std::cout << " -- "  << std::endl;
  std::cout << Q << std::endl;
  std::cout << validRoi[0] << std::endl;
  std::cout << validRoi[1] << std::endl;
  std::cout << " -- "  << std::endl;

	const int input_depth = 8;
	const int input_bytes = input_depth * width * height / 8;
	const int output_depth = 8;
	const int output_bytes = output_depth * width * height / 8;

	sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

	cv::Mat disparity(height, width, CV_8U);
	cv::Mat disparity_8u, disparity_color;
  cv::Mat_<cv::Vec3f> dense_points;

  pcl::visualization::PCLVisualizer viewer ("Viewer");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	device_buffer d_image_l(input_bytes), d_image_r(input_bytes), d_disparity(output_bytes);
	while (1) {
		if (zed.grab() == sl::SUCCESS) {
			zed.retrieveImage(d_zed_image_l, sl::VIEW_LEFT_GRAY, sl::MEM_GPU);
			zed.retrieveImage(d_zed_image_r, sl::VIEW_RIGHT_GRAY, sl::MEM_GPU);
		} else continue;

		cudaMemcpy2D(d_image_l.data, width, d_zed_image_l.getPtr<uchar>(sl::MEM_GPU), d_zed_image_l.getStep(sl::MEM_GPU), width, height, cudaMemcpyDeviceToDevice);
		cudaMemcpy2D(d_image_r.data, width, d_zed_image_r.getPtr<uchar>(sl::MEM_GPU), d_zed_image_r.getStep(sl::MEM_GPU), width, height, cudaMemcpyDeviceToDevice);

		const auto t1 = std::chrono::system_clock::now();

		sgm.execute(d_image_l.data, d_image_r.data, d_disparity.data);
		cudaDeviceSynchronize();

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const double fps = 1e6 / duration;

		cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);

		disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		cv::putText(disparity_color, format_string("sgm execution time: %4.1f[msec] %4.1f[FPS]", 1e-3 * duration, fps),
			cv::Point(50, 50), 2, 0.75, cv::Scalar(255, 255, 255));

		cv::imshow("disparity", disparity_color);
    cv::reprojectImageTo3D(disparity_8u, dense_points, Q, true);
    //cloud->points.resize(dense_points.rows * dense_points.cols);
    //cloud->width = dense_points.rows;
    //cloud->height = dense_points.cols;
    cloud->points.clear();
    std::cout << "  ------------- "  << std::endl;
    for(int i = 0; i < dense_points.rows; i++) {
      for(int j = 0; j < dense_points.cols; j++) {
        int index = i*dense_points.rows + j;
        //std::cout << i << " , " << j << " : " << dense_points.at<cv::Vec3f>(i,j)[0] << std::endl;
        if (!std::isnan(dense_points.at<cv::Vec3f>(i,j)[0]) && !std::isinf(dense_points.at<cv::Vec3f>(i,j)[0])
            && !std::isnan(dense_points.at<cv::Vec3f>(i,j)[1]) && !std::isinf(dense_points.at<cv::Vec3f>(i,j)[1])
            && !std::isnan(dense_points.at<cv::Vec3f>(i,j)[2]) && !std::isinf(dense_points.at<cv::Vec3f>(i,j)[2])
           ) {
          pcl::PointXYZ point;
          point.x = dense_points.at<cv::Vec3f>(i,j)[0];
          point.y = dense_points.at<cv::Vec3f>(i,j)[1];
          point.z = dense_points.at<cv::Vec3f>(i,j)[2];
          cloud->points.push_back(point);
        }
        //cloud->points[index].x = dense_points.at<cv::Vec3f>(i,j)[0];
        //cloud->points[index].y = dense_points.at<cv::Vec3f>(i,j)[1];
        //cloud->points[index].z = dense_points.at<cv::Vec3f>(i,j)[2];
        // when color needs to be added:
        //uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
        //cloud->points[index].rgb = *reinterpret_cast<float*>(&rgb);
      }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    bool ret = viewer.updatePointCloud<pcl::PointXYZ>(cloud, "input");
    if (!ret) {
      viewer.addPointCloud<pcl::PointXYZ>(cloud, "input");
    }

    viewer.spinOnce(10);
		const char c = cv::waitKey(1);
		if (c == 27) // ESC
			break;
	}
	zed.close();
	return 0;
}
