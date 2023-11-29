/********************************************************************************
** @auth�� taify
** @date�� 2022/12/10
** @Ver :  V1.0.0
** @desc�� ������
*********************************************************************************/


#include <iostream>
#include <iomanip>
#include <ctime>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "yolov5.h"
#include "kalmanboxtracker.h"
#include "sort.h"


const int frame_nums = 71;


int main(int argc, char** argv)
{
	Sort mot_tracker = Sort(1, 3, 0.3);
	YOLOv5 yolov5 = YOLOv5("yolov5s.onnx");
	srand(time(0));
	std::vector<cv::Vec3b> colors(32);
	for (size_t i = 0; i < colors.size(); i++)
	{
		colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
	}

	std::string prefix = "./img1/";
	for (size_t i = 1; i <= frame_nums; i++)
	{
		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << i;
		std::string frame = prefix + ss.str() + ".jpg";
		std::cout << "******************************************************************* " << frame << std::endl;		

		cv::Mat image = cv::imread(frame);
		std::vector<cv::Rect> detections = yolov5.infer(image);
		//std::sort(detections.begin(), detections.end(), [](cv::Rect rect1, cv::Rect rect2) {return rect1.x < rect2.x; });

		std::vector<std::vector<float>> trackers = mot_tracker.update(detections);

		//for (auto detection : detections)
			//cv::rectangle(image, detection, cv::Scalar(0, 0, 255), 1);

		for (auto tracker : trackers)
			cv::rectangle(image, cv::Rect(tracker[0], tracker[1], tracker[2]- tracker[0], tracker[3]- tracker[1]), colors[int(tracker[4])%32], 1);

		cv::imshow("image", image);
		cv::waitKey(1);
	}
	cv::destroyAllWindows();

	return 0;
}