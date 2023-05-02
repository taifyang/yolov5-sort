/********************************************************************************
** @auth£º taify
** @date£º 2022/12/10
** @Ver :  V1.0.0
** @desc£º Ö÷º¯Êý
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
	cv::dnn::Net net = cv::dnn::readNet("./model/yolov5s.onnx");
	std::vector<cv::Vec3b> colors(32);
	srand(time(0));
	for (size_t i = 0; i < colors.size(); i++)
		colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);


	std::string prefix = "./data/";
	for (size_t i = 1; i <= frame_nums; i++)
	{
		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << i;
		std::string frame_name = prefix + ss.str() + ".jpg";
		std::cout << "******************************************************************* " << frame_name << std::endl;
		
		cv::Mat frame = cv::imread(frame_name);
		cv::Mat image = frame.clone(), blob;;
		pre_process(image, blob);

		std::vector<cv::Mat> outputs;
		process(blob, net, outputs);

		std::vector<cv::Rect> detections = post_process(image, outputs);
		//std::sort(detections.begin(), detections.end(), [](cv::Rect rect1, cv::Rect rect2) {return rect1.x < rect2.x; });

		std::vector<std::vector<float>> trackers = mot_tracker.update(detections);

		//for (auto detection : detections)
		//	cv::rectangle(image, detection, cv::Scalar(0, 0, 255), 1);

		for (auto tracker : trackers)
			cv::rectangle(image, cv::Rect(tracker[0], tracker[1], tracker[2]- tracker[0], tracker[3]- tracker[1]), colors[int(tracker[4])%32], 1);

		cv::imshow("image", image);
		cv::waitKey(1);
	}
	cv::destroyAllWindows();

	return 0;
}