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
const int distance_threshold = 5;


int main(int argc, char** argv)
{
	Sort mot_tracker = Sort(1, 3, 0.5);
	cv::dnn::Net net = cv::dnn::readNet("./yolov5s.onnx");
	std::vector<cv::Vec3b> colors(32);
	srand(time(0));
	for (size_t i = 0; i < colors.size(); i++)
		colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);

	std::vector<std::vector<int>> ids_history;
	typedef std::vector<std::vector<float>>  v2f;
	std::vector<v2f> trackers_history;

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

		std::vector<std::vector<float>> trackers = mot_tracker.update(detections);

		std::vector<int> ids;
		for (auto tracker : trackers)
		{
			//std::cout << tracker[4] << " ";
			ids.push_back(tracker[4]);
		}
		//std::cout << std::endl;

		ids_history.push_back(ids);
		trackers_history.push_back(trackers);
		if (ids_history.size() <= 1)
			continue;
		if (ids_history.size() > 3)
		{
			ids_history.erase(ids_history.begin());
			trackers_history.erase(trackers_history.begin());
		}

		size_t n = ids_history.size();
		for (size_t i = 0; i < ids_history[n - 1].size(); i++)
		{
			if (std::find(ids_history[n - 2].begin(), ids_history[n - 2].end(), ids_history[n - 1][i]) != ids_history[n - 2].end())
			{
				//std::cout << ids_history[n - 1][i] << " ";
				//std::cout << std::find(ids_history[n - 2].begin(), ids_history[n - 2].end(), ids_history[n - 1][i]) - ids_history[n - 2].begin() << " ";
				size_t id = std::find(ids_history[n - 2].begin(), ids_history[n - 2].end(), ids_history[n - 1][i]) - ids_history[n - 2].begin();
				auto obj_cur = trackers_history[n - 1][i];
				auto obj_old = trackers_history[n - 2][id];
				float distance = get_center_distance(obj_cur, obj_old);
				if (distance > distance_threshold)
				{
					auto tracker = trackers[i];
					cv::rectangle(image, cv::Rect(tracker[0], tracker[1], tracker[2] - tracker[0], tracker[3] - tracker[1]), colors[int(tracker[4]) % 32], 1);
				}
			}
		}
		//std::cout << std::endl;
		
		cv::imshow("image", image);
		cv::waitKey(1);
	}
	cv::destroyAllWindows();

	return 0;
}