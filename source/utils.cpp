/********************************************************************************
** @auth£º taify
** @date£º 2023/04/21
** @Ver :  V1.0.0
** @desc£º utilsÍ·ÎÄ¼þ
*********************************************************************************/


#include "utils.h"
#include "hungarian.h"


void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
	int baseLine;
	cv::Size label_size = cv::getTextSize(label, 1, 1, 2, &baseLine);
	top = std::max(top, label_size.height);
	cv::Point tlc = cv::Point(left, top);
	cv::Point brc = cv::Point(left, top + label_size.height + baseLine);
	cv::putText(input_image, label, cv::Point(left, top + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 1);
}


std::vector<cv::Point> get_center(std::vector<cv::Rect> detections)
{
	std::vector<cv::Point> detections_center(detections.size());
	for (size_t i = 0; i < detections.size(); i++)
	{
		detections_center[i] = cv::Point(detections[i].x + detections[i].width / 2, detections[i].y + detections[i].height / 2);
	}

	return detections_center;
}


float get_distance(cv::Point p1, cv::Point p2)
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}


float get_center_distance(std::vector<float> bbox1, std::vector<float> bbox2)
{
	float x1 = bbox1[0], x2 = bbox2[0];
	float y1 = bbox1[1], y2 = bbox2[1];
	float w1 = bbox1[2] - bbox1[0], w2 = bbox2[2] - bbox2[0];
	float h1 = bbox1[3] - bbox1[1], h2 = bbox2[3] - bbox2[1];
	cv::Point p1(x1 + w1 / 2, y1 + h1 / 2), p2(x2 + w2 / 2, y2 + h2 / 2);
	return get_distance(p1, p2);
}


std::vector<float> convert_bbox_to_z(std::vector<int> bbox)
{
	float w = bbox[2] - bbox[0];
	float h = bbox[3] - bbox[1];
	float x = bbox[0] + w / 2;
	float y = bbox[1] + h / 2;
	float s = w * h;
	float r = w / h;

	return { x, y, s, r };
}


std::vector<float> convert_x_to_bbox(std::vector<float> x)
{
	float w = sqrt(x[2] * x[3]);
	float h = x[2] / w;
	return { x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2 };
}


float iou(std::vector<int> box1, std::vector<int> box2)
{
	int x1 = std::max(box1[0], box2[0]);
	int y1 = std::max(box1[1], box2[1]);
	int x2 = std::min(box1[2], box2[2]);
	int y2 = std::min(box1[3], box2[3]);
	int w = std::max(0, x2 - x1);
	int h = std::max(0, y2 - y1);
	int inter_area = w * h;
	float iou = inter_area * 1.0 / ((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area);

	return iou;
}


template <typename T>
std::vector<size_t> sort_indices(const std::vector<T>& v)
{
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
	return idx;
}


std::vector<std::vector<int>> linear_assignment(cv::Mat iou_matrix)
{
	std::vector<std::vector<float>> costMatrix(iou_matrix.cols, std::vector<float>(iou_matrix.rows));
	for (size_t i = 0; i < iou_matrix.cols; i++)
		for (size_t j = 0; j < iou_matrix.rows; j++)
			costMatrix[i][j] = iou_matrix.at<float>(j, i);

	HungarianAlgorithm HungAlgo;
	std::vector<int> assignment;
	HungAlgo.Solve(costMatrix, assignment);

	std::vector<std::vector<int>> tmp(2);
	for (size_t i = 0; i < assignment.size(); i++)
	{
		if (assignment[i] >= 0)
		{
			tmp[0].push_back(assignment[i]);
			tmp[1].push_back(i);
		}
	}

	std::vector<size_t> indices = sort_indices(tmp[0]);
	std::sort(tmp[0].begin(), tmp[0].end());

	std::vector<std::vector<int>> ret(2);
	ret[0] = tmp[0];
	for (size_t i = 0; i < ret[0].size(); i++)
		ret[1].push_back(tmp[1][indices[i]]);

	return ret;
}


std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
associate_detections_to_tracks(std::vector<cv::Rect> detections, std::vector<std::vector<int>> trackers, float iou_threshold)
{
	if (trackers.size() == 0)
	{
		std::vector<int> unmatched_detections(detections.size());
		std::iota(unmatched_detections.begin(), unmatched_detections.end(), 0);
		return { {}, unmatched_detections, {} };
	}

	cv::Mat iou_matrix(detections.size(), trackers.size(), CV_32F);
	for (size_t i = 0; i < iou_matrix.rows; i++)
	{
		for (size_t j = 0; j < iou_matrix.cols; j++)
		{
			std::vector<int> detection{ detections[i].x, detections[i].y, detections[i].x + detections[i].width, detections[i].y + detections[i].height };
			std::vector<int> tracker = trackers[j];
			iou_matrix.at<float>(i, j) = iou(detection, tracker);
		}
	}
	//std::cout << iou_matrix << std::endl; 

	std::vector<std::vector<int>> matched_indices(2);
	if (std::min(iou_matrix.rows, iou_matrix.cols) > 0)
	{
		cv::Mat a(iou_matrix.rows, iou_matrix.cols, CV_32F, cv::Scalar(0));
		for (size_t i = 0; i < a.rows; i++)
		{
			for (size_t j = 0; j < a.cols; j++)
			{
				if (iou_matrix.at<float>(i, j) > iou_threshold)
					a.at<float>(i, j) = 1;
			}
		}
		//std::cout << a << std::endl;

		cv::Mat a_sum0(iou_matrix.cols, 1, CV_32F, cv::Scalar(0));
		cv::reduce(a, a_sum0, 0, cv::REDUCE_SUM);
		std::vector<float> sum0(iou_matrix.cols);
		for (size_t i = 0; i < sum0.size(); i++)
			sum0[i] = a_sum0.at<float>(0, i);
		float a_sum0_max = *std::max_element(sum0.begin(), sum0.end());

		cv::Mat a_sum1(1, iou_matrix.rows, CV_32F, cv::Scalar(0));
		cv::reduce(a, a_sum1, 1, cv::REDUCE_SUM);
		std::vector<float> sum1(iou_matrix.rows);
		for (size_t i = 0; i < sum1.size(); i++)
			sum1[i] = a_sum1.at<float>(i, 0);
		float a_sum1_max = *std::max_element(sum1.begin(), sum1.end());

		if (int(a_sum0_max) == 1 && int(a_sum1_max) == 1)
		{
			std::vector<cv::Point> nonZeroCoordinates;
			cv::findNonZero(a, nonZeroCoordinates);
			std::sort(nonZeroCoordinates.begin(), nonZeroCoordinates.end(), [](cv::Point p1, cv::Point p2) {return p1.y < p2.y; });
			for (int i = 0; i < nonZeroCoordinates.size(); i++)
			{
				matched_indices[0].push_back(nonZeroCoordinates[i].y);
				matched_indices[1].push_back(nonZeroCoordinates[i].x);
			}
		}
		else
		{
			matched_indices = linear_assignment(-iou_matrix);
		}
	}

	std::vector<int> unmatched_detections;
	for (size_t i = 0; i < detections.size(); i++)
	{
		if (std::find(matched_indices[0].begin(), matched_indices[0].end(), i) == matched_indices[0].end())
			unmatched_detections.push_back(i);
	}

	std::vector<int> unmatched_trackers;
	for (size_t i = 0; i < trackers.size(); i++)
	{
		if (std::find(matched_indices[1].begin(), matched_indices[1].end(), i) == matched_indices[1].end())
			unmatched_trackers.push_back(i);
	}

	std::vector<std::pair<int, int>> matches;
	for (size_t i = 0; i < matched_indices[0].size(); i++)
	{
		//std::cout << matched_indices[0][i] << " " << matched_indices[1][i] << std::endl;
		if (iou_matrix.at<float>(matched_indices[0][i], matched_indices[1][i]) < iou_threshold)
		{
			unmatched_detections.push_back(matched_indices[0][i]);
			unmatched_trackers.push_back(matched_indices[1][i]);
		}
		else
			matches.push_back({ matched_indices[0][i], matched_indices[1][i] });
	}
	
	return { matches, unmatched_detections, unmatched_trackers };
}