/********************************************************************************
** @auth： taify
** @date： 2023/04/21
** @Ver :  V1.0.0
** @desc： sort源文件
*********************************************************************************/


#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "kalmanboxtracker.h"


/**
 * @brief Sort	Sort算法
 */
class Sort
{
public:
	/**
	 * @brief Sort 			构造函数
	 * @param max_age		未命中时间上限
	 * @param min_hits		未命中时间下限
	 * @param iou_threshold iou阈值
	 */
	Sort(int max_age, int min_hits, float iou_threshold);

	/**
	 * @brief update	更新检测框
	 * @param dets		检测框
	 */
	std::vector<std::vector<float>> update(std::vector<cv::Rect> dets);

private:
	/**
	 * @param m_max_age		未命中时间上限
	 */
	int m_max_age;

	/**
	 * @param m_min_hits	未命中时间下限
	 */
	int m_min_hits;

	/**
	 * @param m_iou_threshold	iou阈值
	 */
	float m_iou_threshold;

	/**
	 * @param m_trackers	卡尔曼跟踪器的集合
	 */
	std::vector<KalmanBoxTracker> m_trackers;

	/**
	 * @param m_frame_count	帧数
	 */
	int m_frame_count;
};