/********************************************************************************
** @auth： taify
** @date： 2023/04/21
** @Ver :  V1.0.0
** @desc： kalmanboxtracker头文件
*********************************************************************************/


#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>


/**
 * @brief KalmanBoxTracker	卡尔曼跟踪器
 */
class KalmanBoxTracker
{
public:
	/**
	 * @brief KalmanBoxTracker	卡尔曼跟踪器类构造函数
	 * @param bbox				检测框
	 */
	KalmanBoxTracker(std::vector<int> bbox);
	
	/**
	 * @brief update	通过观测的检测框更新系统状态
	 * @param bbox		检测框
	 */
	void update(std::vector<int> bbox);	
	
	/**
	 * @brief predict	估计预测的边界框
	 */
	std::vector<float> predict();
	
	/**
	 * @brief get_state	返回当前检测框状态
	 */
	std::vector<float> get_state();
	
public:
	/**
	 * @param m_kf	卡尔曼滤波器
	 */
	cv::KalmanFilter* m_kf;

	/**
	 * @param m_time_since_update	从更新开始的时间（帧数）
	 */
	int m_time_since_update;

	/**
	 * @param count	卡尔曼跟踪器的个数
	 */
	static int count;

	/**
	 * @param m_id	卡尔曼跟踪器的id
	 */
	int m_id;

	/**
	 * @param m_history	历史检测框的储存
	 */
	std::vector<std::vector<float>> m_history;

	/**
	 * @param m_hits
	 */
	int m_hits;

	/**
	 * @param m_hit_streak	忽略目标初始的若干帧
	 */
	int m_hit_streak;

	/**
	 * @param m_age	predict被调用的次数计数
	 */
	int m_age;
};