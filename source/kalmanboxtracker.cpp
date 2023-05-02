/********************************************************************************
** @auth£º taify
** @date£º 2023/04/21
** @Ver :  V1.0.0
** @desc£º kalmanboxtrackerÔ´ÎÄ¼þ
*********************************************************************************/


#include "utils.h"
#include "kalmanboxtracker.h"


int KalmanBoxTracker::count = 0;


KalmanBoxTracker::KalmanBoxTracker(std::vector<int> bbox)
{
	m_kf = new cv::KalmanFilter(7, 4);

	m_kf->transitionMatrix = cv::Mat::eye(7, 7, CV_32F);
	m_kf->transitionMatrix.at<float>(0, 4) = 1;
	m_kf->transitionMatrix.at<float>(1, 5) = 1;
	m_kf->transitionMatrix.at<float>(2, 6) = 1;

	m_kf->measurementMatrix = cv::Mat::eye(4, 7, CV_32F);

	m_kf->measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);
	m_kf->measurementNoiseCov.at<float>(2, 2) *= 10;
	m_kf->measurementNoiseCov.at<float>(3, 3) *= 10;

	m_kf->errorCovPost = cv::Mat::eye(7, 7, CV_32F);
	m_kf->errorCovPost.at<float>(4, 4) *= 1000;
	m_kf->errorCovPost.at<float>(5, 5) *= 1000;
	m_kf->errorCovPost.at<float>(6, 6) *= 1000;
	m_kf->errorCovPost *= 10;

	m_kf->processNoiseCov = cv::Mat::eye(7, 7, CV_32F);
	m_kf->processNoiseCov.at<float>(4, 4) *= 0.01;
	m_kf->processNoiseCov.at<float>(5, 5) *= 0.01;
	m_kf->processNoiseCov.at<float>(6, 6) *= 0.001;

	m_kf->statePost = cv::Mat::eye(7, 1, CV_32F);
	m_kf->statePost.at<float>(0, 0) = convert_bbox_to_z(bbox)[0];
	m_kf->statePost.at<float>(1, 0) = convert_bbox_to_z(bbox)[1];
	m_kf->statePost.at<float>(2, 0) = convert_bbox_to_z(bbox)[2];
	m_kf->statePost.at<float>(3, 0) = convert_bbox_to_z(bbox)[3];

	m_time_since_update = 0;

	m_id = count;
	count++;

	m_history = {};

	m_hits = 0;

	m_hit_streak = 0;

	m_age = 0;

	//std::cout << m_kf->transitionMatrix << std::endl;
	//std::cout << m_kf->measurementMatrix << std::endl;
	//std::cout << m_kf->measurementNoiseCov << std::endl;
	//std::cout << m_kf->errorCovPost << std::endl;
	//std::cout << m_kf->processNoiseCov << std::endl;
	//std::cout << m_kf->statePost << std::endl;
}


void KalmanBoxTracker::update(std::vector<int> bbox)
{
	m_time_since_update = 0;
	m_history = {};
	m_hits++;
	m_hit_streak++;
	cv::Mat measurement(4, 1, CV_32F);
	for (size_t i = 0; i < 4; i++)
		measurement.at<float>(i) = convert_bbox_to_z(bbox)[i];
	//std::cout << measurement << std::endl;
	m_kf->correct(measurement);
}


std::vector<float> KalmanBoxTracker::predict()
{
	//std::cout << m_kf->statePost << std::endl;
	if (m_kf->statePost.at<float>(2, 0) + m_kf->statePost.at<float>(6, 0) <= 0)
		m_kf->statePost.at<float>(6, 0) = 0;
	m_kf->predict();
	m_age++;
	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update++;
	m_history.push_back(convert_x_to_bbox(m_kf->statePost));
	return m_history.back();
}


std::vector<float> KalmanBoxTracker::get_state()
{
	//std::cout << m_kf->statePost << std::endl;
	return convert_x_to_bbox(m_kf->statePost);
}