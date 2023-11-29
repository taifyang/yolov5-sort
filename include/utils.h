/********************************************************************************
** @auth�� taify
** @date�� 2023/04/21
** @Ver :  V1.0.0
** @desc�� utilsԴ�ļ�
*********************************************************************************/


#pragma once


#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>


/**
 * @brief LetterBox		LetterBox����
 * @param input_image	����ͼ��
 * @param output_image	���ͼ��
 * @param shape			����ߴ�
 * @param color			�����ɫ
 */
void LetterBox(cv::Mat& input_image, cv::Mat& output_image, cv::Size& shape = cv::Size(640, 640), cv::Scalar& color = cv::Scalar(114, 114, 114));


/**
 * @brief scale_boxes	���Ű�Χ��
 * @param box			��Χ��
 * @param input_size	����ߴ�
 * @param output_size	����ߴ�
 */
void scale_box(cv::Rect& box, cv::Size input_size, cv::Size output_size);


/**
 * @brief draw_label ������
 * @param input_image ����ͼ��
 * @param label ��ǩ����
 * @param left ��ǩ��ͼ��������
 * @param top ��ǩ��ͼ���ϲ����
 */
void draw_label(cv::Mat& input_image, std::string label, int left, int top);


/**
 * @brief get_center �����������
 * @param detections �������
 */
std::vector<cv::Point> get_center(std::vector<cv::Rect> detections);


/**
 * @brief get_center ������������
 * @param p1 ��1
 * @param p2 ��2
 */
float get_distance(cv::Point p1, cv::Point p2);


/**
 * @brief get_center ��������������
 * @param p1 ����1
 * @param p2 ����2
 */
float get_center_distance(std::vector<float> bbox1, std::vector<float> bbox2);


/**
 * @brief convert_bbox_to_z ��������[x1,y1,x2,y2]����ʽת��Ϊ[x,y,s,r]����ʽ
 * @param bbox				����
 */
std::vector<float> convert_bbox_to_z(std::vector<int> bbox);


/**
 * @brief convert_x_to_bbox ��������[x,y,s,r]����ʽת��Ϊ[x1,y1,x2,y2]����ʽ
 * @param x					����
 */
std::vector<float> convert_x_to_bbox(std::vector<float> x);


/**
 * @brief iou	�����������iou
 * @param box1	����1
 * @param box2	����2
 */
float iou(std::vector<float> box1, std::vector<float> box2);


/**
 * @brief associate_detections_to_tracks	��������͸��ٽ������
 * @param detections						�����
 * @param trackers							���ٽ��
 * @param iou_threshold						iou����
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
associate_detections_to_tracks(std::vector<cv::Rect> detections, std::vector<std::vector<int>> trackers, float iou_threshold = 0.3);