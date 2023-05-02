/********************************************************************************
** @auth： taify
** @date： 2023/04/21
** @Ver :  V1.0.0
** @desc： utils源文件
*********************************************************************************/


#pragma once


#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>


/**
 * @brief draw_label 画检测框
 * @param input_image 输入图像
 * @param label 标签名称
 * @param left 标签距图像左侧距离
 * @param top 标签距图像上侧距离
 */
void draw_label(cv::Mat& input_image, std::string label, int left, int top);


/**
 * @brief get_center 计算检测框中心
 * @param detections 输入检测框
 */
std::vector<cv::Point> get_center(std::vector<cv::Rect> detections);


/**
 * @brief get_center 计算两点间距离
 * @param p1 点1
 * @param p2 点2
 */
float get_distance(cv::Point p1, cv::Point p2);


/**
 * @brief get_center 计算两检测框中心
 * @param p1 检测框1
 * @param p2 检测框2
 */
float get_center_distance(std::vector<float> bbox1, std::vector<float> bbox2);


/**
 * @brief convert_bbox_to_z 将检测框由[x1,y1,x2,y2]的形式转化为[x,y,s,r]的形式
 * @param bbox				检测框
 */
std::vector<float> convert_bbox_to_z(std::vector<int> bbox);


/**
 * @brief convert_x_to_bbox 将检测框由[x,y,s,r]的形式转化为[x1,y1,x2,y2]的形式
 * @param x					检测框
 */
std::vector<float> convert_x_to_bbox(std::vector<float> x);


/**
 * @brief iou	计算两检测框的iou
 * @param box1	检测框1
 * @param box2	检测框2
 */
float iou(std::vector<float> box1, std::vector<float> box2);


/**
 * @brief associate_detections_to_tracks	将检测结果和跟踪结果关联
 * @param detections						检测结果
 * @param trackers							跟踪结果
 * @param iou_threshold						iou矩阵
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
associate_detections_to_tracks(std::vector<cv::Rect> detections, std::vector<std::vector<int>> trackers, float iou_threshold = 0.3);