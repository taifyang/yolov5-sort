/********************************************************************************
** @auth： taify
** @date： 2023/04/21
** @Ver :  V1.0.0
** @desc： yolov5头文件
*********************************************************************************/


#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>


const float INPUT_WIDTH = 640.0;			//输入网络图像宽度
const float INPUT_HEIGHT = 640.0;			//输入网络图像高度
const float SCORE_THRESHOLD = 0.5;			//得分阈值
const float NMS_THRESHOLD = 0.5;			//nms阈值
const float CONFIDENCE_THRESHOLD = 0.5;		//置信度阈值

const std::vector<std::string> class_name = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush" };			//类别名称


/**
 * @brief pre_process	预处理
 * @param input_image	输入图像
 * @param net			输入网络
 */
void pre_process(cv::Mat& image, cv::Mat& blob);


/**
 * @brief process	网络处理
 * @param blob		输入图像
 * @param net		输入网络
 * @param outputs	输入网络
 */
void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs);


/**
 * @brief scale_boxes	缩放检测框
 * @param box			检测框
 * @param size			缩放尺寸
 */
void scale_boxes(cv::Rect& box, cv::Size size);


/**
 * @brief post_process		后处理
 * @param origin_image		原始图像
 * @param processed_image	处理后的图像
 */
std::vector<cv::Rect> post_process(cv::Mat& origin_image, std::vector<cv::Mat>& processed_image);