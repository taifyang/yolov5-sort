/********************************************************************************
** @auth�� taify
** @date�� 2023/04/21
** @Ver :  V1.0.0
** @desc�� yolov5ͷ�ļ�
*********************************************************************************/


#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>


const std::vector<std::string> class_names = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush" };			//�������

const int input_width = 640;

const int input_height = 640;

const float score_threshold = 0.2;

const float nms_threshold = 0.5;

const float confidence_threshold = 0.2;

const int input_numel = 1 * 3 * input_width * input_height;

const int num_classes = class_names.size();

const int output_numprob = 5 + num_classes;

const int output_numbox = 3 * (input_width / 8 * input_height / 8 + input_width / 16 * input_height / 16 + input_width / 32 * input_height / 32);

const int output_numel = 1 * output_numprob * output_numbox;


class YOLOv5
{
public:
	/**
	 * @brief YOLOv5		���캯��
	 * @param model_path	ģ��·��
	 */
	YOLOv5(const std::string model_path);

	/**
	 * @brief infer			����ӿ�
	 * @param model_path	ͼƬ·��
	 */
	std::vector<cv::Rect> infer(const cv::Mat image);

private:
	/**
	 * @brief pre_process	Ԥ����
	 */
	void pre_process();

	/**
	 * @brief process		��������
	 */
	void process();

	/**
	 * @brief pre_process	����
	 */
	void post_process();

	/**
	 * @brief m_net			cv::dnn::Net����
	 */
	cv::dnn::Net m_net;

	/**
	 * @brief m_inputs		����ͼ��
	 */
	cv::Mat m_image;

	/**
	 * @brief m_inputs		����tensor
	 */
	cv::Mat m_inputs;

	/**
	 * @brief m_outputs		���tensor
	 */
	std::vector<cv::Mat> m_outputs;

	/**
	 * @brief m_boxes		�����box
	 */
	std::vector<cv::Rect> m_boxes;
};
