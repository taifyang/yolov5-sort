/********************************************************************************
** @auth£º taify
** @date£º 2023/04/21
** @Ver :  V1.0.0
** @desc£º yolov5Ô´ÎÄ¼þ
*********************************************************************************/


#include "yolov5.h"
#include "utils.h"


YOLOv5::YOLOv5(std::string model_path)
{
	m_net = cv::dnn::readNet(model_path);
}


std::vector<cv::Rect> YOLOv5::infer(cv::Mat image)
{
	m_image = image.clone();
	pre_process();
	process();
	post_process();
	return m_boxes;
}


void YOLOv5::pre_process()
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(m_image, letterbox, cv::Size(input_width, input_height));
	cv::dnn::blobFromImage(letterbox, m_inputs, 1. / 255., cv::Size(input_width, input_height), cv::Scalar(), true, false);
}


void YOLOv5::process()
{
	m_net.setInput(m_inputs);
	m_net.forward(m_outputs, m_net.getUnconnectedOutLayersNames());
}


void YOLOv5::post_process()
{
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = (float*)m_outputs[0].data + i * output_numprob;
		float objness = ptr[4];
		if (objness < confidence_threshold)
			continue;

		float* classes_scores = ptr + 5;
		int class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
		float max_class_score = classes_scores[class_id];
		float score = max_class_score * objness;
		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_box(box, m_image.size(), cv::Size(input_width, input_height));
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
	m_boxes.clear();
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		//if(idx == 0)
		m_boxes.push_back(boxes[idx]);
	}
}