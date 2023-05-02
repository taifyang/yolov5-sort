/********************************************************************************
** @auth£º taify
** @date£º 2023/04/21
** @Ver :  V1.0.0
** @desc£º yolov5Ô´ÎÄ¼þ
*********************************************************************************/


#include "yolov5.h"
#include "utils.h"


void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
	if (!scaleUp)
	{
		r = std::min(r, 1.0f);
	}

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		outImage = image.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


void pre_process(cv::Mat& image, cv::Mat& blob)
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(image, letterbox, params, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	cv::dnn::blobFromImage(letterbox, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
}


void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs)
{
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
}


void scale_boxes(cv::Rect& box, cv::Size size)
{
	float gain = std::min(INPUT_WIDTH * 1.0 / size.width, INPUT_HEIGHT * 1.0 / size.height);
	int pad_w = (INPUT_WIDTH - size.width * gain) / 2;
	int pad_h = (INPUT_HEIGHT - size.height * gain) / 2;
	box.x -= pad_w;
	box.y -= pad_h;
	box.x /= gain;
	box.y /= gain;
	box.width /= gain;
	box.height /= gain;
}


std::vector<cv::Rect> post_process(cv::Mat& image, std::vector<cv::Mat>& preprcess_image)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Rect> boxes_nms;

	float* data = (float*)preprcess_image[0].data;

	const int dimensions = 85;
	const int rows = 25200;
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float* classes_scores = data + 5;
			cv::Mat scores(1, class_name.size(), CV_32F, classes_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD)
			{
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = round(x - 0.5 * w);
				int top = round(y - 0.5 * h);
				int width = round(w);
				int height = round(h);
				cv::Rect box = cv::Rect(left, top, width, height);
				scale_boxes(box, image.size());
				boxes.push_back(box);
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
			}
		}
		data += 85;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (size_t i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		//if(idx == 0)
			boxes_nms.push_back(boxes[idx]);
	}

	return boxes_nms;
}