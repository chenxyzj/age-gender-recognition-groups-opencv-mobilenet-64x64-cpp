// Copyright © 2020 by Spectrico

#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "mtcnn.h"

cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor) {
	int bottom_right_x = top_left_x + width;
	int bottom_right_y = top_left_y + height;

	cv::Mat output;
	if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
		// border padding will be required
		int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

		if (top_left_x < 0) {
			width = width + top_left_x;
			border_left = -1 * top_left_x;
			top_left_x = 0;
		}
		if (top_left_y < 0) {
			height = height + top_left_y;
			border_top = -1 * top_left_y;
			top_left_y = 0;
		}
		if (bottom_right_x > input.cols) {
			width = width - (bottom_right_x - input.cols);
			border_right = bottom_right_x - input.cols;
		}
		if (bottom_right_y > input.rows) {
			height = height - (bottom_right_y - input.rows);
			border_bottom = bottom_right_y - input.rows;
		}

		cv::Rect R(top_left_x, top_left_y, width, height);
		copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, paddingColor);
	}
	else {
		// no border padding required
		cv::Rect R(top_left_x, top_left_y, width, height);
		output = input(R);
	}
	return output;
}

// Convert rectangle to square
cv::Rect rect2square(const cv::Rect& rect) {
	cv::Rect square;
	int h = rect.height;
	int w = rect.width;
	int side = h>w ? h : w;
	square.x = rect.x + static_cast<int>((w - side)*0.5);
	square.y = rect.y + static_cast<int>((h - side)*0.5);
	square.width = side;
	square.height = side;
	return square;
}

// Find best class for the blob (i. e. class with maximal probability)
void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb)
{
	// Reshape the blob
	cv::Mat probMat = probBlob.reshape(1, 1);
	cv::Point classNumber;

	cv::minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

int main(int argc, char** argv)
{
	// Read the input image file
	std::string imageFile = argc == 2 ? argv[1] : "family.jpg";
	cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);
	if (img.empty() || !img.data)
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	// Create a new image for displaying the output
	cv::Mat out = img.clone();

	// Init MTCNN Face Detector
	ProposalNetwork::Config pConfig;
	pConfig.caffeModel = "./models/det1.caffemodel";
	pConfig.protoText = "./models/det1.prototxt";
	pConfig.threshold = 0.6f;

	RefineNetwork::Config rConfig;
	rConfig.caffeModel = "./models/det2.caffemodel";
	rConfig.protoText = "./models/det2.prototxt";
	rConfig.threshold = 0.7f;

	OutputNetwork::Config oConfig;
	oConfig.caffeModel = "./models/det3.caffemodel";
	oConfig.protoText = "./models/det3.prototxt";
	oConfig.threshold = 0.7f;

	MTCNNDetector detector(pConfig, rConfig, oConfig);

	const float minFaceSize = 40.f;
	const float scaleFactor = 0.709f;

	// Init age & gender classifier
	const std::string inBlobName = "input_1";
	const std::string outBlobName = "softmax/Softmax";

	cv::dnn::Net net;
	const std::string modelFile = "model-weights-spectrico-age-gender-recognition-groups-mobilenet-64x64-9CFFCA00.pb";
	// Initialize the network
	net = cv::dnn::readNetFromTensorflow(modelFile);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the mode file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	// Fetect the faces
	std::vector<Face> faces;
	faces = detector.detect(img, minFaceSize, scaleFactor);

	for (const auto &face : faces)
	{
		// Get the face rectangle
		cv::Rect r = face.bbox.getRect();
		
		// Convert detected face rectangle to square
		cv::Rect square = rect2square(r);

		// Crop the face 
		cv::Mat img_face = getPaddedROI(img, square.x, square.y, square.width, square.height, cv::BORDER_REPLICATE);

		// Resize the face to 64x64
		cv::resize(img_face, img_face, cv::Size(64, 64), 0, 0);

		// Convert cv::Mat to image batch
		cv::Mat inputBlob = cv::dnn::blobFromImage(img_face, 0.0078431372549019607843137254902, cv::Size(64, 64), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);

		// Set the network input
		net.setInput(inputBlob, inBlobName);

		// Make forward pass and compute output
		cv::Mat result = net.forward(outBlobName);
																				  
		int classId;
		double classProb;
		// find the best class
		getMaxClass(result, &classId, &classProb);

		// Get gender
		std::string gender = (classId <= 25) ? "female" : "male";

		// Get age
		int age = (classId % 26)*3;

		// Draw rectangle on the output image
		cv::rectangle(out, r, cv::Scalar(0, 0, 255), 1, 1, 0);

		// Print age and gender on the output image
		std::string label = cv::format("%d-%d,%s", age, age+2, gender.c_str());
		cv::putText(out, label, cv::Point(r.x, r.y + 20), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));
	}

	// Write the output image
	cv::imwrite("out.png", out);

	return 0;
}
