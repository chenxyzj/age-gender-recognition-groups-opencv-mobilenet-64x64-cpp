// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include "mtcnn.h"

// OpenCV 4.0 update
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#define CV_BGRA2RGB cv::COLOR_BGR2RGB

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;


MTCNNDetector::MTCNNDetector(const ProposalNetwork::Config &pConfig,
	const RefineNetwork::Config &rConfig,
	const OutputNetwork::Config &oConfig) {
	_pnet = std::unique_ptr<ProposalNetwork>(new ProposalNetwork(pConfig));
	_rnet = std::unique_ptr<RefineNetwork>(new RefineNetwork(rConfig));
	_onet = std::unique_ptr<OutputNetwork>(new OutputNetwork(oConfig));
}

std::vector<Face> MTCNNDetector::detect(const cv::Mat &img,
	const float minFaceSize,
	const float scaleFactor) {

	cv::Mat rgbImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, rgbImg, CV_BGR2RGB);
	}
	else if (img.channels() == 4) {
		cv::cvtColor(img, rgbImg, CV_BGRA2RGB);
	}
	if (rgbImg.empty()) {
		return std::vector<Face>();
	}
	rgbImg.convertTo(rgbImg, CV_32FC3);
	rgbImg = rgbImg.t();

	// Run Proposal Network to find the initial set of faces
	std::vector<Face> faces = _pnet->run(rgbImg, minFaceSize, scaleFactor);

	// Early exit if we do not have any faces
	if (faces.empty()) {
		return faces;
	}

	// Run Refine network on the output of the Proposal network
	faces = _rnet->run(rgbImg, faces);

	// Early exit if we do not have any faces
	if (faces.empty()) {
		return faces;
	}

	// Run Output network on the output of the Refine network
	faces = _onet->run(rgbImg, faces);

	for (size_t i = 0; i < faces.size(); ++i) {
		std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
		std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
		for (int p = 0; p < NUM_PTS; ++p) {
			std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
		}
	}

	return faces;
}

inline cv::Mat cropImage(const cv::Mat &img, cv::Rect r) {
	cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
	int dx = std::abs(std::min(0, r.x));
	if (dx > 0) {
		r.x = 0;
	}
	r.width -= dx;
	int dy = std::abs(std::min(0, r.y));
	if (dy > 0) {
		r.y = 0;
	}
	r.height -= dy;
	int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
	r.width -= dw;
	int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
	r.height -= dh;
	if (r.width > 0 && r.height > 0) {
		img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
	}
	return m;
}

OutputNetwork::OutputNetwork(const OutputNetwork::Config &config) {
	_net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
	if (_net.empty()) {
		throw std::invalid_argument("invalid protoText or caffeModel");
	}
	_threshold = config.threshold;
}

OutputNetwork::OutputNetwork() {}

std::vector<Face> OutputNetwork::run(const cv::Mat &img,
	const std::vector<Face> &faces) {
	cv::Size windowSize = cv::Size(48, 48);

	std::vector<Face> totalFaces;

	for (auto &f : faces) {
		cv::Mat roi = cropImage(img, f.bbox.getRect());
		cv::resize(roi, roi, windowSize, 0, 0, cv::INTER_AREA);

		// we will run the ONet on each face
		// TODO : see how this can be optimized such that we run
		// it only 1 time

		// build blob images from the inputs
		auto blobInput =
			cv::dnn::blobFromImage(roi, IMG_INV_STDDEV, cv::Size(),
				cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

		_net.setInput(blobInput, "data");

		const std::vector<cv::String> outBlobNames{ "conv6-2", "conv6-3", "prob1" };
		std::vector<cv::Mat> outputBlobs;

		_net.forward(outputBlobs, outBlobNames);

		cv::Mat regressionsBlob = outputBlobs[0];
		cv::Mat landMarkBlob = outputBlobs[1];
		cv::Mat scoresBlob = outputBlobs[2];

		const float *scores_data = (float *)scoresBlob.data;
		const float *landmark_data = (float *)landMarkBlob.data;
		const float *reg_data = (float *)regressionsBlob.data;

		if (scores_data[1] >= _threshold) {
			Face info = f;
			info.score = scores_data[1];
			for (int i = 0; i < 4; ++i) {
				info.regression[i] = reg_data[i];
			}

			float w = info.bbox.x2 - info.bbox.x1 + 1.f;
			float h = info.bbox.y2 - info.bbox.y1 + 1.f;

			for (int p = 0; p < NUM_PTS; ++p) {
				info.ptsCoords[2 * p] =
					info.bbox.x1 + landmark_data[NUM_PTS + p] * w - 1;
				info.ptsCoords[2 * p + 1] = info.bbox.y1 + landmark_data[p] * h - 1;
			}

			totalFaces.push_back(info);
		}
	}

	Face::applyRegression(totalFaces, true);
	totalFaces = Face::runNMS(totalFaces, 0.7f, true);

	return totalFaces;
}

const float P_NET_WINDOW_SIZE = 12.f;
const int P_NET_STRIDE = 2;

ProposalNetwork::ProposalNetwork(const ProposalNetwork::Config &config) {
	_net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
	if (_net.empty()) {
		throw std::invalid_argument("invalid protoText or caffeModel");
	}
	_threshold = config.threshold;
}

ProposalNetwork::~ProposalNetwork() {}

std::vector<Face> ProposalNetwork::buildFaces(const cv::Mat &scores,
	const cv::Mat &regressions,
	const float scaleFactor,
	const float threshold) {

	auto w = scores.size[3];
	auto h = scores.size[2];
	auto size = w * h;

	const float *scores_data = (float *)(scores.data);
	scores_data += size;

	const float *reg_data = (float *)(regressions.data);

	std::vector<Face> boxes;

	for (int i = 0; i < size; i++) {
		if (scores_data[i] >= (threshold)) {
			int y = i / w;
			int x = i - w * y;

			Face faceInfo;
			BBox &faceBox = faceInfo.bbox;

			faceBox.x1 = (float)(x * P_NET_STRIDE) / scaleFactor;
			faceBox.y1 = (float)(y * P_NET_STRIDE) / scaleFactor;
			faceBox.x2 =
				(float)(x * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
			faceBox.y2 =
				(float)(y * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
			faceInfo.regression[0] = reg_data[i];
			faceInfo.regression[1] = reg_data[i + size];
			faceInfo.regression[2] = reg_data[i + 2 * size];
			faceInfo.regression[3] = reg_data[i + 3 * size];
			faceInfo.score = scores_data[i];
			boxes.push_back(faceInfo);
		}
	}

	return boxes;
}

std::vector<Face> ProposalNetwork::run(const cv::Mat &img,
	const float minFaceSize,
	const float scaleFactor) {

	std::vector<Face> finalFaces;
	float maxFaceSize = static_cast<float>(std::min(img.rows, img.cols));
	float faceSize = minFaceSize;

	while (faceSize <= maxFaceSize) {
		float currentScale = (P_NET_WINDOW_SIZE) / faceSize;
		int imgHeight = static_cast<int>(std::ceil(img.rows * currentScale));
		int imgWidth = static_cast<int>(std::ceil(img.cols * currentScale));
		cv::Mat resizedImg;
		cv::resize(img, resizedImg, cv::Size(imgWidth, imgHeight), 0, 0,
			cv::INTER_AREA);

		// feed it to the proposal network
		cv::Mat inputBlob =
			cv::dnn::blobFromImage(resizedImg, IMG_INV_STDDEV, cv::Size(),
				cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

		_net.setInput(inputBlob, "data");

		const std::vector<cv::String> outBlobNames{ "conv4-2", "prob1" };
		std::vector<cv::Mat> outputBlobs;

		_net.forward(outputBlobs, outBlobNames);

		cv::Mat regressionsBlob = outputBlobs[0];
		cv::Mat scoresBlob = outputBlobs[1];

		auto faces =
			buildFaces(scoresBlob, regressionsBlob, currentScale, _threshold);

		if (!faces.empty()) {
			faces = Face::runNMS(faces, 0.5f);
		}

		if (!faces.empty()) {
			finalFaces.insert(finalFaces.end(), faces.begin(), faces.end());
		}

		faceSize /= scaleFactor;
	}

	if (!finalFaces.empty()) {
		finalFaces = Face::runNMS(finalFaces, 0.7f);
		if (!finalFaces.empty()) {
			Face::applyRegression(finalFaces, false);
			Face::bboxes2Squares(finalFaces);
		}
	}

	return finalFaces;
}


RefineNetwork::RefineNetwork(const RefineNetwork::Config &config) {
	_net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
	if (_net.empty()) {
		throw std::invalid_argument("invalid protoText or caffeModel");
	}
	_threshold = config.threshold;
}

RefineNetwork::~RefineNetwork() {}

std::vector<Face> RefineNetwork::run(const cv::Mat &img,
	const std::vector<Face> &faces) {
	cv::Size windowSize = cv::Size(24, 24);

	std::vector<cv::Mat> inputs;
	for (auto &f : faces) {
		cv::Mat roi = cropImage(img, f.bbox.getRect());
		cv::resize(roi, roi, windowSize, 0, 0, cv::INTER_AREA);
		inputs.push_back(roi);
	}

	// build blob images from the inputs
	auto blobInputs =
		cv::dnn::blobFromImages(inputs, IMG_INV_STDDEV, cv::Size(),
			cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

	_net.setInput(blobInputs, "data");

	const std::vector<cv::String> outBlobNames{ "conv5-2", "prob1" };
	std::vector<cv::Mat> outputBlobs;

	_net.forward(outputBlobs, outBlobNames);

	cv::Mat regressionsBlob = outputBlobs[0];
	cv::Mat scoresBlob = outputBlobs[1];

	std::vector<Face> totalFaces;

	const float *scores_data = (float *)scoresBlob.data;
	const float *reg_data = (float *)regressionsBlob.data;

	for (int k = 0; k < faces.size(); ++k) {
		if (scores_data[2 * k + 1] >= _threshold) {
			Face info = faces[k];
			info.score = scores_data[2 * k + 1];
			for (int i = 0; i < 4; ++i) {
				info.regression[i] = reg_data[4 * k + i];
			}
			totalFaces.push_back(info);
		}
	}

	// nms and regression
	totalFaces = Face::runNMS(totalFaces, 0.7f);
	Face::applyRegression(totalFaces, true);
	Face::bboxes2Squares(totalFaces);

	return totalFaces;
}
