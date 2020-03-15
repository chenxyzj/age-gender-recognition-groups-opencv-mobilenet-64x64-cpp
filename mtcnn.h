#pragma once

#include <opencv2/opencv.hpp>

#define NUM_REGRESSIONS 4
#define NUM_PTS 5

struct BBox {
	float x1;
	float y1;
	float x2;
	float y2;

	cv::Rect getRect() const { return cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)); }

	BBox getSquare() const {
		BBox bbox;
		float bboxWidth = x2 - x1;
		float bboxHeight = y2 - y1;
		float side = std::max(bboxWidth, bboxHeight);
		bbox.x1 = static_cast<float>(x1 + (bboxWidth - side) * 0.5f);
		bbox.y1 = static_cast<float>(y1 + (bboxHeight - side) * 0.5f);
		bbox.x2 = static_cast<float>(bbox.x1 + side);
		bbox.y2 = static_cast<float>(bbox.y1 + side);
		return bbox;
	}
};

struct Face {
	BBox bbox;
	float score;
	float regression[NUM_REGRESSIONS];
	float ptsCoords[2 * NUM_PTS];

	static void applyRegression(std::vector<Face> &faces, bool addOne = false) {
		for (size_t i = 0; i < faces.size(); ++i) {
			float bboxWidth =
				faces[i].bbox.x2 - faces[i].bbox.x1 + static_cast<float>(addOne);
			float bboxHeight =
				faces[i].bbox.y2 - faces[i].bbox.y1 + static_cast<float>(addOne);
			faces[i].bbox.x1 = faces[i].bbox.x1 + faces[i].regression[1] * bboxWidth;
			faces[i].bbox.y1 = faces[i].bbox.y1 + faces[i].regression[0] * bboxHeight;
			faces[i].bbox.x2 = faces[i].bbox.x2 + faces[i].regression[3] * bboxWidth;
			faces[i].bbox.y2 = faces[i].bbox.y2 + faces[i].regression[2] * bboxHeight;
		}
	}

	static void bboxes2Squares(std::vector<Face> &faces) {
		for (size_t i = 0; i < faces.size(); ++i) {
			faces[i].bbox = faces[i].bbox.getSquare();
		}
	}

	static std::vector<Face> runNMS(std::vector<Face> &faces, float threshold,
		bool useMin = false) {
		std::vector<Face> facesNMS;
		if (faces.empty()) {
			return facesNMS;
		}

		std::sort(faces.begin(), faces.end(), [](const Face &f1, const Face &f2) {
			return f1.score > f2.score;
		});

		std::vector<int> indices(faces.size());
		for (size_t i = 0; i < indices.size(); ++i) {
			indices[i] = static_cast<int>(i);
		}

		while (indices.size() > 0) {
			int idx = indices[0];
			facesNMS.push_back(faces[idx]);
			std::vector<int> tmpIndices = indices;
			indices.clear();
			for (size_t i = 1; i < tmpIndices.size(); ++i) {
				int tmpIdx = tmpIndices[i];
				float interX1 = std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1);
				float interY1 = std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1);
				float interX2 = std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2);
				float interY2 = std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2);

				float bboxWidth = std::max(0.f, (interX2 - interX1 + 1));
				float bboxHeight = std::max(0.f, (interY2 - interY1 + 1));

				float interArea = bboxWidth * bboxHeight;
				// TODO: compute outside the loop
				float area1 = (faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) *
					(faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
				float area2 = (faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) *
					(faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
				float o = 0.f;
				if (useMin) {
					o = interArea / std::min(area1, area2);
				}
				else {
					o = interArea / (area1 + area2 - interArea);
				}
				if (o <= threshold) {
					indices.push_back(tmpIdx);
				}
			}
		}
		return facesNMS;
	}
};

class OutputNetwork {
public:
	struct Config {
	public:
		std::string protoText;
		std::string caffeModel;
		float threshold;
	};

private:
	cv::dnn::Net _net;
	float _threshold;

public:
	OutputNetwork(const OutputNetwork::Config &config);
	OutputNetwork();

private:
	OutputNetwork(const OutputNetwork &rhs) = delete;
	OutputNetwork &operator=(const OutputNetwork &rhs) = delete;

public:
	std::vector<Face> run(const cv::Mat &img, const std::vector<Face> &faces);
};

class ProposalNetwork {
public:
	struct Config {
	public:
		std::string protoText;
		std::string caffeModel;
		float threshold;
		float nmsThreshold;
	};

private:
	cv::dnn::Net _net;
	float _threshold;

private:
	std::vector<Face> buildFaces(const cv::Mat &scores,
		const cv::Mat &regressions,
		const float scaleFactor, const float threshold);

public:
	ProposalNetwork(const ProposalNetwork::Config &config);
	~ProposalNetwork();

private:
	ProposalNetwork(const ProposalNetwork &rhs) = delete;
	ProposalNetwork &operator=(const ProposalNetwork &rhs) = delete;

public:
	std::vector<Face> run(const cv::Mat &img, const float minFaceSize,
		const float scaleFactor);
};

class RefineNetwork {
public:
	struct Config {
	public:
		std::string protoText;
		std::string caffeModel;
		float threshold;
	};

private:
	cv::dnn::Net _net;
	float _threshold;

public:
	RefineNetwork(const RefineNetwork::Config &config);
	~RefineNetwork();

private:
	RefineNetwork(const RefineNetwork &rhs) = delete;
	RefineNetwork &operator=(const RefineNetwork &rhs) = delete;

public:
	std::vector<Face> run(const cv::Mat &img, const std::vector<Face> &faces);
};

class MTCNNDetector {
private:
	std::unique_ptr<ProposalNetwork> _pnet;
	std::unique_ptr<RefineNetwork> _rnet;
	std::unique_ptr<OutputNetwork> _onet;

public:
	MTCNNDetector(const ProposalNetwork::Config &pConfig,
		const RefineNetwork::Config &rConfig,
		const OutputNetwork::Config &oConfig);
	std::vector<Face> detect(const cv::Mat &img, const float minFaceSize,
		const float scaleFactor);
};
