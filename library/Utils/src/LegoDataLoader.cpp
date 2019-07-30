//
// Created by phili on 29.07.2019.
//

#include <random>
#include "LegoDataLoader.hpp"
#include <lodepng.hpp>
#include <filesystem>
#include <fstream>


void LegoDataLoader::getData(int samples, std::vector<std::pair<std::string, int>> shuffledFiles, DataSet &data) {
	int numberValidation = std::floor(samples * 0.2);
	assert(samples + numberValidation <= shuffledFiles.size());

	std::vector<Matrix> trainingSamples, trainingLabels;
	std::vector<Matrix> validationSamples, validationLabels;

	unsigned width, height;

	for (int i = 0; i < samples; i++) {
		std::vector<unsigned char> image;
		unsigned error = lodepng::decode(image, width, height, shuffledFiles[i].first);
		if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; }
		int totalSize = width * height;

		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> tmp(1, 160000);
		int idx = 0;
		for (int k = 0; k < shuffledFiles[i].first.size() - 4; k += 4) {
			tmp(0, idx) = shuffledFiles[i].first[k];
			tmp(0, idx + totalSize) = shuffledFiles[i].first[k + 1];
			tmp(0, idx + totalSize * 2) = shuffledFiles[i].first[k + 2];
			tmp(0, idx + totalSize * 3) = shuffledFiles[i].first[k + 3];
			idx++;
		}
		Matrix sample = tmp.cast<float>();
		sample /= 255;
		Matrix C(1, 16);
		C.setZero();
		C(0, shuffledFiles[i].second) = 1;
		trainingSamples.push_back(sample);
		trainingLabels.push_back(C);


	}
	for (int i = samples; i < samples + numberValidation; i++) {
		std::vector<unsigned char> image;
		unsigned error = lodepng::decode(image, width, height, shuffledFiles[i].first);
		if (error) { std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl; }
		int totalSize = width * height;

		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> tmp(1, 160000);
		int idx = 0;
		for (int k = 0; k < shuffledFiles[i].first.size() - 4; k += 4) {
			tmp(0, idx) = shuffledFiles[i].first[k];
			tmp(0, idx + totalSize) = shuffledFiles[i].first[k + 1];
			tmp(0, idx + totalSize * 2) = shuffledFiles[i].first[k + 2];
			tmp(0, idx + totalSize * 3) = shuffledFiles[i].first[k + 3];
			idx++;
		}
		Matrix sample = tmp.cast<float>();
		sample /= 255;
		Matrix C(1, 16);
		C.setZero();
		C(0, shuffledFiles[i].second) = 1;
		validationSamples.push_back(sample);
		validationLabels.push_back(C);


	}

	data = {trainingSamples, trainingLabels, validationSamples, validationLabels};


}


std::vector<std::pair<std::string, int>> LegoDataLoader::shuffleData(std::string dataDir) {
	std::ofstream(dataDir + std::string("lego/shuffledData.csv"));
	std::vector<std::pair<std::string, int>> values;

	for (int i = 0; i < 16; i++) {
		std::string path = dataDir + std::string("lego/train/") + std::to_string(i + 1) + std::string("/");
		for (const auto &entry : std::filesystem::directory_iterator(path)) {
			values.push_back(std::pair<std::string, int>(entry.path(), i));
		}// << std::endl;

	}
	auto rng = std::default_random_engine{};

	std::shuffle(std::begin(values), std::end(values), rng);

	return values;
}