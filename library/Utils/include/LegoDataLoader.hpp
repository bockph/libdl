//
// Created by phili on 29.07.2019.
//

#pragma once

#include <vector>
#include <iostream>
#include <commonDatatypes.hpp>


class LegoDataLoader {
public:
	static void getData(int samples, std::vector<std::pair<std::string, int>> shuffledFiles, DataSet &data);
	static std::vector<std::pair<std::string, int>> shuffleData(std::string dataDir);
};


