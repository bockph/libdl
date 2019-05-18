//
// Created by phili on 10.05.2019.
//

#include "Placeholder.hpp"

Placeholder::Placeholder(float t){
	setForwardData(t);

//	graph->addPlaceholder(std::make_shared<Node>(this));
}
void Placeholder::backwards(float previousGradient) {

}