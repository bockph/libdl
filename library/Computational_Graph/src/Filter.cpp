//
// Created by phili on 14.06.2019.
//

#include "Filter.hpp"


Filter::Filter(Eigen::MatrixXf m,int dim, int channel) {
	setForward(m);
    setOutputDim(dim);
    setOutputChannels(channel);

    /*
     * A Filter should be a row vector (for the beginning ) of the Form RGB, RGB,RGB,... so 3 data points per pixel
     * If several filters are used, a matrix -existing of several row vectors- is used
     */

}

void Filter::backwards() {

	setForward(getForward() - 0.01 * getCurrentGradients());
}