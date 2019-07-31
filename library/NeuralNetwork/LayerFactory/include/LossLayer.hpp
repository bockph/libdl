//
// Created by pbo on 23.07.19.
//


#pragma once

#include <LossFunction.hpp>
#include <AbstractLayer.hpp>
/*!
 * Creates an Layer for the loss calculation
 */
class LossLayer: public AbstractLayer {
public:
    /*!
     * Creates the selected loss function and sets it as outputNode
     * @param input
     * @param computeGraph
     * @param losstype
     */
    LossLayer(std::shared_ptr<AbstractLayer> input,std::shared_ptr<Graph> computeGraph, LossType losstype);
    ~LossLayer()=default; //!default destructor


    /*!
     *
     * During Training this method should be used to set the labels for the loss calculation
     *
     * @param the Labels that should be used for training
     */
    void updateLabels(Matrix &newLabels);

    /*!
     *  throws a runtime_error if no computation has yet been done
     * @return the current Prediciton
     */
    const Matrix getPrediction() const;
    /*!
     *  throws a runtime_error if no training or prediction has yet been done
     * @return the current Loss
     */
    float getLoss();

private:
    std::shared_ptr<LossFunction> _lossNode;
};



