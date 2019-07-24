//
// Created by pbo on 24.07.19.
//
#pragma once

enum Optimizer {
    Adam
};

struct hyperParameters {
    hyperParameters(int batchsize=8, float learningRate = 0.01, Optimizer optimizer = Optimizer::Adam, float

    beta1 = 0.9, float beta2 = 0.999) :
            _batchsize(batchsize),
            _learningRate(learningRate),
            _optimizer(optimizer),
            _beta1(beta1),
            _beta2(beta2) {}

    float _learningRate;
    float _beta1;
    float _beta2;
    int _batchsize;
    Optimizer _optimizer;
};
