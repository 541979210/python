import pickle
import gzip
import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T

class LR(object):
    def __init__(self, input, n_in, n_out):
        """chushihua"""
        self.W=theano.shared(
            value=numpy.zeros(
                (n_in,n_out),
                dtype='float32'
            ),
            name='W',
            borrow=True
        )
        self.b=theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype='float32'
            ),
            name='b',
            borrow=True

        )
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        self.params=[self.W,self.b]
        self.input=input

    def
