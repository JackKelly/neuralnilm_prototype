#!/usr/bin/python
from __future__ import print_function, division
import unittest
from timeit import default_timer as timer
import numpy as np
import theano
import theano.tensor as T
from neuralnilm import objectives
from neuralnilm.utils import gen_pulse

SEQ_LENGTH = 512
N_SEQ_PER_BATCH = 8
N_OUTPUTS = 5
TARGET_SHAPE = (N_SEQ_PER_BATCH, SEQ_LENGTH, N_OUTPUTS)
DURATIONS = (0, 10, 100, 300, 502)
DTYPE = np.float32

def gen_target():
    t = np.zeros(shape=TARGET_SHAPE, dtype=DTYPE)
    for seq_i in range(N_SEQ_PER_BATCH):
        for output_i in range(N_OUTPUTS):
            pulse = gen_pulse(amplitude=1, duration=DURATIONS[output_i],
                              start_index=10, seq_length=SEQ_LENGTH,
                              dtype=DTYPE)
            t[seq_i, :, output_i] = pulse
    return t

class TestObjectives(unittest.TestCase):

    def test_scaled_cost3(self):
        t = theano.shared(gen_target())
        y = theano.shared(np.zeros(shape=TARGET_SHAPE, dtype=DTYPE))
        self.assertEqual(t.dtype, y.dtype)
        start_time = timer()
        cost = objectives.scaled_cost3(y, t)
        end_time = timer()
        print("Time: {:.3f}s".format(end_time - start_time))
        print(cost.eval())
        cost = objectives.scaled_cost3(y, t, ignore_inactive=False)
        print(cost.eval())

    
if __name__ == '__main__':
    unittest.main()
