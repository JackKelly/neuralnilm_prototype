"""
Functions to create initializers for parameter variables
"""

import numpy as np

from lasagne.utils import floatX
from lasagne.init import Initializer


class Identity(Initializer):
    """Initialise with the identity matrix.  Can be used to initialise ReLU RNNs
    as per [#le2015ReLU-RNNs]_.


    :references:
        .. [#le2015ReLE-RNNs] Le, Jaitly, Hinton, "A Simple Way to
            Initialize Recurrent Networks of Rectified Linear Units" (2015)
    """

    def __init__(self, scale=1):
        """
        :parameters:
            - scale : int or float
                To quote from [#le2015ReLU-RNNs]_:
                "for tasks that exhibit less long range dependencies, scaling the
                identity matrix by a small scalar is an effective mechanism to
                forget long range effects. This is the same behavior as LTSMs
                when their forget gates are set so that the memory decays
                fast."
        """
        self.scale = scale

    def sample(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise RuntimeError(
                "Identity initialisation only works with"
                " square weight matrices.")
        return floatX(np.identity(n=shape[0]) * self.scale)
