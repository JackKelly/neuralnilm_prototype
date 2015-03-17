from theano.ifelse import ifelse
import theano.tensor as T


THRESHOLD = 0
def scaled_cost(x, t, loss_func=lambda x, t: (x - t) ** 2):
    error = loss_func(x, t)
    def mask_and_mean_error(mask):
        masked_error = error[mask.nonzero()]
        mean = masked_error.mean()
        mean = ifelse(T.isnan(mean), 0.0, mean)
        return mean
    above_thresh_mean = mask_and_mean_error(t > THRESHOLD)
    below_thresh_mean = mask_and_mean_error(t <= THRESHOLD)
    return (above_thresh_mean + below_thresh_mean) / 2.0
