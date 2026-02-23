from scipy.stats import norm

#define normal cdf:
def normal_cdf(x):
    return norm.cdf(x)

#define normal pdf:
def normal_pdf(x):
    return norm.pdf(x)

#convert float in array
def as_float_array(x):
    return np.asarray(x, dtype=float)
#avoiding array 0D
def to_scalar_if_0d(x):
    x = np.asarray(x)
    return x.item() if x.shape == () else x