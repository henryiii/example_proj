from scipy.stats import norm
import numpy as np

def gauss_bg(x, mu, sigma, mix):
    pdf1 = norm.pdf(x, mu, sigma)
    pdf2 = x*0 + 1/20
    return pdf1*(1-mix) + pdf2*mix

def nll_gauss(params, x):
    mu, sigma, mix = params
    y = gauss_bg(x, mu, sigma, mix)
    return -np.sum(np.log(y))