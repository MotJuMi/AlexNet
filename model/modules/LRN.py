import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.util import local_response_normalization

class LocalResponseNorm(nn.Module):
	"""
	:param n: 	  number of adjucent kernels used for normalization
	:param alpha: coefficient
	:beta: 		  power
	:k: 		  additive factor
	"""
	def __init__(self, n, alpha=1e-4, beta=0.75, k=1):
		super(LocalResponseNorm, self).__init__()
		self.n = n
		self.alpha = alpha
		self.beta = beta
		self.k = k

	def forward(self, input):
		return local_response_normalization(input,
											self.n,
											self.alpha,
											self.beta,
											self.k)