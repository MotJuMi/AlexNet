import os
import torch
import torch.nn.functional as F


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def local_response_normalization(x, n, alpha=1e-4, beta=0.75, k=1):
	"""
	:param x:
	:param size:
	:param alpha:
	:param beta:
	:param k:
	"""
	dim = x.dim()
	if dim < 3:
		raise ValueError('Expected 3D or higher dimensionality \
						  input (got %d dimensions)' % (dim))
	denom = x.mul(x).unsqueeze(1)
	if dim == 3:
		denom = F.pad(denom, (0, 0, n // 2, (n - 1) // 2))
		denom = F.avg_pool2d(denom, (n, 1), stride=1).squeeze()
	else:
		sizes = x.size()
		denom = denom.view(sizes[0], 1, sizes[1], sizes[2], -1)
		denom = F.pad(denom, (0, 0, 0, 0, size // 2, (size - 1) // 2))
		denom = F.avg_pool3d(denom, (size, 1, 1), stride=1).squeeze(1)
		denom = denom.view(sizes)
	denom = denom.mul(alpha).add(k).pow(beta)
	return x / denom
