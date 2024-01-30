import torch
import time

from models.algorithms import LM_H


def lev_mar_H(JtWJ):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    B, _, _ = JtWJ.shape
    diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ, (2, 1))
    epsilon = (traceJtJ * 1e-6).view(B, 1, 1) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian

B = 1
L = 6

JtWJ = torch.rand((B, L, L))

time_H1 = 0
time_H2 = 0
for i in range(20):
    start1 = time.time()
    H1 = LM_H(JtWJ)
    time_H1 += time.time() - start1

    start2 = time.time()
    H2 = lev_mar_H(JtWJ)
    time_H2 += time.time() - start2

    
diag_mask = torch.eye(6).view(1, L, L).type_as(JtWJ)
diagJtJ = diag_mask * JtWJ
diag = torch.diag_embed(torch.diagonal(JtWJ, dim1=-2, dim2=-1))

print(torch.equal(diagJtJ, diag))
print(torch.sum(diagJtJ - diag))

traceJtJ = (torch.sum(diagJtJ, (2, 1)) * 1e-6).view(B, 1, 1) * diag_mask
trace = torch.diag_embed(torch.sum(diag, (2, 1))[:, None].expand(B, L) * 1e-6)

print(torch.equal(traceJtJ, trace))
print(torch.sum(traceJtJ - trace))

print(torch.equal(H1, H2))
print(torch.sum(H2 - H1))
print(time_H1/20 - time_H2/20)
