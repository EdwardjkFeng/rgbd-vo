import torch


def check_cuda(items):
    if torch.cuda.is_available():
        to_cuda=[]
        for x in items:
            if torch.is_tensor(x):
                to_cuda.append(x.cuda())
            elif type(x).__module__ == 'numpy':
                to_cuda.append(torch.from_numpy(x).cuda())
            else:
                to_cuda.append(x)
        return to_cuda
    else:
        return items
    