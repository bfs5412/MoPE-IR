
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res



def compute_psnr_ssim(recoverd, clean):

    assert recoverd.shape == clean.shape

    if hasattr(recoverd, 'detach'):
        recoverd = recoverd.detach().cpu().numpy()
    if hasattr(clean, 'detach'):
        clean = clean.detach().cpu().numpy()

    recoverd = np.clip(recoverd, 0, 1)
    clean = np.clip(clean, 0, 1)

    psnr_total = 0.0
    ssim_total = 0.0
    valid_count = 0

    if recoverd.ndim == 4:  
        recoverd = recoverd.transpose(0, 2, 3, 1)  
        clean = clean.transpose(0, 2, 3, 1)

        for i in range(recoverd.shape[0]):
            mse = np.mean((clean[i] - recoverd[i]) ** 2)
            if mse == 0:
                continue
            psnr_val = peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)

            win_size = min(clean[i].shape[0], clean[i].shape[1], 7)
            if win_size % 2 == 0:
                win_size -= 1
            ssim_val = structural_similarity(
                clean[i], recoverd[i], data_range=1, channel_axis=-1, win_size=win_size
            )


            psnr_total += psnr_val
            ssim_total += ssim_val
            valid_count += 1

    elif recoverd.ndim == 3:  
        recoverd = recoverd.transpose(1, 2, 0)  
        clean = clean.transpose(1, 2, 0)

        mse = np.mean((clean - recoverd) ** 2)
        if mse == 0:
            valid_count = 0
            return 0, 0, 0
        psnr_val = peak_signal_noise_ratio(clean, recoverd, data_range=1)
        win_size = min(clean.shape[0], clean.shape[1], 7)
        if win_size % 2 == 0:
            win_size -= 1
        ssim_val = structural_similarity(
            clean, recoverd, data_range=1, channel_axis=-1, win_size=win_size
        )

        psnr_total = psnr_val
        ssim_total = ssim_val
        valid_count = 1

    else:
        raise ValueError(f"Unsupported input ndim={recoverd.ndim}")

    if valid_count == 0:
        return 0, 0, 0

    return psnr_total / valid_count, ssim_total / valid_count, valid_count
def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0