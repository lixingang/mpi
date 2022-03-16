class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    def avg(self):
        return self.sum/self.count

class AccMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.y = []
        self.y_hat = []

    def update(self, y, y_hat):
        self.y.append(y)
        self.y_hat.append(y_hat)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count