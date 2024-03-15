import torch.optim as optim


class CosineAnnealingWarmRestartsWithWarmup:
    def __init__(self, optimizer, args):
        self.index = 0
        self.learning_rate = args["learning_rate"]
        self.optimizer = optimizer
        self.warmup = args["warmup"]
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                        T_0=args["T_0"],
                                                                        T_mult=1,
                                                                        eta_min=0.000001)

    def override_learning_rate(self, i, optimizer):
        for g in optimizer.param_groups:
            g['lr'] = self.learning_rate * pow(i / self.warmup, 4)

    def step(self):
        if self.index < self.warmup:
            self.override_learning_rate(self.index, self.optimizer)
        else:
            self.scheduler.step()
        self.index += 1


class LambdaLRConfigured:
    def __init__(self, optimizer, args):
        self.index = args["index"]
        self.scale = args["scale"]
        self.warmup = args["warmup"]
        self.factor = self.scale[0]
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.burn_in_schedule)

    def burn_in_schedule(self, i):

        factor = self.factor
        for s, f in zip(self.index, self.scale):
            if i < s:
                factor = f
                break
        if i < self.warmup:
            factor = pow(i / self.warmup, 4)
        return factor

    def step(self):
        self.scheduler.step()


def get_lr_scheduler(optimizer, args):

    if args["strategy"] == "LambdaLR":
        scheduler = LambdaLRConfigured(optimizer, args)
    elif args["strategy"] == "CosineAnnealingWarmRestartsWithWarmup":
        scheduler = CosineAnnealingWarmRestartsWithWarmup(optimizer, args)
    else:
        assert False, "Error: Unsupported scheduler specified in yolo config!"

    return scheduler
