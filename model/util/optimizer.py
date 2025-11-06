import torch.optim as optim
from torch.optim import lr_scheduler


def init_optimizer_params(args):
    # args = cfg.TRAIN.optimizer
    if args.name not in ['SGD', 'RMSprop', 'Adam', 'AdamW']:
        raise NotImplementedError(f"Optimizer [{args.name}] is not implemented.")
    return {
        'name': args.name, # SGD, RMSprop, Adam, AdamW
        'lr': float(args.lr),
        'betas': args.betas,  # Adam, AdamW
        'eps': float(args.eps),  # Adam, AdamW
        'weight_decay': float(args.weight_decay),

        'momentum': float(args.momentum), # SGD, RMSprop
        'nesterov': args.nesterov,  # SGD
        'rmsprop_alpha': float(args.rmsprop_alpha),  # RMSprop
        'rmsprop_centered': args.rmsprop_centered,  # RMSprop
    }

def init_scheduler_params(args):
    # args = cfg.TRAIN.lr_scheduler
    if args.name not in ['LinearLR', 'LambdaLR', 'StepLR', 'MultiStepLR', 'CosineAnnealingLR']:
        raise NotImplementedError(f"Scheduler [{args.name}] is not implemented.")
    return {
        'name': args.name, # LinearLR, LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
        'step_size': args.step_size,  # StepLR
        'milestones': args.milestones,  # MultiStepLR
        'gamma': float(args.gamma),  # StepLR, MultiStepLR
        'eta_min': float(args.eta_min),  # CosineAnnealingLR
        'steps': [args.steps[0], args.steps[1]],  # LambdaLR
        'T_max': args.T_max,  # CosineAnnealingLR
    }

def get_optimizer_params(params, model):
    #params = init_optimizer_params(cfg.TRAIN.optimizer)
    name = params['name']
    
    if name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=float(params['lr']),
            momentum=float(params['momentum']),
            weight_decay=float(params['weight_decay']),
            nesterov=params['nesterov']
        )
    elif name == 'RMSprop':
        return optim.RMSprop(
            model.parameters(),
            lr=float(params['lr']),
            momentum=float(params['momentum']),
            weight_decay=float(params['weight_decay']),
            alpha=float(params['rmsprop_alpha']),
            centered=params['rmsprop_centered']
        )
    elif name == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=float(params['lr']),
            betas=params['betas'],
            eps=float(params['eps']),
            weight_decay=float(params['weight_decay'])
        )
    elif name == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=float(params['lr']),
            betas=params['betas'],
            eps=float(params['eps']),
            weight_decay=float(params['weight_decay'])
        )
    else:
        raise NotImplementedError(f"Optimizer [{name}] is not implemented.")


def get_scheduler_params(params, optimizer):
    #params = init_scheduler_params(cfg.TRAIN.lr_scheduler)
    name = params['name']

    if name == 'LinearLR':
        return lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)

    elif name == 'LambdaLR':
        epoch_count = 0
        n_epochs = params['steps'][0]
        n_epochs_decay = params['steps'][1]

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif name == 'StepLR':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )

    elif name == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params['milestones'],
            gamma=params['gamma']
        )

    elif name == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'],
            eta_min=float(params['eta_min'])
        )

    else:
        raise NotImplementedError(f"Scheduler [{name}] is not implemented.")


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.optimizer.name == 'SGD':
        optimizer = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            momentum=float(cfg.TRAIN.optimizer.momentum),
            weight_decay=float(cfg.TRAIN.optimizer.weight_decay),
            nesterov=cfg.TRAIN.optimizer.nesterov
        )
    elif cfg.TRAIN.optimizer.name == 'RMSprop':
        optimizer = optim.RMSprop(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            momentum=float(cfg.TRAIN.optimizer.momentum),
            weight_decay=float(cfg.TRAIN.optimizer.weight_decay),
            alpha=float(cfg.TRAIN.optimizer.rmsprop_alpha),
            centered=cfg.TRAIN.optimizer.rmsprop_centered
        )
    elif cfg.TRAIN.optimizer.name == 'Adam':
        # print(f'float(cfg.TRAIN.optimizer.lr) = {float(cfg.TRAIN.optimizer.lr)}')
        # print(f'float(cfg.TRAIN.optimizer.eps) = {float(cfg.TRAIN.optimizer.eps)}')
        optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            betas=cfg.TRAIN.optimizer.betas,  # (0.9, 0.999)
            eps=float(cfg.TRAIN.optimizer.eps),  # 1e-08
            weight_decay=float(cfg.TRAIN.optimizer.weight_decay)  # 0.0
        )
    elif cfg.TRAIN.optimizer.name == 'AdamW':
        # print(f'float(cfg.TRAIN.optimizer.lr) = {float(cfg.TRAIN.optimizer.lr)}')
        # print(f'float(cfg.TRAIN.optimizer.eps) = {float(cfg.TRAIN.optimizer.eps)}')
        optimizer = optim.AdamW(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(cfg.TRAIN.optimizer.lr),
            betas=cfg.TRAIN.optimizer.betas,  # (0.9, 0.999)
            eps=float(cfg.TRAIN.optimizer.eps),  # 1e-08
            weight_decay=float(cfg.TRAIN.optimizer.weight_decay)  # 0.01 - 0.1
        )
    return optimizer


def get_optimizer_lr(cfg, model, lr, weight_decay):
    optimizer = None
    if cfg.TRAIN.optimizer.name == 'SGD':
        optimizer = optim.SGD(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(lr),
            momentum=float(cfg.TRAIN.optimizer.momentum),
            weight_decay=float(weight_decay),
            nesterov=cfg.TRAIN.optimizer.nesterov
        )
    elif cfg.TRAIN.optimizer.name == 'RMSprop':
        optimizer = optim.RMSprop(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(lr),
            momentum=float(cfg.TRAIN.optimizer.momentum),
            weight_decay=float(weight_decay),
            alpha=float(cfg.TRAIN.optimizer.rmsprop_alpha),
            centered=cfg.TRAIN.optimizer.rmsprop_centered
        )
    elif cfg.TRAIN.optimizer.name == 'Adam':
        # print(f'float(cfg.TRAIN.optimizer.lr) = {float(cfg.TRAIN.optimizer.lr)}')
        # print(f'float(cfg.TRAIN.optimizer.eps) = {float(cfg.TRAIN.optimizer.eps)}')
        optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(lr),
            betas=cfg.TRAIN.optimizer.betas,  # (0.9, 0.999)
            eps=float(cfg.TRAIN.optimizer.eps),  # 1e-08
            weight_decay=float(weight_decay)  # 0.0
        )
    elif cfg.TRAIN.optimizer.name == 'AdamW':
        # print(f'float(cfg.TRAIN.optimizer.lr) = {float(cfg.TRAIN.optimizer.lr)}')
        # print(f'float(cfg.TRAIN.optimizer.eps) = {float(cfg.TRAIN.optimizer.eps)}')
        optimizer = optim.AdamW(
            # filter(lambda p: p.requires_grad, model.parameters()),
            model.parameters(),
            lr=float(lr),
            betas=cfg.TRAIN.optimizer.betas,  # (0.9, 0.999)
            eps=float(cfg.TRAIN.optimizer.eps),  # 1e-08
            weight_decay=float(weight_decay)  # 0.01 - 0.1
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.lr_scheduler.name == 'LinearLR':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    if cfg.TRAIN.lr_scheduler.name == 'LambdaLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
        epoch_count = 0  # the starting epoch count, eg: 0
        n_epochs = cfg.TRAIN.lr_scheduler.steps[0]  # eg: 99
        n_epochs_decay = cfg.TRAIN.lr_scheduler.steps[1]  # eg: 100

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.TRAIN.lr_scheduler.name == 'StepLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.TRAIN.lr_scheduler.step_size,
                                              gamma=cfg.TRAIN.lr_scheduler.gamma)
    elif cfg.TRAIN.lr_scheduler.name == 'MultiStepLR':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.TRAIN.lr_scheduler.milestones,
                                                   gamma=cfg.TRAIN.lr_scheduler.gamma)
    elif cfg.TRAIN.lr_scheduler.name == 'CosineAnnealingLR':
        # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6
        # https://www.tutorialexample.com/understand-torch-optim-lr_scheduler-cosineannealinglr-with-examples-pytorch-tutorial/
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=cfg.TRAIN.end_epoch,  # Maximum number of iterations.
                                                         eta_min=float(cfg.TRAIN.lr_scheduler.eta_min)
                                                         # Minimum learning rate.
                                                         )
    else:
        return NotImplementedError('Learning rate policy [%s] is not implemented', cfg.TRAIN.lr_scheduler.name)

    return scheduler


def update_learning_rate(cfg, optimizer):
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]['lr']
    scheduler = get_scheduler(cfg, optimizer)
    scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))


def update_learning_rate_linear(cfg, optimizer, epoch):
    epoch_count = 0  # the starting epoch count, eg: 0
    n_epochs = cfg.TRAIN.lr_scheduler.steps[0]  # eg: 99
    n_epochs_decay = cfg.TRAIN.lr_scheduler.steps[1]  # eg: 100
    base_lr = cfg.TRAIN.optimizer.learning_rate

    rule = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay)
    lr = base_lr * rule
    optimizer.param_groups[0]['lr'] = lr
