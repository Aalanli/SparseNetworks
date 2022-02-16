# %%
import math

import torch
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.models import resnet34

from denseConv import DenseConvNet, dense_conv
from convNeXtRef import convnext_tiny
from coat import coat_tiny
from data import build_loader
from trainer import TrainerWandb, make_equal_3D


def calculate_param_size(model):
    params = 0
    for i in model.parameters():
        params += math.prod(list(i.shape))
    return params

params = dict(
    send_dim=64,
    d_model=256,
    patch_size=9,
    proc_kernel_size=3,
    comp_kernel_size=1,
    n_layers=9,
    splits=2,
    n_classes=102
)

data_parameters = dict(
    batch_size=4,
    train_percent=0.8
)

training_parameters = dict(
    lr=1e-3,
    weight_decay=None,
    lr_drop=100
)



comp = torch.nn.Conv2d(params['send_dim'], params['d_model'], params['comp_kernel_size'], padding='same')
col = dense_conv(params['d_model'], params['send_dim'])
model = DenseConvNet(params['splits'], params['send_dim'], params['n_layers'], params['patch_size'], comp, col, params['n_classes']).cuda()

#model = convnext_tiny(num_classes=102).cuda()

#model = resnet34(num_classes=2).cuda()
#model = coat_tiny(num_classes=102).cuda()
print("parameters: ", calculate_param_size(model))


train_data, eval_data = build_loader(data_parameters['batch_size'], 2, workers=4, train_split_percent=data_parameters['train_percent'])
optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters['lr'])
criterion = torch.nn.CrossEntropyLoss()
 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, training_parameters['lr_drop'])

name = 'denseNetV2-splits2'

config = {}
config.update(training_parameters)
config.update(data_parameters)
config.update(params)
config['name'] = name
config['parameters'] = calculate_param_size(model)
trainer = TrainerWandb(model, criterion, optimizer, f'experiments/image_classification/cifar10/{name}', 300, 700, False, lr_scheduler, 10, 0, config)

lr_scheduler.get_lr()

# %%

#trainer.train(train_data)
trainer.train_epochs(70, train_data, eval_data, project='ContinousSum', entity='allanl')
print(lr_scheduler.get_last_lr())
print(lr_scheduler.get_lr())

