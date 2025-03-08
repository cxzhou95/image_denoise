import os
import yaml
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model import DnCNN
from dataset import DenoisingDataset
from torch.utils.data import DataLoader

# Apply He initialization
def weights_init(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


def cal_psnr(x, y):
    '''
    Parameters
    ----------
    x, y are two tensors has the same shape (1, C, H, W)

    Returns
    -------
    score : PSNR.
    '''

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse)
    return score


def adjust_learning_rate(optimizer, epoch, total_epochs, initial_lr, final_lr):
    lr = initial_lr * (final_lr / initial_lr) ** (epoch / (total_epochs - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, dataloader, criteria, device, optimizer, cur_epoch, total_epochs, initial_lr, final_lr):
    loss_epoch = 0.
    for noisy_patch, clean_patch, _ in dataloader:
        optimizer.zero_grad()
        noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
        pred = model(noisy_patch)
        loss = criteria(pred, clean_patch)
        # Backpropagation
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # print(loss.item())

    loss_epoch /= len(dataloader)
    lr_epoch = adjust_learning_rate(optimizer, cur_epoch, total_epochs, initial_lr, final_lr)
    return loss_epoch, lr_epoch


def test(model, dataloader, criteria, device):
    loss_epoch = 0.
    psnr_epoch = 0.
    pred_list = []
    gt_list = []
    name_list = []
    with torch.no_grad():
        for noisy_patch, clean_patch, imgname in dataloader:
            noisy_patch, clean_patch = noisy_patch.to(device), clean_patch.to(device)
            pred = model(noisy_patch)
            loss = criteria(pred, clean_patch)
            loss_epoch += loss.item()
            psnr_epoch += cal_psnr(pred, clean_patch).item()
            pred_list.append(pred)
            gt_list.append(clean_patch)
            name_list += imgname

    loss_epoch /= len(dataloader)
    psnr_epoch /= len(dataloader)
    return loss_epoch, psnr_epoch, pred_list, gt_list, name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/first-try', help='the dir where the model and ')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--patch-size', type=int, default=40, help='')
    parser.add_argument('--noise-mean', type=float, default=0, help='')
    parser.add_argument('--noise-std', type=float, default=25, help='')
    parser.add_argument('--num-patches', type=int, default=512, help='')
    parser.add_argument('--trainset_path', type=str, default='data/BSDS500', help='')
    parser.add_argument('--testset_path', type=str, default='data/BSD68', help='')
    parser.add_argument('--total-epochs', type=int, default=50, help='')
    parser.add_argument('--initial-lr', type=float, default=1e-03, help='')
    parser.add_argument('--final-lr', type=float, default=1e-04, help='')

    opt = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)

    # create folders to save results
    if os.path.exists(opt.save_dir):
        print(f"Warning: {opt.save_dir} exists, please delete it manually if it is useless.")

    os.makedirs(opt.save_dir, exist_ok=True)

    # save hyp-parameter
    with open(os.path.join(opt.save_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(opt, f, sort_keys=False)

    # folder to save the predicted HR image in the validation
    valid_folder = os.path.join(opt.save_dir, 'valid_res')
    os.makedirs(valid_folder, exist_ok=True)

    # create model
    model = DnCNN(channels=1, num_layers=17)
    model.apply(weights_init)
    model.to(device)

    # dataloader
    train_dataset = DenoisingDataset(image_dir=opt.trainset_path, phase='train', patch_size=opt.patch_size,
                                     mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_dataset = DenoisingDataset(image_dir=opt.testset_path, phase='test', patch_size=opt.patch_size,
                                    mean=opt.noise_mean, sigma=opt.noise_std, num_patches=opt.num_patches)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    # loss
    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.initial_lr, momentum=0.9, weight_decay=0.0001)

    for idx in range(0, opt.total_epochs - 1):
        t0 = time.time()
        train_loss_epoch, lr_epoch = train(model=model, dataloader=train_dataloader, criteria=criteria, device=device,
                                           optimizer=optimizer, cur_epoch=idx, total_epochs=opt.total_epochs,
                                           initial_lr=opt.initial_lr, final_lr=opt.final_lr)
        t1 = time.time()
        valid_loss_epoch, psnr_epoch, pred_HR, gt_list, img_names = test(model=model, dataloader=test_dataloader, criteria=criteria, device=device)
        t2 = time.time()
        print(
            f"Epoch: {idx} | lr: {lr_epoch:.5f} | training loss: {train_loss_epoch:.5f} | validation loss: {valid_loss_epoch:.5f} | PSNR: {psnr_epoch:.3f} | Time: {t2 - t0:.1f}")
