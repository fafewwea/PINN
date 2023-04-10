import argparse
import os

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import warnings
import sys
import torch.cuda
from data_preprocessing import data_preprocess

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time

from calculating_loss import train_loss, validation_loss
from utils import net_o
from adaptive_sampling import space_adaptive_sampling, sampling_plot

from layers import DNN
from collections import OrderedDict

print('Start PINN in 230105')
print(time.localtime())

def parse_args():
    parser = argparse.ArgumentParser(description='PINN Multi-GPU Training')
    parser.add_argument('--start_epoch', default=1, type=int, help='starting epoch')
    parser.add_argument('--epochs', default=100000, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--patience', default=200, type=int, help='for Early Stopping')

    # gpu
    parser.add_argument('--gpu', type=int, help='GPU id to use')
    parser.add_argument('--distributed', type=int, default=True, help='DDP training --gpu is ignored if this is true')
    parser.add_argument('--num_gpu', type=int, default=4, help='Number of GPU to use in distributed mode; ignored otherwise')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')

    # layer
    parser.add_argument('--layers', type=list, default=[6, 100,  100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 7])
    parser.add_argument('--activation', type=str, default='torch.nn.Tanh')

    # restore
    parser.add_argument('--restore', type=int, default=0, help='load checkpoints')
    parser.add_argument('--restore_file', type=str, default='checkpoint_model.pt')

    # data
    parser.add_argument('--empirical', type=int, default=200000, help='empirical data')
    parser.add_argument('--physics', type=int, default=4000, help='physics data')

    # loss balancing
    parser.add_argument('--uvw', type=float, default=1, help='uvw')
    parser.add_argument('--P', type=float, default=1, help='P')
    parser.add_argument('--T', type=float, default=1, help='T')
    parser.add_argument('--I', type=float, default=1, help='I')
    parser.add_argument('--M', type=float, default=1, help='M')

    parser.add_argument('--w_phy', type=float, default=1.0, help='weight of physical loss')

    args = parser.parse_args()

    return args

def main_worker(gpu, world_size, args):
    warnings.filterwarnings('ignore')
    best_val_loss = 100000.0
    best_epoch = 1

    # Device Setting
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    if not args.distributed or (args.distributed and args.gpu == 0):
        print('\n\nStart Training at:', 230104)

    if args.distributed:
        if torch.cuda.current_device() == 0:
            print('Calling Initializing Process Group')

            os.mkdir('/home/yubinryu/ml/2023/PINN/result_{}'.format(args.w_phy))
            os.mkdir('/home/yubinryu/ml/2023/PINN/result_{}/checkpoint'.format(args.w_phy))

        path = '/home/yubinryu/ml/2023/PINN/result_{}'.format(args.w_phy)

        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=world_size,
                                rank=args.gpu)

    # Data Preprocessing
    X_train, X_val, Y_train, Y_val = data_preprocess(args.empirical, args.physics)

    if not args.distributed or (args.distributed and args.gpu == 0):
        print('Trainset Size : {}, Validation Size : {}'.format(X_train.shape[0], X_val.shape[0]))

    train_data = torch.tensor(X_train, dtype=torch.float64)
    target_data = torch.tensor(Y_train, dtype=torch.float64)

    train_lb = torch.min(train_data, dim=0)[0]
    train_ub = torch.max(train_data, dim=0)[0]

    target_lb = torch.min(target_data, dim=0)[0]
    target_ub = torch.max(target_data, dim=0)[0]

    X_val = torch.tensor(X_val, dtype=torch.float64)
    Y_val = torch.tensor(Y_val, dtype=torch.float64)

    # Target data scaling
    target_data = 2.0 * (target_data - target_lb) / (target_ub - target_lb) - 1.0
    Y_val = 2.0 * (Y_val - target_lb) / (target_ub - target_lb) - 1.0

    trainset = TensorDataset(train_data, target_data)

    # Creating Dataloader
    if args.distributed:
        train_sampler = DistributedSampler(trainset)

    else:
        train_sampler = None

    # nw = 4 * args.num_gpu
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler)

    model = DNN(args.layers)

    if not args.distributed or (args.distributed and args.gpu == 0):
        print(model)

    if args.gpu is not None:
        model.cuda(args.gpu)

    if args.distributed:
        ddp_model = DDP(model, device_ids = [args.gpu])
        model = ddp_model

    # Setting Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # Setting Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=False, min_lr=0.000001)

    loss_history = []
    val_loss_history = []
    uvw_loss_history = []
    P_loss_history = []
    T_loss_history = []
    I_loss_history = []
    M_loss_history = []

    cont_loss_history = []
    moment_loss_history = []
    species_loss_history = []

    # load checkpoint
    if args.restore == 1:

        if torch.cuda.current_device() == 0:
            print('Loading Checkpoint')

        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(path + '/checkpoint/' + args.restore_file, map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        triggertimes = checkpoint['trigger_times']
        args.start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        best_epoch = checkpoint['epoch']

        # loss history
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        uvw_loss_history = checkpoint['uvw_loss_history']
        P_loss_history = checkpoint['P_loss_history']
        T_loss_history = checkpoint['T_loss_history']
        I_loss_history = checkpoint['I_loss_history']
        M_loss_history = checkpoint['M_loss_history']

        cont_loss_history = checkpoint['cont_loss_history']
        moment_loss_history = checkpoint['moment_loss_history']
        species_loss_history = checkpoint['species_loss_history']

        if torch.cuda.current_device() == 0:
            print('Successfully loaded checkpoints, epoch {}, best_val_loss {}, trigger_times {}'.format(args.start_epoch-1, best_val_loss, triggertimes))
            print(type(loss_history))
            print(len(loss_history))
            print(loss_history)

    # Start Training
    if not args.distributed or (args.distributed and args.gpu == 0):
        sys.stdout.flush()

    ti = time.time()

    if torch.cuda.current_device() == 0:
        print('TRAINING START')

    patience = args.patience
    triggertimes = 0
    last_epoch = args.epochs

    for epoch in range(args.start_epoch, last_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = train(optimizer, model, train_loader, args.gpu, args.uvw, args.P, args.T, args.I, args.M, args.w_phy, train_lb, train_ub, target_lb, target_ub)

        if epoch % 1 == 0:

            val_loss = validation_loss(X_val, Y_val, args.gpu, model, args.uvw, args.P, args.T, args.I, args.M, args.w_phy, train_lb, train_ub, target_lb, target_ub)
            scheduler.step(val_loss)

            if torch.cuda.current_device() == 0:
                print('Iteration: %d, Loss: %.3e, Validation Loss: %.3e, Learning Rate: %.3e, Best Val Epoch: %d, Time: %.4f' % (
                    epoch, loss.item(), val_loss.item(), optimizer.param_groups[0]['lr'], best_epoch, time.time() - ti))
                print('uvw: %.3e, P: %.3e, T: %.3e, I: %.3e, M: %.3e, cont: %.3e, momentum: %.3e, species: %.3e' % (
                    loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item(), loss7.item(),
                    loss8.item()))

            if torch.cuda.current_device() == 0:
                loss_check = loss.item()
                if np.isnan(loss_check) == True:
                    print('uvw: %.3e, P: %.3e, T: %.3e, I: %.3e, M: %.3e, cont: %.3e, momentum: %.3e, species: %.3e' % (
                        loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item(), loss7.item(), loss8.item()))

            loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())

            uvw_loss_history.append(loss1.item())
            P_loss_history.append(loss2.item())
            T_loss_history.append(loss3.item())
            I_loss_history.append(loss4.item())
            M_loss_history.append(loss5.item())

            cont_loss_history.append(loss6.item())
            moment_loss_history.append(loss7.item())
            species_loss_history.append(loss8.item())

            # save best model
            if torch.cuda.current_device() == 0:
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch
                    print('Best Validation Loss : ', best_val_loss)
                    print('Saving Best Model...')

                    CHECKPOINT_PATH = path + '/' + 'best_model.pth'
                    torch.save(model.state_dict(), CHECKPOINT_PATH)

                    print('Trigger Times: 0')
                    triggertimes = 0

                    # save checkpoint
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss, 'val_loss': val_loss,
                                'best_epoch': best_epoch, 'trigger_times': triggertimes,
                                'loss_history': loss_history, 'val_loss_history': val_loss_history,
                                'uvw_loss_history': uvw_loss_history, 'P_loss_history': P_loss_history, 'T_loss_history': T_loss_history, 'I_loss_history': I_loss_history, 'M_loss_history': M_loss_history,
                                'cont_loss_history': cont_loss_history, 'moment_loss_history': moment_loss_history, 'species_loss_history': species_loss_history
                                },
                               path + '/' + 'checkpoint/checkpoint_model.pt'.format(epoch))

                else:
                    triggertimes += 1
                    print('Trigger Times:', triggertimes)

                    if triggertimes >= patience:
                        print('Early Stopping!\nStart to test process.')
                        last_epoch = epoch
                        break

        if epoch % 1000 == 1:

            if torch.cuda.current_device() == 0:
                print("space adaptive sampling...")

            total_data = pd.read_pickle('/home/yubinryu/ml/PINN_final/final_data/total.pkl')
            data_s = total_data.sample(1000000)

            train_s = data_s[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()

            points_X, points_Y = space_adaptive_sampling(data_s, args.gpu, model, train_lb, train_ub, target_lb, target_ub)

            train_data = torch.cat((train_data, points_X), dim=0)
            target_data = torch.cat((target_data, points_Y), dim=0)

            u_pred, v_pred, w_pred, p_pred, T_pred, I_pred, M_pred = predict(train_s, args.gpu, model, train_lb, train_ub, target_lb, target_ub)

            sampling_plot(train_s, u_pred, v_pred, w_pred, points_X, epoch, path)

    if torch.cuda.current_device() == 0:
        print('Total Training Time: %.4f' % (time.time() - ti))
        print('TRAINING END')

    # cp = plt.figure()
    #
    # total_data = pd.read_pickle('/home/yubinryu/ml/PINN_final/final_data/total.pkl')
    # data = total_data.sample(20000)
    #
    # train_t = data[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()
    #
    # model = DNN(layers=[6, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 7])
    #
    # loaded_state_dict = torch.load(path + '/best_model.pth')
    # new_state_dict = OrderedDict()
    #
    # for n, v in loaded_state_dict.items():
    #     name = n.replace("module.", "")
    #     new_state_dict[name] = v
    #
    # model.load_state_dict(new_state_dict)
    # model = model.to(torch.device('cpu'))
    #
    # u_pred, v_pred, w_pred, p_pred, T_pred, I_pred, M_pred = predict(train_t, torch.device('cpu'), model, train_lb, train_ub, target_lb, target_ub)
    #
    # nn = 100
    #
    # xx = np.linspace(-0.095, 0.095, nn)
    # yy = np.linspace(0, 0, nn)
    # zz = np.linspace(0.005, 0.295, nn)
    #
    # X, Y, Z = np.meshgrid(xx, yy, zz)
    #
    # index = np.reshape(np.round(train_t[:, 4:5], 5) == 360, (train_t.shape[0],)) * np.reshape(np.round(train_t[:, 3:4], 5) == 0.00012,
    #                                                                              (train_t.shape[0],))
    # idx_120_360 = np.array([i for i, x in enumerate(index) if x == True])
    #
    # u_predict = griddata(train_t[idx_120_360, 0:3], u_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    # v_predict = griddata(train_t[idx_120_360, 0:3], v_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    # w_predict = griddata(train_t[idx_120_360, 0:3], w_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    # I_predict = griddata(train_t[idx_120_360, 0:3], I_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    # M_predict = griddata(train_t[idx_120_360, 0:3], M_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    #
    # levels = np.linspace(0, 0.6465366, 1000)
    # cp1 = plt.contourf(X[0], Z[0], (u_predict[0] ** 2 + v_predict[0] ** 2 + w_predict[0] ** 2) ** 0.5, levels, cmap='jet')
    # cb1 = plt.colorbar(cp1)
    # plt.gca().invert_yaxis()
    # # plt.show()
    # plt.savefig(path + '/Velocity.png', dpi=300)
    # cb1.remove()
    # plt.gca().invert_yaxis()
    #
    # levels = np.linspace(0, 0.00032, 1000)
    # cp2 = plt.contourf(X[0], Z[0], (I_predict[0]), levels, cmap='jet')
    # cb2 = plt.colorbar(cp2)
    # plt.gca().invert_yaxis()
    # # plt.show()
    # plt.savefig(path + '/Initiator.png', dpi=300)
    # cb2.remove()
    # plt.gca().invert_yaxis()
    #
    # levels = np.linspace(0, 18.57001, 1000)
    # cp3 = plt.contourf(X[0], Z[0], (M_predict[0]), levels, cmap='jet')
    # cb3 = plt.colorbar(cp3)
    # plt.gca().invert_yaxis()
    # # plt.show()
    # plt.savefig(path + '/Monomer.png', dpi=300)
    # cb3.remove()
    # plt.gca().invert_yaxis()
    #
    # # Loss Visualization
    # x_loss = np.reshape([i for i in range(1, last_epoch+1)], newshape=[last_epoch, 1])
    # train_loss_data = np.reshape(loss_history, newshape=[last_epoch, 1])
    # validation_loss_data = np.reshape(val_loss_history, newshape=[last_epoch, 1])
    #
    # plt.figure()
    # plt.plot(x_loss, train_loss_data, '#ff7c5e', label='train')
    # plt.plot(x_loss, validation_loss_data, '#4561aa', label='validation')
    # plt.yscale('log')
    # plt.legend()
    # # plt.show()
    # plt.savefig(path + '/train_valid_loss.png', dpi=300)
    #
    # # Loss check
    # uvw_loss_data = np.reshape(uvw_loss_history, newshape=[last_epoch, 1])
    # P_loss_data = np.reshape(P_loss_history, newshape=[last_epoch, 1])
    # T_loss_data = np.reshape(T_loss_history, newshape=[last_epoch, 1])
    # I_loss_data = np.reshape(I_loss_history, newshape=[last_epoch, 1])
    # M_loss_data = np.reshape(M_loss_history, newshape=[last_epoch, 1])
    #
    # cont_loss_data = np.reshape(cont_loss_history, newshape=[last_epoch, 1])
    # moment_loss_data = np.reshape(moment_loss_history, newshape=[last_epoch, 1])
    # species_loss_data = np.reshape(species_loss_history, newshape=[last_epoch, 1])
    #
    # plt.figure()
    # plt.plot(x_loss, uvw_loss_data, label='uvw', c='#ae3c60')
    # plt.plot(x_loss, P_loss_data, label='P', c='#ffa101')
    # plt.plot(x_loss, T_loss_data, label='T', c='#ffa778')
    # plt.plot(x_loss, I_loss_data, label='I', c='#267778')
    # plt.plot(x_loss, M_loss_data, label='M', c='#82b4bb')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.savefig(path + '/data_loss.png', dpi=300)
    #
    # plt.figure(figsize=(9, 3))
    # plt.plot(x_loss, cont_loss_data, label='cont', c='#df473c')
    # plt.plot(x_loss, moment_loss_data, label='momentum', c='#f3c33c')
    # plt.plot(x_loss, species_loss_data, label='species', c='#255e79')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.savefig(path + '/physics_loss.png', dpi=300)
    #
    # # uvw loss check
    # plt.figure()
    # plt.plot(x_loss, uvw_loss_data, label='uvw', c='#ae3c60')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.savefig(path + '/uvw.png', dpi=300)
    #
    # # P loss check
    # plt.figure()
    # plt.plot(x_loss, P_loss_data, label='P', c='#ffa101')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/pressure.png', dpi=300)
    #
    # # T loss check
    # plt.figure()
    # plt.plot(x_loss, T_loss_data, label='T', c='#ffa778')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/temperature.png', dpi=300)
    #
    # # I loss check
    # plt.figure()
    # plt.plot(x_loss, I_loss_data, label='I', c='#267778')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/initiator.png', dpi=300)
    #
    # # M loss check
    # plt.figure()
    # plt.plot(x_loss, M_loss_data, label='M', c='#82b4bb')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/monomer.png', dpi=300)
    #
    # # cont loss check
    # plt.figure()
    # plt.plot(x_loss, cont_loss_data, label='cont', c='#df473c')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/cont.png', dpi=300)
    #
    # # momentum loss check
    # plt.figure()
    # plt.plot(x_loss, moment_loss_data, label='momentum', c='#f3c33c')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/momentum.png', dpi=300)
    #
    # # species loss check
    # plt.figure()
    # plt.plot(x_loss, species_loss_data, label='species', c='#255e79')
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.savefig(path + '/species.png', dpi=300)
    #
    # if torch.cuda.current_device() == 0:
    #     print('Plotting Complete')

def train(optimizer, model, data_loader, device, w_uvw, w_P, w_T, w_I, w_M, w_phy, train_lb, train_ub, target_lb, target_ub):
    model.train()

    running_train_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0
    running_3_loss = 0.0
    running_4_loss = 0.0
    running_5_loss = 0.0
    running_6_loss = 0.0
    running_7_loss = 0.0
    running_8_loss = 0.0

    for train_batch, target_batch in data_loader:
        loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8 = train_loss(train_batch, target_batch, device, model, w_uvw, w_P, w_T, w_I, w_M, w_phy, train_lb, train_ub, target_lb, target_ub)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        running_train_loss += loss
        running_1_loss += loss1
        running_2_loss += loss2
        running_3_loss += loss3
        running_4_loss += loss4
        running_5_loss += loss5
        running_6_loss += loss6
        running_7_loss += loss7
        running_8_loss += loss8

    train_loss_value = running_train_loss / len(data_loader)
    loss1 = running_1_loss / len(data_loader)
    loss2 = running_2_loss / len(data_loader)
    loss3 = running_3_loss / len(data_loader)
    loss4 = running_4_loss / len(data_loader)
    loss5 = running_5_loss / len(data_loader)
    loss6 = running_6_loss / len(data_loader)
    loss7 = running_7_loss / len(data_loader)
    loss8 = running_8_loss / len(data_loader)

    return train_loss_value, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

def predict(train_s, device, model, train_lb, train_ub, target_lb, target_ub):

    x = torch.tensor(train_s[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(train_s[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(train_s[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(train_s[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(train_s[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(train_s[:, 5:6], requires_grad=True).float().to(device)

    model.eval()
    u, v, w, p, T, I, M, output = net_o(x, y, z, input_C, input_T, domain, model, train_lb, train_ub)

    target_lb = target_lb.detach().cpu().numpy()
    target_ub = target_ub.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    output = (target_lb * (1.0 - output) + target_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]
    M = output[:, 6:7]

    # u = u.detach().cpu().numpy()
    # v = v.detach().cpu().numpy()
    # w = w.detach().cpu().numpy()
    # p = p.detach().cpu().numpy()
    # T = T.detach().cpu().numpy()
    # I = I.detach().cpu().numpy()
    # M = M.detach().cpu().numpy()

    return u, v, w, p, T, I, M

def main():
    args = parse_args()
    print(args)

    if args.distributed:
        world_size = torch.cuda.device_count() if args.num_gpu is None else args.num_gpu
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

    else:
        main_worker(args.gpu, None, args)

if __name__ == "__main__":

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    main()