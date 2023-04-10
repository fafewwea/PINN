import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from utils import net_o

def space_adaptive_sampling(data, device, network, train_lb, train_ub, target_lb, target_ub):

    data_resample = data.sample(20000)
    ###############################################################
    # resample data
    train_resample = data_resample[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()
    target_resample = data_resample[['x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']].to_numpy()

    # Target Variable Scaling
    target_resample = 2.0 * (target_resample - target_lb.numpy()) / (target_ub.numpy() - target_lb.numpy()) - 1.0

    train_resample = torch.tensor(train_resample, dtype=torch.float64)
    target_resample = torch.tensor(target_resample, dtype=torch.float64)

    train_ = torch.tensor(train_resample, requires_grad=True).float().to(device)
    target_ = torch.tensor(target_resample, requires_grad=True).float().to(device)

    u_total, v_total, w_total, p_total, T_total, I_total, M_total, output = net_o(train_[:, 0:1], train_[:, 1:2], train_[:, 2:3], train_[:, 3:4], train_[:, 4:5], train_[:, 5:6], network, train_lb, train_ub)
    residual = (torch.abs(target_[:, 0:1] - u_total) + torch.abs(target_[:, 1:2] - v_total) + torch.abs(target_[:, 2:3] - w_total)) + torch.abs(target_[:, 5:6] - I_total) + torch.abs(target_[:, 6:7] - M_total) + torch.abs(target_[:, 4:5] - T_total)
    residual = residual.detach().cpu().numpy().ravel()
    index = np.argsort(-1.0 * np.abs(residual))[:200]

    points_X = train_resample[index]
    points_Y = target_resample[index]

    points_X = torch.tensor(points_X, dtype=torch.float64)
    points_Y = torch.tensor(points_Y, dtype=torch.float64)

    return points_X, points_Y

def sampling_plot(train, u_pred, v_pred, w_pred, points_X, epoch, path):

    points_x = points_X[:, 0:1]
    points_z = points_X[:, 2:3]

    # contour plot
    nn = 100

    xx = np.linspace(-0.095, 0.095, nn)
    yy = np.linspace(0, 0, nn)
    zz = np.linspace(0.005, 0.295, nn)

    X, Y, Z = np.meshgrid(xx, yy, zz)

    index = np.reshape(np.round(train[:, 4:5], 5) == 360, (train.shape[0],)) * np.reshape(np.round(train[:, 3:4], 5) == 0.00012, (train.shape[0],))
    idx_120_360 = np.array([i for i, x in enumerate(index) if x == True])

    u_predict = griddata(train[idx_120_360, 0:3], u_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    v_predict = griddata(train[idx_120_360, 0:3], v_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')
    w_predict = griddata(train[idx_120_360, 0:3], w_pred[idx_120_360].flatten(), (X, Y, Z), method='linear')

    # speed = sqrt(u ** 2 + v ** 2 + w ** 2)
    levels = np.linspace(0, 0.6493596846124651, 1000)
    plt.contourf(X[0], Z[0], (u_predict[0] ** 2 + v_predict[0] ** 2 + w_predict[0] ** 2) ** 0.5, levels, cmap='jet')
    cb = plt.colorbar()
    plt.scatter(points_x, points_z, marker="x", s=12, c='red')
    plt.gca().invert_yaxis()
    plt.axis('off')

    plt.savefig(path + '/{}'.format(epoch))

    plt.gca().invert_yaxis()
    cb.remove()