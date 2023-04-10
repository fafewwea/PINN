import torch
import numpy as np
from utils import net_o, net_f_in, net_f_out
import math

def train_loss(train_batch, target_batch, device, network, w_uvw, w_P, w_T, w_I, w_M, w_phy, train_lb, train_ub, target_lb, target_ub):

    x = torch.tensor(train_batch[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(train_batch[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(train_batch[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(train_batch[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(train_batch[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(train_batch[:, 5:6], requires_grad=True).float().to(device)

    u = torch.tensor(target_batch[:, 0:1], requires_grad=True).float().to(device)
    v = torch.tensor(target_batch[:, 1:2], requires_grad=True).float().to(device)
    w = torch.tensor(target_batch[:, 2:3], requires_grad=True).float().to(device)
    p = torch.tensor(target_batch[:, 3:4], requires_grad=True).float().to(device)
    T = torch.tensor(target_batch[:, 4:5], requires_grad=True).float().to(device)
    I = torch.tensor(target_batch[:, 5:6], requires_grad=True).float().to(device)
    M = torch.tensor(target_batch[:, 6:7], requires_grad=True).float().to(device)

    index = np.reshape(train_batch[:, 5:6] == 1, (train_batch.shape[0],))
    in_idx = np.array([i for i, x in enumerate(index) if x == True])

    index = np.reshape(train_batch[:, 5:6] == 0, (train_batch.shape[0],))
    out_idx = np.array([i for i, x in enumerate(index) if x == True])

    x_in = torch.tensor(train_batch[in_idx, 0:1], requires_grad=True).float().to(device)
    y_in = torch.tensor(train_batch[in_idx, 1:2], requires_grad=True).float().to(device)
    z_in = torch.tensor(train_batch[in_idx, 2:3], requires_grad=True).float().to(device)
    input_C_in = torch.tensor(train_batch[in_idx, 3:4], requires_grad=True).float().to(device)
    input_T_in = torch.tensor(train_batch[in_idx, 4:5], requires_grad=True).float().to(device)
    domain_in = torch.tensor(train_batch[in_idx, 5:6], requires_grad=True).float().to(device)

    x_out = torch.tensor(train_batch[out_idx, 0:1], requires_grad=True).float().to(device)
    y_out = torch.tensor(train_batch[out_idx, 1:2], requires_grad=True).float().to(device)
    z_out = torch.tensor(train_batch[out_idx, 2:3], requires_grad=True).float().to(device)
    input_C_out = torch.tensor(train_batch[out_idx, 3:4], requires_grad=True).float().to(device)
    input_T_out = torch.tensor(train_batch[out_idx, 4:5], requires_grad=True).float().to(device)
    domain_out = torch.tensor(train_batch[out_idx, 5:6], requires_grad=True).float().to(device)

    u_pred, v_pred, w_pred, p_pred, T_pred, I_pred, M_pred, output = net_o(x, y, z, input_C, input_T, domain, network, train_lb, train_ub)
    cont_in, f_u_in, f_v_in, f_w_in, f_I_in = net_f_in(x_in, y_in, z_in, input_C_in, input_T_in, domain_in, network, train_lb, train_ub, target_lb, target_ub)
    cont_out, f_u_out, f_v_out, f_w_out, f_I_out = net_f_out(x_out, y_out, z_out, input_C_out, input_T_out, domain_out, network, train_lb, train_ub, target_lb, target_ub)


    # uvw data loss
    loss1 = torch.mean(torch.abs(u - u_pred)) + torch.mean(torch.abs(v - v_pred)) + torch.mean(torch.abs(w - w_pred))

    # P data loss
    loss2 = torch.mean(torch.abs(p - p_pred))

    # T data loss
    loss3 = torch.mean(torch.abs(T - T_pred))

    # I data loss
    loss4 = torch.mean(torch.abs(I - I_pred))

    # M data Loss
    loss5 = torch.mean(torch.abs(M - M_pred))

    # continuity loss
    loss6 = torch.mean(torch.abs(cont_in)) + torch.mean(torch.abs(cont_out))

    if math.isnan(loss6) == True:
        loss6 = torch.tensor(0).float().to(device)
    else:
        loss6 = loss6

    # momentum loss
    loss7 = torch.mean(torch.abs(f_u_in)) + torch.mean(torch.abs(f_v_in)) + torch.mean(
        torch.abs(f_w_in)) + torch.mean(torch.abs(f_u_out)) + torch.mean(torch.abs(f_v_out)) + torch.mean(
        torch.abs(f_w_out))

    if math.isnan(loss7) == True:
        loss7 = torch.tensor(0).float().to(device)
    else:
        loss7 = loss7

    # species transport
    loss8 = torch.mean(torch.abs(f_I_in)) + torch.mean(torch.abs(f_I_out))

    if math.isnan(loss8) == True:
        loss8 = torch.tensor(0).float().to(device)
    else:
        loss8 = loss8

    loss = (loss1 * w_uvw + loss2 * w_P + loss3 * w_T + loss4 * w_I + loss5 * w_M) * 100

    return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8

def validation_loss(X_val, Y_val, device, network, w_uvw, w_P, w_T, w_I, w_M, w_phy, train_lb, train_ub, target_lb, target_ub):

    x = torch.tensor(X_val[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(X_val[:, 1:2], requires_grad=True).float().to(device)
    z = torch.tensor(X_val[:, 2:3], requires_grad=True).float().to(device)
    input_C = torch.tensor(X_val[:, 3:4], requires_grad=True).float().to(device)
    input_T = torch.tensor(X_val[:, 4:5], requires_grad=True).float().to(device)
    domain = torch.tensor(X_val[:, 5:6], requires_grad=True).float().to(device)

    u = torch.tensor(Y_val[:, 0:1], requires_grad=True).float().to(device)
    v = torch.tensor(Y_val[:, 1:2], requires_grad=True).float().to(device)
    w = torch.tensor(Y_val[:, 2:3], requires_grad=True).float().to(device)
    p = torch.tensor(Y_val[:, 3:4], requires_grad=True).float().to(device)
    T = torch.tensor(Y_val[:, 4:5], requires_grad=True).float().to(device)
    I = torch.tensor(Y_val[:, 5:6], requires_grad=True).float().to(device)
    M = torch.tensor(Y_val[:, 6:7], requires_grad=True).float().to(device)

    index = np.reshape(X_val[:, 5:6] == 1, (X_val.shape[0],))
    in_idx = np.array([i for i, x in enumerate(index) if x == True])

    index = np.reshape(X_val[:, 5:6] == 0, (X_val.shape[0],))
    out_idx = np.array([i for i, x in enumerate(index) if x == True])

    x_in = torch.tensor(X_val[in_idx, 0:1], requires_grad=True).float().to(device)
    y_in = torch.tensor(X_val[in_idx, 1:2], requires_grad=True).float().to(device)
    z_in = torch.tensor(X_val[in_idx, 2:3], requires_grad=True).float().to(device)
    input_C_in = torch.tensor(X_val[in_idx, 3:4], requires_grad=True).float().to(device)
    input_T_in = torch.tensor(X_val[in_idx, 4:5], requires_grad=True).float().to(device)
    domain_in = torch.tensor(X_val[in_idx, 5:6], requires_grad=True).float().to(device)

    x_out = torch.tensor(X_val[out_idx, 0:1], requires_grad=True).float().to(device)
    y_out = torch.tensor(X_val[out_idx, 1:2], requires_grad=True).float().to(device)
    z_out = torch.tensor(X_val[out_idx, 2:3], requires_grad=True).float().to(device)
    input_C_out = torch.tensor(X_val[out_idx, 3:4], requires_grad=True).float().to(device)
    input_T_out = torch.tensor(X_val[out_idx, 4:5], requires_grad=True).float().to(device)
    domain_out = torch.tensor(X_val[out_idx, 5:6], requires_grad=True).float().to(device)

    u_pred, v_pred, w_pred, p_pred, T_pred, I_pred, M_pred, output = net_o(x, y, z, input_C, input_T, domain, network, train_lb, train_ub)
    cont_in, f_u_in, f_v_in, f_w_in, f_I_in = net_f_in(x_in, y_in, z_in, input_C_in, input_T_in, domain_in, network, train_lb, train_ub, target_lb, target_ub)
    cont_out, f_u_out, f_v_out, f_w_out, f_I_out = net_f_out(x_out, y_out, z_out, input_C_out, input_T_out, domain_out, network, train_lb, train_ub, target_lb, target_ub)

    # uvw data loss
    loss1 = torch.mean(torch.abs(u - u_pred)) + torch.mean(torch.abs(v - v_pred)) + torch.mean(torch.abs(w - w_pred))

    # P data loss
    loss2 = torch.mean(torch.abs(p - p_pred))

    # T data loss
    loss3 = torch.mean(torch.abs(T - T_pred))

    # I data loss
    loss4 = torch.mean(torch.abs(I - I_pred))

    # M data Loss
    loss5 = torch.mean(torch.abs(M - M_pred))

    # continuity loss
    loss6 = torch.mean(torch.abs(cont_in)) + torch.mean(torch.abs(cont_out))

    if math.isnan(loss6) == True:
        loss6 = torch.tensor(0).float().to(device)
    else:
        loss6 = loss6

    # momentum loss
    loss7 = torch.mean(torch.abs(f_u_in)) + torch.mean(torch.abs(f_v_in)) + torch.mean(
        torch.abs(f_w_in)) + torch.mean(torch.abs(f_u_out)) + torch.mean(
        torch.abs(f_v_out)) + torch.mean(torch.abs(f_w_out))

    if math.isnan(loss7) == True:
        loss7 = torch.tensor(0).float().to(device)
    else:
        loss7 = loss7

    # species transport
    loss8 = torch.mean(torch.abs(f_I_in)) + torch.mean(torch.abs(f_I_out))

    if math.isnan(loss8) == True:
        loss8 = torch.tensor(0).float().to(device)
    else:
        loss8 = loss8

    val_loss = (loss1 * w_uvw + loss2 * w_P + loss3 * w_T + loss4 * w_I + loss5 * w_M) * 100

    return val_loss