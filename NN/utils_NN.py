import torch

rho = 520
mu = 0.0072
k_i = 1.54 * (10 ** 14)
Ea = 15023

def net_o(x, y, z, input_C, input_T, domain, network, train_lb, train_ub):
    output = network(torch.cat([x, y, z, input_C, input_T, domain], dim=1).float(), train_lb.float(), train_ub.float())

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]
    M = output[:, 6:7]

    return u, v, w, p, T, I, M, output

def net_f_in(x, y, z, input_C, input_T, domain, network, train_lb, train_ub, target_lb, target_ub):

    u, v, w, p, T, I, M, output = net_o(x, y, z, input_C, input_T, domain, network, train_lb, train_ub)

    target_lb = torch.tensor(target_lb).float().to(torch.cuda.current_device())
    target_ub = torch.tensor(target_ub).float().to(torch.cuda.current_device())

    output = (target_lb * (1.0 - output) + target_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]
    M = output[:, 6:7]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_y = torch.autograd.grad(I, y, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_z = torch.autograd.grad(I, z, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    # rad/s
    vel = 10.471975499999981

    # Relative Velocity viewed from rotational reference frame
    ur = u - vel * y
    vr = v + vel * x

    ur_x = u_x
    ur_y = u_y - vel
    ur_z = u_z

    vr_x = v_x + vel
    vr_y = v_y
    vr_z = v_z

    cont = u_x + v_y + w_z

    f_u = rho * (ur_x * ur + ur_y * vr + ur_z * w) + p_x - mu * (u_xx + u_yy + u_zz) - (vel * vel * x) - (2 * vel * vr)

    f_v = rho * (vr_x * ur + vr_y * vr + vr_z * w) + p_y - mu * (v_xx + v_yy + v_zz) - (vel * vel * y) + (2 * vel * ur)

    f_w = rho * (w_x * ur + w_y * vr + w_z * w) + p_z - mu * (w_xx + w_yy + w_zz)

    f_I = (u * I_x + u_x * I + v * I_y + v_y * I + w * I_z + w_z * I) - k_i * torch.exp(-Ea / torch.abs(T))

    return cont, f_u, f_v, f_w, f_I

def net_f_out(x, y, z, input_C, input_T, domain, network, train_lb, train_ub, target_lb, target_ub):
    # pytorch autograd version of calculating residual
    u, v, w, p, T, I, M, output = net_o(x, y, z, input_C, input_T, domain, network, train_lb, train_ub)

    target_lb = torch.tensor(target_lb).float().to(torch.cuda.current_device())
    target_ub = torch.tensor(target_ub).float().to(torch.cuda.current_device())

    output = (target_lb * (1.0 - output) + target_ub * (1.0 + output)) / 2.0

    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    T = output[:, 4:5]
    I = output[:, 5:6]
    M = output[:, 6:7]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    I_x = torch.autograd.grad(I, x, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_y = torch.autograd.grad(I, y, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]
    I_z = torch.autograd.grad(I, z, grad_outputs=torch.ones_like(I), retain_graph=True, create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), retain_graph=True, create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), retain_graph=True, create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), retain_graph=True, create_graph=True)[0]

    cont = u_x + v_y + w_z

    f_u = rho * (u_x * u + u_y * v + u_z * w) + p_x - mu * (u_xx + u_yy + u_zz)
    f_v = rho * (v_x * u + v_y * v + v_z * w) + p_y - mu * (v_xx + v_yy + v_zz)
    f_w = rho * (w_x * u + w_y * v + w_z * w) + p_z - mu * (w_xx + w_yy + w_zz)

    f_I = (u * I_x + u_x * I + v * I_y + v_y * I + w * I_z + w_z * I) - k_i * torch.exp(- Ea / torch.abs(T))

    return cont, f_u, f_v, f_w, f_I