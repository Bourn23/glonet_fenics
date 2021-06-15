## __ IMPORTS __
from torch_fenics_model import varied_density_field,  plot_only
import torch
from tqdm import trange
from dolfin import *
import math
import numpy as np
t = trange(200, desc = "", leave = True)
## __ SAMPLE GENERATOR __
def sample_z(batch_size, noise_dims, noise_amplitude):
    '''
    smaple noise vector z
    '''
    return (torch.rand(batch_size, noise_dims[0], noise_dims[1]).type(torch.float64)*2.-1.) * noise_amplitude

## __ HYPER PARAMS __
batch_size = 4
noise_dims = (40, 30)
noise_amplitude = 0.5

# theta_generated = torch.clip(sample_z(batch_size, noise_dims, noise_amplitude).reshape(1, -1), 0.001, 1.)
## __ GENERATE SAMPLE __
noise = torch.randn(batch_size * noise_dims[0] * noise_dims[1], dtype = torch.float64)
v = torch.ones_like(noise) * 0.5 + noise * 1e-3
v = torch.clip(v, .001, 1.).requires_grad_()
optimizer = torch.optim.Adam([v], lr = 1e-2)
c = v.clone()

## __ VISUALIZE GENERATED SAMPLE __
visualize = plot_only()
u_ = visualize(v.reshape(1, -1))

## __ PHYSICS ENGINE __
varproblem = varied_density_field()

## __ LOAD GROUND TRUTH DENSITY __
# penf = []
# with open('./gt_density.txt', 'r') as f:
#     denf = f.readlines()
# for vi in denf:
#     penf.append(float(vi.split('\n')[0]))
# penf = torch.tensor(penf, dtype = torch.float64)
# U = visualize(penf.reshape(1, -1))
# ## -- OPTIMIZING FOR DENSITY -- NO PHYSICS ENGINE JUST MSE
# loss = torch.nn.MSELoss()
# for _ in tqdm.tqdm(range(100)):
#     diff = loss(penf, v.flatten())
#     optimizer.zero_grad()
#     diff.backward()
#     # print(v.grad)
#     optimizer.step()
#     # print('different ?', torch.any(v != c))
# u = visualize(v.reshape(1, -1))

## COMMENTED
# penf = torch.tensor([penf], requires_grad=True, dtype=torch.float64)
# print('before', v)

# # theta, u = varproblem(theta_generated)
# # ?theta, \
# # print(type(v))
# # print(v)


# # -- PUTTING THE PHYSICS ENGINE IN THE LOOP AND COMPARING DISPLACEMENTS --
gt_disp = []
with open('./GLOnet/gt_displacement.txt', 'r') as f:
    dsp = f.readlines()
for vii in dsp:
    gt_disp.append(float(vii.split('\n')[0]))
gt_disp = torch.tensor(gt_disp, dtype = torch.float64)
# u = visualize(gt_disp.reshape(1, -1))
# print('gt mean squared', torch.mean(gt_disp**2))


# read in the indices
# with open('./gt_indices.txt', 'r') as f:
indices = np.genfromtxt('./GLOnet/gt_indices.txt', dtype = 'int32')
# print('indexes are ', indices)

true_density = np.loadtxt('./GLOnet/gt_density.txt')
print('true density values are ', true_density)

loss = torch.nn.MSELoss()
print('density field values before updating', v.reshape(1, -1))
# for i in range(100):
# mesh00 = RectangleMesh(Point(-2, 0), Point(2, 1), 60, 40, "crossed")
for _ in t:
    u = varproblem(v.reshape(1, -1))
    u = u.flatten()[indices]
    # computing loss
    #TODO: indexing here for the vertices
    diff = torch.log(loss(gt_disp[indices], u))
    optimizer.zero_grad()

    diff.backward()
    # print('grad', v.grad)
    optimizer.step()
    # v  = torch.clip(v, 0.001, 1).requires_grad_(True)

    # update progess bar
    t.set_description(f'error: {diff}')
    t.refresh()
# print('after', v)
#     print('loss is ', diff)
#     print('different ?', torch.any(v != c))
visualize = plot_only()
u = visualize(v.reshape(1, -1))
# print(u)
# u = visualize(u.reshape(1, -1))
print('density field values after updating', v.reshape(1, -1))


#
# ft_disp = []
# with open('./fake_displacement.txt', 'r') as f:
#     dsp_f = f.readlines()
# for v in dsp_f:
#     ft_disp.append(float(v.split('\n')[0]))
#
# for m,n in zip(gt_disp, ft_disp):
#     print(math.isclose(m, n, rel_tol = 0.2, abs_tol = 0.1))
# plot(u, 'Theta')
# plt.show();
#
# plot(u, 'displacement')
# plt.show();

