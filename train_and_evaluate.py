
import os
import logging
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.nn.functional as F 
import torch
import torch.fft
import utils
import scipy.io as io
import numpy as np
from dolfin import *; from mshr import *

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def evaluate(generator, eng, numImgs, params):
    # generator.eval()
    
    # generate images
    # z = sample_z(numImgs, params)
    z = sample_z(numImgs, generator)
    images = generator(z, params)
    logging.info('Generation is done. \n')

    # evaluate efficiencies
    images = torch.sign(images)
    effs = compute_effs(images, eng, params)

    # save images
    filename = 'imgs_w' + str(params.wavelength) +'_a' + str(params.angle) +'deg.mat'
    file_path = os.path.join(params.output_dir,'outputs',filename)
    io.savemat(file_path, mdict={'imgs': images.cpu().detach().numpy(), 
                                 'effs': effs.cpu().detach().numpy()})

    # plot histogram
    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.numIter, fig_path)




def train(generator, optimizer, scheduler, eng, params, pca=None):

    # generator.train()

    # initialization
    if params.restore_from is None:
        effs_mean_history = []
        binarization_history = []
        diversity_history = []
        iter0 = 0   
    else:
        effs_mean_history = params.checkpoint['effs_mean_history']
        binarization_history = params.checkpoint['binarization_history']
        diversity_history = params.checkpoint['diversity_history']
        iter0 = params.checkpoint['iter']

    
    # training loop
    with tqdm(total=params.numIter) as t:
        it = 0  
        while True:
            it +=1 
            params.iter = it + iter0

            # normalized iteration number
            normIter = params.iter / params.numIter

            # specify current batch size 
            params.batch_size = int(params.batch_size_start +  (params.batch_size_end - params.batch_size_start) * (1 - (1 - normIter)**params.batch_size_power))
            
            # sigma decay
            params.sigma = params.sigma_start + (params.sigma_end - params.sigma_start) * normIter

            # learning rate decay
            scheduler.step()

            # binarization amplitude in the tanh function
            if params.iter < 1000:
                params.binary_amp = int(params.iter/100) + 1 
            else:
                params.binary_amp = 10

            # save model 
            if it % 5000 == 0 or it > params.numIter:
                  logging.info(f"generator values are {generator.parameters()}")
            #     model_dir = os.path.join(params.output_dir, 'model','iter{}'.format(it+iter0))
            #     os.makedirs(model_dir, exist_ok = True)
            #     utils.save_checkpoint({'iter': it + iter0 - 1,
            #                            'gen_state_dict': generator.state_dict(),
            #                            'optim_state_dict': optimizer.state_dict(),
            #                            'scheduler_state_dict': scheduler.state_dict(),
            #                            'effs_mean_history': effs_mean_history,
            #                            'binarization_history': binarization_history,
            #                            'diversity_history': diversity_history
            #                            },
            #                            checkpoint=model_dir)

            # terminate the loop
            if it > params.numIter:
                return 

            
            # sample  z
            # if it == 0: z = sample_z(params.batch_size, params)
            z = sample_z(params.batch_size, generator)

            # generate a batch of iamges; NN's is out of loop
            # gen_imgs = generator(z, params)


            # calculate efficiencies and gradients using EM solver
            effs, gradients, g_loss = compute_effs_and_gradients(z, eng, params) # gen_imgs ~ z

            # construct the loss function
            # binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            # g_loss = global_loss_function(gen_imgs, effs, gradients, params.sigma, binary_penalty)
            t.set_description(f"Loss is {g_loss}", refresh=True)

            # train the generator
            g_loss.backward()
            optimizer.step()
            # free optimizer buffer 
            optimizer.zero_grad()


            # evaluate 
            if it % params.plot_iter == 0:
                # generator.eval()

                # vilualize generated images at various conditions
                t.set_description(f"Values are {generator.parameters()}", refresh=True)
                visualize_generated_images(generator, params, eng)

                # evaluate the performance of current generator
                # effs_mean, binarization, diversity = evaluate_training_generator(generator, eng, params)

                # # add to history 
                # effs_mean_history.append(effs_mean)
                # binarization_history.append(binarization)
                # diversity_history.append(diversity)

                # plot current history
                # utils.plot_loss_history((effs_mean_history, diversity_history, binarization_history), params)
                # generator.train()

            t.update()



def sample_z(batch_size, generator):
    '''
    smaple noise vector z

    Returns:
        params: [mu, lambda, beta]
    '''
    return generator.generate()


def compute_effs_and_gradients(gen_imgs, eng, params):
    '''
    Args:
        imgs: N x C x H
        labels: N x labels_dim 
        eng: matlab engine
        params: parameters 

    Returns:
        effs: N x 1
        gradients: N x C x H
    '''
    img = gen_imgs

    effs_and_gradients, loss = eng.GradientFromSolver_1D_parallel(img)  
    effs = effs_and_gradients[0]          
    gradients = torch.tensor(effs_and_gradients[1:], dtype = torch.float64)

    return (effs, gradients, loss)


def compute_effs(imgs, eng, params):
    '''
    Args:
        imgs: N x C x H
        eng: matlab engine
        params: parameters 

    Returns:
        effs: N x 1
    '''
    img = imgs
    effs = eng.Eval_Eff_1D_parallel(img)
    
    return effs



def global_loss_function(gen_imgs, effs, gradients, sigma=0.5, binary_penalty=0):
    '''
    Args:
        gen_imgs: N x C x H (x W)
        effs: N x 1
        gradients: N x C x H (x W)
        max_effs: N x 1
        sigma: scalar
        binary_penalty: scalar
    '''
    loss = nn.MSEloss()
    output = loss(gen_imgs, effs)
    # efficiency loss
    gradients =  gradients.squeeze(2).T.unsqueeze(2).repeat(1, 88, 3)
    difference = torch.mean(torch.mean(effs - gen_imgs, dim=2), dim=1)

    actual_fft = torch.fft.fft(effs)
    pred_fft = torch.fft.fft(gen_imgs)
    fft_loss = torch.square(torch.real(actual_fft-pred_fft))
    fft_loss = torch.sum(torch.mean(fft_loss, dim=0).view(-1))

    eff_loss_tensor = - gen_imgs * gradients# * torch.exp(difference/fft_loss).view(-1, 1, 1)
    eff_loss = torch.sum(torch.mean(eff_loss_tensor, dim=0).view(-1))

    loss = eff_loss + fft_loss #+ time_loss
    return loss


def save_images(imgs, eng, fig_path):
    import plotly.graph_objects as go
    import numpy as np
    v2d = vertex_to_dof_map(eng.model.V)
    imgs = imgs[0].flatten()[v2d].reshape(-1, 3)# / 10.#0.
    # logging.info(f"images are {imgs}")
    scene_settings = dict(
        xaxis = dict(range=[-1.2, 1.2], showbackground=False, zerolinecolor="black"),
        yaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"),
        zaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"))


    # triangles = []
    # for cell in cells(eng.model.mesh):
    #     for facet in facets(cell):
    #         vertex_coords = []
    #         vertex_indices = []
    #         for vertex in vertices(facet):
    #             vertex_coords.append(list(vertex.point().array()))
    #             vertex_indices.append(vertex.index())
    #         triangles.append(vertex_indices)
    
    # tris = np.array(triangles)
    
    # x, y, z = eng.model.mesh.coordinates().T# (eng.model.mesh.coordinates()).T# + imgs.detach().numpy()).T
    # i, j, k = tris.T
    # disp = np.linalg.norm(imgs.detach().numpy(), axis=1).T
    # logging.info(f"disp is :{disp}")
    # fig = go.Figure(data=[
    #     go.Mesh3d(
    #         x=x,
    #         y=y,
    #         z=z,
    #         # Intensity of each vertex, which will be interpolated and color-coded
    #         intensity=disp,
    #         # i, j and k give the vertices of triangles
    #         # here we represent the 4 triangles of the tetrahedron surface
    #         i=i,
    #         j=j,
    #         k=k,
    #         name='y',
    #         showscale=True
    #     )
    # ])
    # fig.update_layout(scene = scene_settings)
    # fig.write_image(fig_path)

    disp = np.linalg.norm(imgs.numpy(), axis=1).T  # the zero index is because of the "N" above!

    x, y, z = eng.model.mesh.coordinates().T
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers',
                                    marker=dict(
            color=disp,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
        )
                                    )
    ])
    fig.update_layout(scene = scene_settings)
    fig.update_layout(scene_aspectmode = 'cube')
    # fig.show()
    fig.write_image(fig_path)



def visualize_generated_images(generator, params, eng, n_row = 10, n_col = 1):
    # generate images and save
    fig_path = params.output_dir +  '/figures/deviceSamples/Iter{}.png'.format(params.iter) 
    
    # z = sample_z(n_col * n_row, params) # generates n_row devices
    z = sample_z(n_col * n_row, generator) # generates n_row devices
    # imgs = generator(z, params)
    # imgs_2D = imgs.cpu().detach()
    imgs = eng.Eval_Eff_1D_parallel(z)
    save_images(imgs, eng, fig_path)
    
    


def evaluate_training_generator(generator, eng, params, num_imgs = 1):
    # generate images
    # z = sample_z(num_imgs, params)
    z = sample_z(num_imgs, generator)
    imgs = generator(z, params)

    # efficiencies of generated images
    effs = compute_effs(imgs, eng, params)
    # effs_mean = torch.mean(effs.view(-1))
    effs_mean = effs.cpu().detach().numpy()

    # binarization of generated images
    binarization = torch.mean(torch.abs(imgs.view(-1))).cpu().detach().numpy()

    # diversity of generated images
    diversity = torch.mean(torch.std(imgs, dim=0)).cpu().detach().numpy()

    # plot histogram
    fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter) 
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.iter, fig_path)

    
    return effs_mean, binarization, diversity

