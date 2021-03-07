"""
TODO next time:
i am implementing the train and eval functions in the models.
i am thinking of how the values should be generated/and if they need to be updated? if yes, how?

Archives of codes
# GOT FROM PLOTTING
                    # utils.err_distribution(history, params, fig_path)
                
                # fig_path = params.output_dir +  '/figures/deviceSamples/Iter{}.png'.format(params.iter) 
                # GPR(history, params, fig_path)



                # if not params.generate_samples_mode:
                #     # SGD code
                #     z = generator.params_sgd()

                #     E_f, nu_f = utils.youngs_poisson(z[0][0, 0].detach().numpy(),
                #                     z[1][0, 0].detach().numpy())

                #     data = np.vstack([data, [E_f, nu_f]])
                #     fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter) 
                #     utils.err_distribution_sgd(data, params, fig_path)
"""

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
from net import *

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def evaluate(eng, numImgs, params):
    pass



def train(eng, params, pca=None):

    # initialization
    #TODO: enable restoring model
    if params.restore_from is None:
        data = np.zeros([0,3]) # data for gradient descent
        iter0 = 0   
        err = 0
    else:
        iter0 = params.checkpoint['iter']

    #TODO: params.models = ['gpr_1', 'nn', 'abcd'] or it could be a dictionary of {'model': params, 'model_2': params}
    # this naming convention allows us to set up different models of a single algorithm. name + _ + number
    active_models = {}
    for model in params.models:
        # #TODO: try to get models' arguments from params file. like 'model_1' : {'lr': 1, 'beta' : 2, ...}
        # if model in params: # checks if we have configuration for this model; type dict
        #     model_param = params.model
        # else: model_param = None
        model_param = None
        # active_models[model] = name
        # exec(f"from net import {name}")
        # model_ = exec(f"active_models[model](params, eng)")
        # active_models[model] = model_
        #import

        name = model.split('_')[0]
        

        if model_param: exec(f"{model} = {name}(model_params, eng)") #Init with params
        else:           exec(f"{model} = {name}(params, eng)"); active_models[f'{model}'] = model
        
        # experiment with having exec insantiate the class; how else I can insantiate?
    print(active_models)


    
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

            # mu amplitude in the tanh function
            if params.iter < 1000:
                params.binary_amp = int(params.iter/100) + 1 
            else:
                params.binary_amp = 10

            # terminate the loop
            if it > params.numIter:
                return 




            # training model:
            for model in active_models:
                # generate new samples
                #TODO: remove exec (for faster execution); is it faster to pass eng in each round or should we keep it in the model's memory?
                exec(f"{model}.train(eng, t)") #TODO: implement it



                # err, mu, beta, mu_sgd, beta_sgd = evaluate_training_generator(generator, eng, params)




                # add to history 
                



            # if not params.generate_samples_mode:
            #     # generate new values
            #     #
            #     # 
            #     # z = generator.params_sgd()

            #     # calculate efficiencies and gradients using EM solver
            #     effs, gradients, g_loss = compute_effs_and_gradients(z, eng, params) # gen_imgs ~ z
            #     t.set_description(f"Loss is {g_loss}", refresh=True)

            #     # compute gradients
                    

            #     optimizer.zero_grad()
            #     g_loss.backward()
            #     optimizer.step()

            # evaluate
            if it % 50 == 0 or it > params.numIter:
                pass
                # for model in active_models:
                #     exec(f"{model}.evaluate(eng, t)") #TODO: remove exec; implement evaluate

            # plot 
            if it % params.plot_iter == 0:
                #TODO: a unified structure for each model's plotting function is needed.
                for model in active_models:
                    fig_path = params.output_dir +  f'/figures/{model}/Iter{params.iter}.png'
                    exec(f'{model}.plot(fig_path)')

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
    effs = eng.Eval_Eff_1D_parallel(imgs)
    
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


def stress_distribution(imgs, eng, fig_path):
    import plotly.graph_objects as go
    import numpy as np

    v2d = vertex_to_dof_map(eng.model.V)
    imgs = imgs[0].flatten()[v2d].reshape(-1, 3)# / 10.#0.
    
    scene_settings = dict(
        xaxis = dict(range=[-1.2, 1.2], showbackground=False, zerolinecolor="black"),
        yaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"),
        zaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"))

    tribetas = []
    for cell in cells(eng.model.mesh):
        for facet in facets(cell):
            vertex_indices = []
            for vertex in vertices(facet):
                vertex_indices.append(vertex.index())
            vertex_dofs = eng.model.V.dofmap().entity_dofs(eng.model.mesh, 0, vertex_indices)
            tribetas.append(vertex_indices)
    tris = np.array(tribetas)

    x, y, z = eng.model.mesh.coordinates().T
    i, j, k = tris.T
    disp = np.linalg.norm(imgs.detach().numpy(), axis=1).T  # the zero index is because of the "N" above!

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=disp,
            # i, j and k give the vertices of tribetas
            # here we represent the 4 tribetas of the tetrahedron surface
            i=i,
            j=j,
            k=k,
            name='y',
            showscale=True
        )
    ])
    fig.update_layout(scene = scene_settings)
    fig.update_layout(scene_aspectmode = 'cube')
    
    fig.write_image(fig_path)

def save_images(imgs, eng, fig_path):
    import plotly.graph_objects as go
    import numpy as np
    v2d = vertex_to_dof_map(eng.model.V)

    imgs = imgs[0].flatten()[v2d].reshape(-1, 3)# / 10.#0.
    org_imgs = eng.target_deflection.flatten()[v2d].reshape(-1, 3)

    scene_settings = dict(
        xaxis = dict(range=[-1.2, 1.2], showbackground=False, zerolinecolor="black"),
        yaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"),
        zaxis = dict(range=[-1, 1], showbackground=False, zerolinecolor="black"))


    tribetas = []
    for cell in cells(eng.model.mesh):
        for facet in facets(cell):
            vertex_indices = []
            for vertex in vertices(facet):
                vertex_indices.append(vertex.index())
            tribetas.append(vertex_indices)
    tris = np.array(tribetas)

    x, y, z = (eng.model.mesh.coordinates() + imgs.detach().numpy()).T
    i, j, k = tris.T
    disp = np.linalg.norm(imgs.detach().numpy(), axis=1).T  # the zero index is because of the "N" above!

    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=disp,
            # i, j and k give the vertices of tribetas
            # here we represent the 4 tribetas of the tetrahedron surface
            i=i,
            j=j,
            k=k,
            name='y',
            showscale=True
        )
    ])

    # add the second graph
    x_, y_, z_ = (eng.model.mesh.coordinates() + org_imgs.detach().numpy()).T
    disp_ = np.linalg.norm(org_imgs.detach().numpy(), axis=1).T  # the zero index is because of the "N" above!
    fig.add_trace(go.Mesh3d(
        x=x_,
        y=y_,
        z=z_,
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=disp_,
        # i, j and k give the vertices of tribetas
        # here we represent the 4 tribetas of the tetrahedron surface
        i=i,
        j=j,
        k=k,
        colorscale = 'teal'
    ))

    fig.update_layout(scene = scene_settings)
    fig.update_layout(scene_aspectmode = 'cube')
    fig.write_image(fig_path)



def visualize_generated_images(generator, params, eng, n_row = 10, n_col = 1):
    # generate images and save
    fig_path = params.output_dir +  '/figures/deviceSamples/Iter{}.png'.format(params.iter) 
    
    z = sample_z(n_col * n_row, generator)[0] # generates n_row devices
    imgs = eng.Eval_Eff_1D_parallel(z)
    save_images(imgs, eng, fig_path)
    

def evaluate_training_generator(generator, eng, params, num_imgs = 1):
        # generate images
        t = sample_z(num_imgs, generator)
        z,v = t[0], t[1]
        

        # efficiencies of generated images
        effs = eng.Eval_Eff_1D_parallel(z)
        loss = torch.nn.MSELoss()
        error = loss(effs, eng.target_deflection)
        # error = loss(effs.cpu().detach(), eng.target_deflection)

        # get most recent mu and beta values
        mu_sgd, beta_sgd, force = generator.params_sgd()


        # plot histogram
        #TODO: replace utils.plot_histogram with wes' plotting function
        # fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter) 
        # utils.plot_histogram(error, params.iter, fig_path)

        
        return error.detach(), v[0], v[1], mu_sgd, beta_sgd

