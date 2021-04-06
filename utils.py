"""General utility functions"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import json
import logging
import csv
import scipy.io as io
import torch
import numpy as np

from dolfin import *; from mshr import *


import glob
import shutil
from PIL import Image

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def row_csv2dict(csv_file):
    dict_club={}
    with open(csv_file)as f:
        reader=csv.reader(f,delimiter=',')
        for row in reader:
            dict_club[(row[0],row[1])]=row[2]
    return dict_club


def save_checkpoint(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'model.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)



def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['gen_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def plot_loss_history(params, active_models, global_memory, global_count):
    #TODO: fix when we only have 1 model.
    path = params.output_dir + f'/figures/ensemble_of_decision_{global_count}.png'
    fig, ax = plt.subplots(1, len(active_models), figsize = (9, 4))
    counter = 0
    for name, model in active_models.items():
       
        # ax[counter] = model.plot(path, global_memory, axis = ax[counter])
        if len(active_models) == 1:
            model.plot(path, global_memory, axis = ax)
        else:
            model.plot(path, global_memory, axis = ax[counter])
        counter += 1



        # iterations = [i*params.plot_iter for i in range(len(eff_history))]
        # fig, ax = plt.subplots(1, 3, figsize=(6,3))
        # # logging.info(f"iterations is {iterations}")
        # ax[0].plot(iterations, eff_history)
        # ax[0].set_xlabel('iteration')
        # ax[0].set_title('Error History')
        # ax[0].axis([0, len(eff_history)*params.plot_iter, 0, 1.05])

        # ax[1].plot(iterations, nu_history)
        # ax[1].set_xlabel('iteration')
        # ax[1].set_title('Nu History')
        # ax[1].axis([0, len(nu_history)*params.plot_iter, 0, 1.05])

        # ax[2].plot(iterations, E_history)
        # ax[2].set_xlabel('iteration')
        # ax[2].set_title('E History')
        # ax[2].axis([0, len(E_history)*params.plot_iter, 0, 1.05])
        # plt.legend(('E', 'nu', 'error'))


        #TODO: uncomment for enable saving
        # E_history, nu_history, eff_history = model.data[:, 0], model.data[:, 1], model.data[:, 2]
        # history_path = os.path.join(params.output_dir,f'history_{name}.mat')
        # io.savemat(history_path, mdict={'eff_history'   :np.asarray(eff_history), 
        #                                 'E_history'   :np.asarray(E_history),
        #                                 'nu_history':np.asarray(nu_history)})

    plt.savefig(path)
    plt.close()
         
# convergence of the algorithms (error - iteration)

def plot_histogram(Effs, Iter, fig_path):
    ax = plt.figure()
    plt.plot(Effs, alpha=0.5)
    # plt.xlim(0, 100)
    # plt.ylim(0, 50)
    # plt.yticks([])
    # plt.xticks(fontsize=12)
    #plt.yticks(fontsize=20)
    plt.xlabel('Loss (%)', fontsize=12)
    # plt.title('Iteration {}'.format(Iter), fontsize=16)
    plt.savefig(fig_path, dpi=300)
    plt.close()


# ___ WBR Helper functions ___
def magnitude(num):
    new_num = str(round(num, 0))
    count = len(new_num[1:-2])
    return count

def youngs_poisson(mu, lambda_):
    youngs = mu * (3 * lambda_ + 2 * mu) / (lambda_ + mu)
    poisson = lambda_ / (2 * (lambda_ + mu))
    return youngs, poisson

def lame(E, nu):
    mu = E / (2*(1+nu))
    lambda_ = E*nu / ((1+nu)*(1-2*nu))
    return mu, lambda_


def err_distribution(data, params, fig_path):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=np.log(data[:, 2]))
    ax.plot(params.E_0, params.nu_0, 'kx')

    plt.savefig(fig_path, dpi = 300)
    plt.close()

def err_distribution_sgd(data, params, fig_path):
    fig, ax = plt.subplots()

    ax.contourf(X, Y, Z.reshape(X.shape))
    ax.plot(data[:, 0], data[:, 1], 'rx')  # values obtained by torch
    ax.plot(params.E_0, params.nu_0, 'ws')  # white = true value

    plt.savefig(fig_path, dpi = 300)
    plt.close()

# ___ PLOTTING ___
def make_gif_from_folder(folder, out_file_path, remove_folder=False):
    files = os.path.join(folder, '*.png')
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(files))]
    img.save(fp=out_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
    if remove_folder:
        shutil.rmtree(folder, ignore_errors=True)


def plot_3d(eng, particles=None, velocity=None, normalize=True, color='#000', ax=None):
    import plotly.graph_objects as go
    import numpy as np                      #8.5 # .32
    X_grid, Y_grid = np.meshgrid(np.linspace(8*1e6, 9*1e6, 25), # seq(start, stop, #)
                                 np.linspace(0.25, 0.45, 25))
    x, y = X_grid, Y_grid
    X_grid, Y_grid = lame(X_grid, Y_grid)
    # does it make a disfference? how to make it more efficient?
    Z_grid, syn_data = eng.GradientFromSolver_1D_parallel({'mu': X_grid, 'beta': Y_grid})

    z = Z_grid * 1e14
    print('min z', z.min())
    print('location of min,z is', z.argmin())
    print('the 2dlocation is', np.where(z == z.min()))
    x_min, y_min = np.where(z == z.min())
    sh_0, sh_1 = z.shape
    for i in range(3):
        fig = None
        print('i is ', i)
        if i == 0:
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            fig.update_layout(scene = dict(
                xaxis = dict(nticks=5, range=[x.min(),x.max()],),
                yaxis = dict(nticks=6, range=[y.min(),y.max()],),
                zaxis = dict(nticks=5, range=[z.min(),z.max() + 0.1],),
                zaxis_title = "Displacement (1E4)",
                xaxis_title="Young's Modulus",
                yaxis_title="Poisson's Ratio",),
                title='Displacement Field of E, nu', autosize=True,   
                width=500, height=500,
                margin=dict(l=2, r=50, b=65, t=90),    
            )
        if i == 1:
            fig = go.Figure(data=[go.Surface(z=z[:,::-1], x=x, y=y[:,::-1])])

            # indices of minimum
            x_min, y_min = np.where(z[:,::-1] == z[:,::-1].min())
            x_min, y_min = x_min[0], y_min[0]
            y_min = np.where(y[:,::-1] == y[x_min, y_min])

            y = y[:,::-1]
            z = z[:,::-1]
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y),
                go.Scatter3d(z=[z.min()], x=[x[x_min]], y=[y[y_min]], mode='markers', marker=dict(size=25, color=z, colorscale='reds'), opacity = 1)])
            
            fig.update_layout(scene = dict(
                xaxis = dict(nticks=5, range=[x.min(),x.max()],),
                yaxis = dict(nticks=6, range=[y.max(),y.min()],),
                zaxis = dict(nticks=5, range=[z.min() - 1,z.max() + 0.5],),
                zaxis_title = "Displacement (1E14)",
                xaxis_title="Young's Modulus",
                yaxis_title="Poisson's Ratio",),
                title='Displacement Field of E, nu', autosize=True,   
                width=500, height=500,
                margin=dict(l=2, r=50, b=65, t=90),    
            )

            camera = dict(
                eye=dict(x=2, y=2, z=0.64)
            )
            fig.update_layout(scene_camera=camera)
        if i == 2:
            fig = go.Figure(data=[go.Surface(z=z[:,::-1], x=x, y=y[::-1,::-1])])
            fig.update_layout(scene = dict(
                            xaxis = dict(nticks=5, range=[x.min(),x.max()],),
                            yaxis = dict(nticks=6, range=[y.max(),y.min()],),
                            zaxis = dict(nticks=5, range=[z.min(),z.max() + 0.1],),
                            zaxis_title = "Displacement (1E4)",
                            xaxis_title="Young's Modulus",
                            yaxis_title="Poisson's Ratio",),
                            title='Displacement Field of E, nu', autosize=True,   
                            width=500, height=500,
                            margin=dict(l=2, r=50, b=65, t=90),    
            )

        # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
        #                 highlightcolor="limegreen", project_z=True))
        fig.write_image(f'./results/figures/error_history/3d_plot_{i}.png')
        # scene_settings = dict(
        #         xaxis = dict(range=[X_grid.min(), X_grid.max()], showbackground=False, zerolinecolor="black"),
        #         yaxis = dict(range=[Y_grid.min(), Y_grid.max()], showbackground=False, zerolinecolor="black"),
        #         zaxis = dict(range=[Z_grid.min(), Z_grid.max() + 1], showbackground=False, zerolinecolor="black"))


        # x, y, z = (X_grid, Y_grid, Z_grid)

        # disp = np.linalg.norm(Z_grid, axis=1).T  # the zero index is because of the "N" above!

        # fig = go.Figure(data=[
        #     go.Mesh3d(
        #         x=x, y=y, z=z,
        #         # Intensity of each vertex, which will be interpolated and color-coded
        #         intensity=disp,
        #         name='y', showscale=True
        #     )
        # ])
        # fig.update_layout(scene = scene_settings)
        # fig.update_layout(scene_aspectmode = 'cube')


    return syn_data

# def plot_3d(eng, particles=None, velocity=None, normalize=True, color='#000', ax=None):
#     from matplotlib import cm
#     from matplotlib.ticker import LinearLocator, FormatStrFormatter

#     cmap = cm.colors.LinearSegmentedColormap.from_list('Custom',
#                                                    [(0, '#2f9599'),
#                                                     (0.45, '#eee'),
#                                                     (1, '#8800ff')], N=256)
#     X_grid, Y_grid = np.meshgrid(np.linspace(8, 9, 2), # seq(start, stop, #)
#                                  np.linspace(0.25, 0.45, 2))
#     # does it make a difference? how to make it more efficient?
#     Z_grid, syn_data = eng.GradientFromSolver_1D_parallel({'mu': X_grid, 'beta': Y_grid})

#     # 400 synthetic U
#     # print('Z_Grids is', Z_grid)
    
#     # get coordinates and velocity arrays
#     if particles is not None:
#         X, Y = particles.swapaxes(0, 1)
#         Z = eng.GradientFromSolver_1D_parallel(X, Y) # gotta fix if we want to visualize swarm
#         if velocity is not None:
#             U, V = velocity.swapaxes(0, 1)
#             W = eng.GradientFromSolver_1D_parallel(X + U, Y + V) - Z # gotta fix if we want to visualize swarm

#     # create new ax if None
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1, projection='3d')

#     # Plot the surface.
#     surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap,
#                            linewidth=0, antialiased=True, alpha=0.7)
#     ax.contour(X_grid, Y_grid, Z_grid, zdir='z', offset=0, levels=30, cmap=cmap)
#     if particles is not None:
#         ax.scatter(X, Y, Z, color=color, depthshade=True)
#         if velocity is not None:
#             ax.quiver(X, Y, Z, U, V, W, color=color, arrow_length_ratio=0., normalize=normalize)

#     len_space = 10
#     # Customize the axis
#     max_z = (np.max(Z_grid) // len_space + 1).astype(np.int) * len_space
#     ax.set_xlim3d(np.min(X_grid), np.max(X_grid))
#     ax.set_ylim3d(np.min(Y_grid), np.max(Y_grid))
#     ax.set_zlim3d(0, max_z)
#     ax.zaxis.set_major_locator(LinearLocator(max_z // len_space + 1))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#     # Rmove fills and set labels
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     ax.set_xlabel('$E$')
#     ax.set_ylabel('$Nu$')
#     ax.set_zlabel('$delta$')

#     # Add a color bar which maps values to colors.
#     # fig.colorbar(surf)
#     plt.savefig('./results/figures/error_history/3d_plot.png', dpi = 300)
#     plt.close()

#     return syn_data



def save_images(imgs, eng, fig_path):
    import plotly.graph_objects as go
    import numpy as np
    
    v2d = vertex_to_dof_map(eng.model.V)

    # imgs = imgs[0].flatten()[v2d].reshape(-1, 3)# / 10.#0.
    imgs = eng.target_deflection.flatten()[v2d].reshape(-1, 3) * 1e5 # orig_imgs => imgs

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

    # add the second graph - TRUE VALUE
    # x_, y_, z_ = (eng.model.mesh.coordinates() + org_imgs.detach().numpy()).T
    # disp_ = np.linalg.norm(org_imgs.detach().numpy(), axis=1).T  # the zero index is because of the "N" above!
    # fig.add_trace(go.Mesh3d(
    #     x=x_,
    #     y=y_,
    #     z=z_,
    #     # Intensity of each vertex, which will be interpolated and color-coded
    #     intensity=disp_,
    #     # i, j and k give the vertices of tribetas
    #     # here we represent the 4 tribetas of the tetrahedron surface
    #     i=i,
    #     j=j,
    #     k=k,
    #     colorscale = 'teal'
    # ))

    fig.update_layout(scene = scene_settings)
    fig.update_layout(scene_aspectmode = 'cube')
    fig.write_image(fig_path)

def save_imagesz(imgs, eng, fig_path):
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