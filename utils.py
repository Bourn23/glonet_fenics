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


def plot_loss_history(active_models, global_memory):
    for name, model in active_models.items():
        E_history, nu_history, eff_history = model.data[:, 0], model.data[:, 1], model.data[:, 2]
        iterations = [i*params.plot_iter for i in range(len(eff_history))]
        plt.figure()
        # logging.info(f"iterations is {iterations}")
        plt.plot(iterations, E_history)
        plt.plot(iterations, nu_history)
        plt.plot(iterations, eff_history)
        plt.xlabel('iteration')
        plt.legend(('E', 'nu', 'error'))
        plt.axis([0, len(eff_history)*params.plot_iter, 0, 1.05])
        plt.savefig(params.output_dir + f'/figures/Train_history_{name}.png')

        history_path = os.path.join(params.output_dir,f'history_{name}.mat')
        io.savemat(history_path, mdict={'effs_mean_history'   :np.asarray(effs_mean_history), 
                                        'beta_history'   :np.asarray(beta_history),
                                        'mu_history':np.asarray(mu_history)})

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
