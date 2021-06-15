from torch_fenics_model import *
from matlab import *
import time
import os
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train, summarize
from net import Model
import utils
import torch
from tqdm import trange

# Adaptive Experiment
from ax import Arm, ChoiceParameter, Models, ParameterType, SearchSpace, SimpleExperiment
from ax.plot.scatter import plot_fitted
# from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem
 

# start matlab engine
varproblem = varied_density_field()




# parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results',
                                        help="Results folder")
parser.add_argument('--mu', default=None) # mu = mu
parser.add_argument('--beta', default=None) # beta = lambda
parser.add_argument('--restore_from', default=None,
                                        help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Load the directory from commend line
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.output_dir, 'train.log'))

    # Load parameters from json file
    json_path = os.path.join(args.output_dir,'Params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # model, num_iterations

    search_space = SearchSpace(
        parameters=[
            ChoiceParameter(
                name="model",
                parameter_type=ParameterType.STRING,
                values=["SGD", "PSOL"],
                #values=["SGD", "GAP", "GPR", "PSOL"],
            ),
            ChoiceParameter(
                name="batch",
                parameter_type=ParameterType.STRING,
                values=["50", "60", "70"],
            ),
        ]
    )

    exp = SimpleExperiment(
        name="my_factorial_closed_loop_experiment",
        search_space=search_space,
        evaluation_function=summarize,
        objective_name="success_metric",
    )
    exp.status_quo = Arm(
        parameters={"model": "SGD", "batch": "50"}
    )

    factorial = Models.FACTORIAL(search_space=exp.search_space)
    factorial_run = factorial.gen(n=-1)  # Number of arms to generate is derived from the search space.
    print(factorial_run.arms[0].parameters['model'])

    models_rslt = []
    for i in range(len(factorial_run.arms)):

        # Add attributes to params
        params.output_dir = args.output_dir
        params.cuda = torch.cuda.is_available()
        params.restore_from = args.restore_from
        params.numIter = int(factorial_run.arms[i].parameters['batch']) # int(params.numIter)
        params.numGenerations = int(params.numGenerations)
        params.generate_samples_mode = int(params.generate_samples_mode)

        try:       params.noise_dims = int(params.noise_dims)
        except:    params.noise_dims = list(params.noise_dims)

        params.gkernlen = int(params.gkernlen)
        params.step_size = int(params.step_size)
        params.force = float(params.force)

        # TODO:must fix this. no longer need it!
        if args.mu is not None:
            params.E_0 = torch.tensor(args.mu, requires_grad = True, dtype = torch.float64)
        if args.beta is not None:
            params.nu_0 = torch.tensor(args.beta, requires_grad = True, dtype = torch.float64)



        # eng = engine(varproblem, params.batch_size_start, params.E_0, params.nu_0, params.force)

        # Retrieve GTrurth
        penf = []
        with open('./gt_density.txt', 'r') as f:
            denf = f.readlines()
        for v in denf:
            penf.append(float(v.split('\n')[0]))

        penf = torch.tensor([penf], requires_grad=True, dtype=torch.float64)

        eng = engine(varproblem, batch_size=4, density_field=penf)
        # make directory
        os.makedirs(args.output_dir + '/outputs', exist_ok = True)
        os.makedirs(args.output_dir + '/model', exist_ok = True)
        os.makedirs(args.output_dir + '/figures/histogram', exist_ok = True)
        os.makedirs(args.output_dir + '/figures/deviceSamples', exist_ok = True)
        os.makedirs(args.output_dir + '/figures/error_history', exist_ok = True)

        models = [factorial_run.arms[i].parameters['model']]
        print("params model is ", models)
        for model in models:
            os.makedirs(args.output_dir + f'/figures/{model}', exist_ok = True)

        # utils.save_images(eng.target_deflection, eng, './results/figures/error_history/3d_plot_real_column.png')
        # Define the models
        global_memory = Model(params)

        # Move to gpu if possible
        # if params.cuda:
        #     generator.cuda()


        # Define the optimizer
        #optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))

        # Define the scheduler
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma = params.gamma)


        # Load model data
        if args.restore_from is not None :
            params.checkpoint = utils.load_checkpoint(restore_from, generator, optimizer, scheduler)
            logging.info('Model data loaded')


        # Train the model and save
        # for replica in tqdm.tqdm(np.arange(params.numGenerations)):
        draw_contour = False
        if draw_contour:
            print('Computing contour')
            x, y, z = utils.plot_3d(eng)
            global_memory.contour_X = x
            global_memory.contour_Y = y
            global_memory.contour_Z = z
        for global_optimizer in trange(params.numGenerations, desc = "Global Optimization"):
            start_time = time.time()
            if params.numIter != 0 :
                train(eng, params, global_memory, global_optimizer, models)
            end_time = time.time()
            elapsed = round(end_time - start_time, 2)
            logging.info('Evaluate Results and Ensemble Process')
            evaluate(eng, params, global_memory, global_optimizer, elapsed)
        # Generate images and save
        # logging.info('Start generating devices')
        logging.info('\nWrapping Up Results')
        reslt_sumry = summarize(global_memory)
        models_rslt.append(reslt_sumry)
        print(models_rslt)