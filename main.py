from torch_fenics_model import *
from matlab import *
import os
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train
from net import Generator
import utils
import torch
from tqdm import tqdm
 

# start matlab engine
varproblem = HomogeneousBeam()



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

    # Add attributes to params
    params.output_dir = args.output_dir
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.numIter = int(params.numIter)
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



    eng = engine(varproblem, params.batch_size_start, params.E_0, params.nu_0, params.force)

    # make directory
    os.makedirs(args.output_dir + '/outputs', exist_ok = True)
    os.makedirs(args.output_dir + '/model', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/histogram', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/deviceSamples', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/error_history', exist_ok = True)

    for model in params.models:
        os.makedirs(args.output_dir + f'/figures/{model}', exist_ok = True)


    # Define the models 
    #generator = Generator(params)
        
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
    for global_optimizer in tqdm(range(params.numGenerations)):
        if params.numIter != 0 :
            # logging.info('Start training')   
            train(eng, params)

    # Generate images and save 
    # logging.info('Start generating devices')
    evaluate(eng, numImgs=1, params=params)




