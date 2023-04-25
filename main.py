import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA#, MaskBaseDataset
from datasets.maskbasedataset import MaskBaseDataset
from utils import get_model_from_sd, test_model_on_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/opt/ml/input/data/train/images/'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('model/'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--random-seed", 
        type=int,
        default=42,
    )
    parser.add_argument(
        "--name",
        default='finetune_cp'
    )
    parser.add_argument(
        "--model-num", 
        type=int,
        default=40,
    )
    parser.add_argument(
        "--epoch", 
        type=int,
        default=20,
    )
    parser.add_argument(## default 0.2 / None값을 넣는다면, 전체 dataset에 대해 evaluation 진행
        "--val-ratio", 
        type=float,
        default=0.2,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    NUM_MODELS = args.model_num
    epoch = args.epoch
    val_ratio=args.val_ratio 
    
    model_name = args.name 

    if args.random_seed != -1 : 
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)


    INDIVIDUAL_MODEL_RESULTS_FILE = f'logs/individual_results_{model_name}_seed{args.random_seed}_num{NUM_MODELS}_epoch{epoch}.jsonl'
    GREEDY_SOUP_LOG_FILE = f'logs/greedy_soup_log_{model_name}_seed{args.random_seed}_num{NUM_MODELS}_epoch{epoch}.txt'

    UNIFORM_SOUP_RESULTS_FILE = 'logs/uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'logs/greedy_soup_results.jsonl'

    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
                )

    model_paths = [os.path.join(args.model_location, f'{model_name}_seed{args.random_seed}_i{i}_epoch{epoch}.pt') for i in range(NUM_MODELS)]
    # Step 2: Evaluate individual models.
    if args.eval_individual_models or args.uniform_soup or args.greedy_soup:
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    if args.eval_individual_models:
        # if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            # os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            # assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            model = get_model_from_sd(state_dict, base_model)
            results = {'model_name' : f'{model_name}_seed{args.random_seed}_i{j}_epoch{epoch}'}
            # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
            # It is called 2p for 2 percent of ImageNet, or 26k images.
            # See utils on how this dataset is handled slightly differently.
            for dataset_cls in [MaskBaseDataset]:  #[ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

                print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')

                # dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                dataset = dataset_cls(batch_size=args.batch_size, val_ratio=val_ratio, random_seed=args.random_seed)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):

            print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = get_model_from_sd(uniform_soup, base_model)

        results = {'model_name' : f'uniform_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

            print(f'Evaluating on {dataset_cls.__name__}.')

            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)
       
        with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    # Step 4: Greedy Soup.
    if args.greedy_soup:
        # if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
        #     os.remove(GREEDY_SOUP_RESULTS_FILE)

        # Sort models by decreasing accuracy on the held-out validation set ImageNet2p
        # (We call the held out-val set ImageNet2p because it is 2 percent of ImageNet train)
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row['MaskBaseDataset']
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]
        
        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
        best_val_acc_so_far = individual_model_val_accs[0][1]
        # held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)
        held_out_val_set = MaskBaseDataset(batch_size=args.batch_size, random_seed=args.random_seed)

        # Now, iterate through all models and consider adding them to the greedy soup.
        for i in range(1, NUM_MODELS):
            print(f'Testing model {i} of {NUM_MODELS}')

            # Get the potential greedy soup, which consists of the greedy soup with the new model added.
            new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            # Run the potential greedy soup on the held-out val set.
            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            with open(GREEDY_SOUP_LOG_FILE, 'a+') as f:
                print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
                f.write(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.\n')
                if held_out_val_accuracy > best_val_acc_so_far:
                    greedy_soup_ingredients.append(sorted_models[i])
                    best_val_acc_so_far = held_out_val_accuracy
                    greedy_soup_params = potential_greedy_soup_params
                    print(f'Adding to soup. New soup is {greedy_soup_ingredients}')
                    f.write(f'Adding to soup. New soup is {greedy_soup_ingredients}\n')


        # Finally, evaluate the greedy soup.
        model = get_model_from_sd(greedy_soup_params, base_model)
        # save final model
        model_path = os.path.join(args.model_location, f'{model_name}_seed{args.random_seed}_epoch{epoch}_greedysoup_num{NUM_MODELS}.pt')
        print('Saving model to', model_path)
        torch.save(model.module.state_dict(), model_path)

        # results = {'model_name' : f'greedy_soup'}
        # for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
        #     print(f'Evaluating on {dataset_cls.__name__}.')
        #     dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
        #     accuracy = test_model_on_dataset(model, dataset)
        #     results[dataset_cls.__name__] = accuracy
        #     print(accuracy)

        # with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
        #     f.write(json.dumps(results) + '\n')

    # Step 5: Plot.
    if args.plot:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_db['OOD'] = 1./5 * (individual_model_db['ImageNetV2'] + 
            individual_model_db['ImageNetR'] + individual_model_db['ImageNetSketch'] + 
            individual_model_db['ObjectNet'] + individual_model_db['ImageNetA'])
        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db['OOD'] = 1./5 * (uniform_soup_db['ImageNetV2'] + 
            uniform_soup_db['ImageNetR'] + uniform_soup_db['ImageNetSketch'] + 
            uniform_soup_db['ObjectNet'] + uniform_soup_db['ImageNetA'])
        greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
        greedy_soup_db['OOD'] = 1./5 * (greedy_soup_db['ImageNetV2'] + 
            greedy_soup_db['ImageNetR'] + greedy_soup_db['ImageNetSketch'] + 
            greedy_soup_db['ObjectNet'] + greedy_soup_db['ImageNetA'])

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            greedy_soup_db['ImageNet'], 
            greedy_soup_db['OOD'], 
            marker='*', 
            color='C4',
            s=400,
            label='Greedy Soup',
            zorder=10
        )

        ax.scatter(
            uniform_soup_db['ImageNet'], 
            uniform_soup_db['OOD'], 
            marker='o', 
            color='C0',
            s=200,
            label='Uniform Soup',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[0], 
            individual_model_db['OOD'].values[0], 
            marker='h', 
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[1:], 
            individual_model_db['OOD'].values[1:], 
            marker='d', 
            color='C2',
            s=130,
            label='Various hyperparameters',
            zorder=10
        )

        ax.set_ylabel('Avg. accuracy on 5 distribution shifts', fontsize=16)
        ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig('figure.png', bbox_inches='tight')