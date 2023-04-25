import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.maskbasedataset import TestDataset, MaskBaseDataset
import clip
import math


def load_model(model_path, device, model):
    base_model, preprocess = clip.load(model, 'cuda', jit=False)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model = get_model_from_sd(state_dict, base_model)
    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    
    model_path = os.path.join(args.model_location, args.model_name) 
    model = load_model(model_path, device, args.model).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    if args.weighted_ensemble  :
        model_path2 = os.path.join(args.model_location, args.weighted_ensemble) 
        model2 = load_model(model_path2, device, args.model).to(device)
        model2.eval()
        print('Weighted average Ensemble')
    if args.soft_voting: 
        model_path2 = os.path.join(args.model_location, args.soft_voting) 
        model2 = load_model(model_path2, device, args.model).to(device)
        model2.eval()
        print('Soft Voting')

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)

            if args.weighted_ensemble :
                pred2 = model2(images)
                for i in range(len(pred)): 
                    
                    age1_percent = 1. + pred2[i][0] / pred2[i].sum()
                    age2_percent = 1. + pred2[i][1] / pred2[i].sum()
                    age3_percent = 1. + pred2[i][2] / pred2[i].sum()
                    
                    age_percent = torch.tensor([age1_percent, age2_percent, age3_percent,
                        age1_percent, age2_percent, age3_percent,
                        age1_percent, age2_percent, age3_percent,
                        age1_percent, age2_percent, age3_percent,
                        age1_percent, age2_percent, age3_percent,
                        age1_percent, age2_percent, age3_percent] , device='cuda')

                    pred[i] *= age_percent

            if args.soft_voting: 
                pred2  = model2(images)
                for i in range(len(pred)): 
                    pred[i] = (pred[i] - pred[i].min()) / (pred[i].max() - pred[i].min())
                    pred2[i] = (pred2[i] - pred2[i].min()) / (pred2[i].max() - pred2[i].min())
                    pred[i] += pred2[i]

            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    if args.weighted_ensemble : 
        save_path = os.path.join(output_dir, f'output_{args.model_name[:-3]}_WE_{args.weighted_ensemble[:-3]}.csv')
    elif args.soft_voting : 
        save_path = os.path.join(output_dir, f'output_{args.model_name[:-3]}_SV_{args.soft_voting[:-3]}.csv')
    else : 
        save_path = os.path.join(output_dir, f'output_{args.model_name[:-3]}.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

def get_model_from_sd(state_dict, base_model):
    
    if not 'classification_head.weight' in state_dict : 
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
            state_dict = new_state_dict

    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    return torch.nn.DataParallel(model,  device_ids=devices)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits

if __name__ == '__main__':#####################################resize default값 고치기!!!!!!!!!!!!!!#############################
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='ViT-B/32', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model-location', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument("--weighted-ensemble", default=None, help='Filename of age model to apply weighted average ensemble')
    parser.add_argument("--soft-voting", default=None, help='Write file name of model to apply soft voting ensemble')
    parser.add_argument("--model-name", default=None, help='Filename of model')

    args = parser.parse_args()

    assert not (args.weighted_ensemble and args.soft_voting), "Activate one either Weighted Average Ensemble or Soft Voting "

    data_dir = args.data_dir
    model_dir = args.model_location
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
