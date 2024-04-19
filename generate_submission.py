from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from classification_evaluation import get_label
import csv
import numpy as np

def classifier(model, data_loader, device):
    model.eval()
    classifications = []
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        answer = get_label(model, model_input, device)
        classifications.extend(answer.cpu().numpy().tolist())
        
    return classifications


def get_file_names(test_csv):
    file_names = []
    with open(test_csv, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Extract the file name from the first column (assuming file names are in the first column)
            file_name = row[0].split('/')[-1]  # Split by '/' and get the last part (file name)
            
            # Append the file name to the list
            file_names.append(file_name)
    return file_names


def save_classifications_to_csv(classifications, file_names, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'label'])  # Header row

        for classification, id in zip(classifications, file_names):
            csv_writer.writerow([id, classification])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=100, input_channels=3, nr_logistic_mix=10)
    model = model.to(device)
    model.load_state_dict(torch.load('models/pcnn_cpen455_load_model_49.pth'))
    model.eval()

    output_csv = 'submission.csv'
    test_csv = 'data/test.csv'

    classifications = classifier(model, dataloader, device)
    file_names = get_file_names(test_csv)

    save_classifications_to_csv(classifications, file_names, output_csv)