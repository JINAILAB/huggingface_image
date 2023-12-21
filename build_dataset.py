from datasets import load_dataset, DatasetDict
import argparse


def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Huggingface build imagedataset", add_help=add_help)

    parser.add_argument("--dataset-dir", default="./image/image.zip", type=str, help="dataset path")
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--trainval-dataset", default=False)
    return parser
    



def one_dataset(args):
    dataset = load_dataset("imagefolder", data_files=args.dataset_dir)
    splits = dataset["train"].train_test_split(test_size=args.test_size, stratify_by_column='label')
    
    splits.save_to_disk('./image_data.hf')
    
    
    
def trainval_dataset(args):
    train_dataset = load_dataset("imagefolder", data_files='image/train.zip')
    valid_dataset = load_dataset("imagefolder", data_files='image/valid.zip')
    
    
    combined_dataset = DatasetDict({
            "train": train_dataset,
            "test": valid_dataset})
    
    combined_dataset.save_to_disk('./image_data.hf')

    
    
    

