from datasets import load_dataset, DatasetDict
import argparse
import os

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Huggingface build imagedataset", add_help=add_help)

    parser.add_argument("--dataset-dir", default="csv_file/clahe_train_valid", type=str, help="dataset path")
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--dataset_name", default='clahe_hiphf', type=str)
    parser.add_argument("--trainval-dataset", action="store_true")
    return parser



def one_dataset(args):
    dataset = load_dataset("imagefolder", data_files=args.dataset_dir)
    splits = dataset["train"].train_test_split(test_size=args.test_size, stratify_by_column='label')
    
    splits.save_to_disk(args.dataset_name)
    
    
    
def trainval_dataset(args):
    train_dataset = load_dataset("imagefolder", data_files=os.path.join(args.dataset_dir, 'train.zip'))
    valid_dataset = load_dataset("imagefolder", data_files=os.path.join(args.dataset_dir, 'valid.zip'))
    
    
    combined_dataset = DatasetDict({
            "train": train_dataset['train'],
            "test": valid_dataset['train']})
    
    combined_dataset.save_to_disk(args.dataset_name)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.trainval_dataset:
        trainval_dataset(args)
    else:
        one_dataset(args)
    
    

