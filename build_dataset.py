from datasets import load_dataset, DatasetDict
import argparse
import os

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Huggingface build imagedataset", add_help=add_help)

    parser.add_argument("--dataset-dir", default="csv_file/clahe_train_valid", type=str, help="dataset path")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--dataset-name", default='clahe_hiphf', type=str)
    parser.add_argument("--dataset-type", type=str, default='one_dataset')
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
    
def csv_dataset(args):
    train_dataset = load_dataset('csv', data_files=os.path.join(args.dataset_dir, 'train.csv'))
    valid_dataset = load_dataset('csv', data_files=os.path.join(args.dataset_dir, 'valid.csv'))
    
    combined_dataset = DatasetDict({
            "train": train_dataset['train'],
            "test": valid_dataset['train']})
    combined_dataset.save_to_disk(args.dataset_name)
    
    

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.dataset_type == 'trainval_dataset':
        trainval_dataset(args)
    elif args.dataset_type == 'one_dataset':
        one_dataset(args)
    elif args.dataset_type == 'csv_dataset':
        csv_dataset(args)
    
    

