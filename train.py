from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
import datetime
from image_utils import save_confusion_matrix, compute_metrics, create_hook, get_embedding_layer, save_umap, check_embedding
import argparse
from datasets import load_from_disk
from data_presets import load_dataset
import os
import numpy as np
import wandb
import logging
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from gradcam_utils import category_name_to_index, save_gradcam

 

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="Huggingface Classification Training", add_help=add_help)
    
    parser.add_argument("--project-name", default='small_hipjoint', type=str,
                        help='This is the project name of wandb and save folder name')
    parser.add_argument("--dataset-dir", default="small_hiphf", type=str,
                        help="dataset path")
    parser.add_argument("--model", default="microsoft/resnet-50", type=str, help="we only support hugginface pretrained model. check here https://huggingface.co/models?pipeline_tag=image-classification&sort=trending. \
                        microsoft/resnet-50, microsoft/swin-tiny-patch4-window7-224, facebook/convnext-large-224")
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("--epochs", "-e", default=80, type=int, metavar="N")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-03,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0, type=float, help="mixup alpha (recommend: 0.2)")
    parser.add_argument("--cutmix-alpha", default=0, type=float, help="cutmix alpha (recommend: 1.0)")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: cosineannealinglr)")
    parser.add_argument("--lr-warmup-epochs", default=3, type=int,
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--custom-processor", action="store_true",
                        help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--early-stopping-epoch", default=10, type=int, help='early-stopping-epoch')
    parser.add_argument("--fp16", action="store_true", help='fp16, operate like amp.')
    parser.add_argument("--report", default='wandb', type=str, help='support azure_ml, clearml, codecarbon, comet_ml, dagshub, dvclive, flyte, mlflow, neptune, tensorboard, and wandb. recommend : wandb and tensorboard')
    parser.add_argument(
        "--resize-size", default=None, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--crop-pct", default=None, type=int, help="resize and crop for train. and centercrop for validate resizesize * croppct.  (recommend = 0.875)"
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--torch-compile", default=False, type=bool, help="torch-compile make your model fast."
    )
    parser.add_argument("--auto-augment", default='ta_wide', type=str, help="auto augment policy (default: ta_wide)")
    parser.add_argument("--gradient-accumulation-steps", '-g', default=8, type=int, help="help batchnormalization for small batch.")
    parser.add_argument("--gradcam", action="store_true", help='visualize gradcam for validset')
    parser.add_argument("--umap", action="store_true", help='visualize umap for validset')
    
    return parser



def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def main(args):
    
    logger = logging.getLogger(__name__)
    # 현재 날짜와 시간을 얻습니다.
    # print(args)
    # logger.debug(args)
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%d_%H%M")

    
    model_name = args.model.split("/")[-1]
    output_dir = os.path.join(args.project_name, model_name + '_' + now)
    if args.report == 'wandb':
        wandb.init(project=args.project_name, name=model_name + '_' + now)
    

    dataset = load_from_disk(args.dataset_dir)
    
    
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    image_processor  = AutoImageProcessor.from_pretrained(args.model)
    
    if args.resize_size is None:
        if 'height' in image_processor.size:
            args.val_resize_size = image_processor.size['height']
        if 'shortest_edge' in image_processor.size:
            args.val_resize_size = image_processor.size['shortest_edge']
    if args.crop_pct is None:
        if hasattr(image_processor, 'crop_pct'):
            args.train_crop_size = int(image_processor.crop_pct * args.val_resize_size)
            args.val_crop_size = int(image_processor.crop_pct * args.val_resize_size)
        else:
            args.train_crop_size = args.val_resize_size
            args.val_crop_size = args.val_resize_size
    
    train_ds, valid_ds, _, valid_transform = load_dataset(dataset, args)

    
    model = AutoModelForImageClassification.from_pretrained(
        args.model, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    ).to('cuda')

    
    trainargs = TrainingArguments(
        output_dir = output_dir,
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.lr,
        lr_scheduler_type='constant_with_warmup',
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=300,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc_macro",
        push_to_hub=False,
        report_to=args.report,
        logging_dir= output_dir,
        fp16=args.fp16,
        logging_steps=10,
        label_smoothing_factor=args.label_smoothing,
    )
    
    
    
    trainer_callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_epoch)] if args.early_stopping_epoch > 0 else None
    
    trainer = Trainer(
        model,
        trainargs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks = trainer_callbacks
    )
    
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'best.hf'))
    if args.report == 'wandb':
        wandb.finish(exit_code=0)
    
    if args.umap:
        hook, embedding_outputs = create_hook()
        embedding = get_embedding_layer(model)
        hook_handle = embedding.register_forward_hook(hook)
    
    preds_output= trainer.predict(valid_ds)
    y_preds = np.argmax(preds_output.predictions, axis=-1)
    y_valid = np.array(valid_ds['label'])
    save_confusion_matrix(y_preds, y_valid, labels, output_dir)
    
    if args.umap:
        all_embeddings = check_embedding(model, embedding_outputs)
        save_umap(all_embeddings, y_valid, labels, output_dir)
        hook_handle.remove()
    
    if args.gradcam:
        save_gradcam(model, id2label, output_dir, valid_ds, valid_transform, args.val_crop_size)
        
        

    
     
    
    
    



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)