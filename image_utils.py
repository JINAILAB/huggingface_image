import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from transformers import AutoImageProcessor
import os
import umap
import torch
import numpy as np

def save_confusion_matrix(y_preds, y_true, labels, save_dir):
    cm = confusion_matrix(y_true, y_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format='.2f', ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.savefig(os.path.join(save_dir, 'cm_normalilze.png'))
    
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("confusion matrix")
    plt.savefig(os.path.join(save_dir, 'cm.png'))



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1= f1_score(labels, preds, average='weighted')
    
    acc = accuracy_score(labels, preds)
    
    preds_proba = softmax(pred.predictions, axis=1)
    
    roc_auc_macro = roc_auc_score(labels, preds_proba, multi_class='ovr')
    roc_auc_weighted = roc_auc_score(labels, preds_proba, multi_class='ovr', average='weighted')
    
    acc2 = top_k_accuracy_score(labels, preds_proba, k=2)
    
    return {'acc' : acc, 'acc2' : acc2, 'f1' : f1 ,'roc_auc_macro' : roc_auc_macro, 'roc_auc_weighted' : roc_auc_weighted}


def create_hook():
    outputs = []  # 이 함수의 지역 변수로 outputs를 정의

    def hook(module, input, output):
        outputs.append(output)
        return output  # 필요에 따라 output을 반환할 수 있음

    return hook, outputs 

def get_embedding_layer(model): 
    """
    모델의 종류에 따라 target_layer를 추출합니다. dff_layer_dff와 target_layer_gradcam을 return
    """
    # resnet
    if hasattr(model, 'resnet'):
        return model.classifier[0]
    # ConvNeXt
    elif hasattr(model, 'convnext'):
        return model.convnext.layernorm
    # swin
    elif hasattr(model, 'swin'):
        return model.swin.pooler
    elif hasattr(model, 'swinv2'):
        return model.swinv2.pooler
    # vit
    elif hasattr(model, 'vit'):
        return model.vit.layernorm
    # regnet
    # elif hasattr(model, 'regnet'):
    #     return model.regnet.encoder.stages[-1], model.regnet.encoder.stages[-1]
    # # cvt
    # elif hasattr(model, 'cvt'):
    #     return model.cvt.encoder.stages[-1].layers[-1], model.cvt.encoder.stages[-1].layers[-2]
    # 기타 모델에 대해서는 지원 X
    else:
        return None
    
    
def check_embedding(model, embedding_outputs):
    # embedding_outputs은 리스트이기 때문에 torch.cat으로 tensor로 만들어줌
    all_embeddings = torch.cat(embedding_outputs, dim=0).cpu().detach().numpy()
    # resnset
    if hasattr(model, 'resnet'):
        return all_embeddings
    # ConvNeXt
    elif hasattr(model, 'convnext'):
        return all_embeddings
    # swin
    elif hasattr(model, 'swin'):
        return all_embeddings.squeeze()
    # swinv2
    elif hasattr(model, 'swinv2'):
        return all_embeddings.squeeze()
    # vit
    elif hasattr(model, 'vit'):
        return all_embeddings.squeeze()
    
    
def save_umap(all_embeddings, y_true, labels, save_dir):
    # umap reducer 생성
    reducer = umap.UMAP(n_neighbors=15, metric='correlation', min_dist=0.01)
    embedding = reducer.fit_transform(all_embeddings)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap='nipy_spectral', s=8)
    plt.gca().set_aspect('equal', 'datalim')

    colorbar = plt.colorbar(boundaries=np.arange(len(labels)+1)-0.5)
    colorbar.set_ticks(np.arange(len(labels)))
    colorbar.set_ticklabels(labels)

    plt.title('UMAP projection of the valid dataset', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'umap.png'))