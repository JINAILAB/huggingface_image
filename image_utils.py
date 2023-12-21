import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from transformers import AutoImageProcessor
import os



def save_confusion_matrix(y_preds, y_true, labels, save_dir):
    cm = confusion_matrix(y_true, y_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format='.2f', ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.savefig(os.path.join(save_dir, 'cm_normalilze.png'))
    
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("confusion matrix")
    plt.savefig(os.path.join(save_dir, 'cm.png'))



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    
    acc = accuracy_score(labels, preds)
    
    preds_proba = softmax(pred.predictions, axis=1)
    
    roc_auc_macro = roc_auc_score(labels, preds_proba, multi_class='ovr')
    roc_auc_weighted = roc_auc_score(labels, preds_proba, multi_class='ovr', average='weighted')
    
    acc2 = top_k_accuracy_score(labels, preds_proba, k=2)
    
    return {'acc' : acc, 'acc2' : acc2, 'f1' : f1 ,'roc_auc_macro' : roc_auc_macro, 'roc_auc_weighted' : roc_auc_weighted}

