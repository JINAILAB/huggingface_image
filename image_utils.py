import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score, precision_score, recall_score, roc_curve, auc
import os
import umap
import torch
import numpy as np
from transformers import Pipeline, AutoModelForImageClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(y_preds, y_true, labels, save_dir, top_n=20):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_preds, labels=labels)
    
    # Calculate the support (total true instances per class)
    support = cm.sum(axis=1)
    
    # Get indices of labels sorted by support in descending order
    sorted_indices = np.argsort(support)[::-1]
    
    # Select top N labels
    top_n = min(top_n, len(labels))
    top_indices = sorted_indices[:top_n]
    
    # Map indices to labels
    top_labels = [labels[i] for i in top_indices]
    
    # Extract submatrix for top labels
    cm_top = cm[np.ix_(top_indices, top_indices)]
    
    # Normalize the confusion matrix
    cm_norm = cm_top.astype('float') / cm_top.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs with zeros if division by zero occurs
    
    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=top_labels)
    disp.plot(cmap="Blues", values_format='.2f', ax=ax, xticks_rotation=90)
    plt.title(f"Top {top_n} Normalized Confusion Matrix", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cm_normalized_top.png'), bbox_inches='tight')
    plt.close()
    
    # Plot non-normalized confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_top, display_labels=top_labels)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=90)
    plt.title(f"Top {top_n} Confusion Matrix", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cm_top.png'), bbox_inches='tight')
    plt.close()



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    
    acc = accuracy_score(labels, preds)
    
    preds_proba = softmax(pred.predictions, axis=1)
    
    roc_auc_micro = roc_auc_score(labels, preds_proba, multi_class='ovr', average='micro')
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    
    acc2 = top_k_accuracy_score(labels, preds_proba, k=2)
    
    return {'acc' : acc, 'acc2' : acc2, 'f1' : f1 ,'roc_auc_micro' : roc_auc_micro, 'precision' : precision, 'recall' : recall}


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
    
    
class CustomImagePipeline(Pipeline):
    def __init__(self, model, id2label, label2id, **kwargs):
        
        if model and isinstance(model, str):
        # 모델을 미리 학습된 모델로부터 불러오기
            model = AutoModelForImageClassification.from_pretrained(
                model,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes = True, # 이미 fine-tuning된 체크포인트를 다시 fine-tuning하려는 경우, 크기 불일치 무시
            )
        else:
            model = model
        super().__init__(model=model, **kwargs)
        # self.image_processor = image_processor
        # 모델 설정에서 id2label 매핑 사용
        self.id2label = id2label
        

    def _sanitize_parameters(self, **kwargs):
        # 나이와 성별 처리 더 이상 필요하지 않음
        return {}, {}, {}

    def preprocess(self, inputs):
        if isinstance(inputs, torch.Tensor):
            # 텐서의 차원 확인
            if inputs.dim() == 3:
                # 배치 차원이 없으면 추가
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() == 4:
                # 이미 배치 차원이 있으면 그대로
                pass
            else:
                raise ValueError("예상치 못한 입력 텐서 모양입니다. 3 또는 4차원이어야 합니다.")
        else:
            # 다른 유형의 입력일 경우, image_processor로 처리
            inputs = self.image_processor(images=inputs, return_tensors="pt").pixel_values
        
        # 디바이스 설정 (CUDA 사용 가능하면 사용)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        return {
            "pixel_values": inputs.to(device)  # 입력값을 해당 디바이스로 전송
        }

    def _forward(self, model_inputs):
        # 모델에 입력값 전달하여 출력 받기
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs['logits']  # 모델의 로짓 값 출력
        probabilities = torch.softmax(logits, dim=-1)  # 소프트맥스 함수 적용하여 확률값으로 변환
        probabilities = probabilities.squeeze()  # 배치 차원이 있으면 제거

        # 확률값을 NumPy 배열로 변환
        probs = probabilities.detach().cpu().numpy()
        

        # 라벨과 점수를 포함한 딕셔너리 리스트 생성
        result = []
        for idx, score in enumerate(probs):
            label = self.id2label.get(idx, f"Class {idx}")  # 라벨이 없을 경우 기본값 설정
            result.append({'label': label, 'score': float(score)})


        return result
    
def run_model(model, dataset, id2label):
    """
    이 함수는 주어진 모델과 데이터셋을 사용하여 이미지 분류를 수행한 후, 결과를 DataFrame으로 반환합니다.

    Args:
        model: 사전 훈련된 모델 객체 또는 모델이 저장된 경로 (model_dir).
        dataset: 이미지 데이터셋으로, 각 항목은 'pixel_values' (이미지 데이터)와 'label' (클래스 레이블)을 포함합니다.
        id2label: 클래스 ID를 해당 레이블로 매핑하는 딕셔너리.

    Returns:
        pd.DataFrame: 분류 결과가 담긴 DataFrame으로, 각 열은 클래스 레이블, 각 행은 이미지에 대한 예측 결과를 포함합니다.
                      'label'과 'encoded_label' 열은 실제 레이블 정보를 포함합니다.
    """
    
    # 데이터셋에서 이미지 픽셀 값을 추출
    pixel_values = [image['pixel_values'] for image in dataset]
    
    # 레이블 변환 (id -> label)
    label2id = {v: k for k, v in id2label.items()}
    
    # 커스텀 분류 파이프라인 초기화
    classifier = CustomImagePipeline(model, id2label, label2id)
    
    # 분류 작업 수행
    data = classifier(pixel_values)  # 테스트 데이터셋에서 이미지 분류 수행
    
    # id2label 순서대로 레이블 정렬
    sorted_labels = [id2label[idx] for idx in range(len(id2label))]

    # 모델 출력으로부터 DataFrame 생성 (레이블 순서 보장)
    df = pd.DataFrame(
        [{item['label']: item['score'] for item in sublist} for sublist in data], 
        columns=sorted_labels
    )
    
    # 원래 레이블과 인코딩된 레이블 추가
    df['label'] = dataset['label']
    df['encoded_label'] = df['label']
    
    # 레이블을 id2label을 사용해 매핑
    df['label'] = df['label'].map(id2label)
    
    return df



def get_metric_from_df(df):
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])

    prob_df = df.iloc[:, :-2]
    ordered_prob_df = prob_df[le.classes_]
    predicted_probs = ordered_prob_df.values

    # Predicted labels are the columns with highest probability
    predictions = df.iloc[:, :-2].idxmax(axis=1)
    encoded_predictions = le.transform(predictions)

    # Calculate top-2 accuracy
    top2_predictions = df.iloc[:, :-2].apply(lambda x: x.nlargest(2).index.tolist(), axis=1)
    top2_accuracy = np.mean([label in pred for label, pred in zip(df['label'], top2_predictions)])

    # Calculate metrics
    conf_matrix = confusion_matrix(df['encoded_label'], encoded_predictions)
    accuracy = accuracy_score(df['encoded_label'], encoded_predictions)
    precision = precision_score(df['encoded_label'], encoded_predictions, average='micro')
    recall = recall_score(df['encoded_label'], encoded_predictions, average='micro')
    f1 = f1_score(df['encoded_label'], encoded_predictions, average='micro')
    roc_auc = roc_auc_score(df['encoded_label'], predicted_probs, average='micro', multi_class='ovr')


    Total = conf_matrix.sum()
    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    TN = Total - (TP + FP + FN)
    specificity_per_class = TN / (TN + FP)
    specificity = specificity_per_class.mean()

    return {
        'accuracy': accuracy,
        'top2_accuracy': top2_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'specificity': specificity
        }   
    

def plot_rocauc_from_df(df, label2id, n=10, plot_top=True, save_dir=None, plot_name='ROC curve for the test set', target_order=None):
    """
    데이터프레임에서 ROC 곡선을 그립니다. 각 클래스의 ROC AUC를 기준으로 상위 또는 하위 N개의 레이블을 선택합니다.

    매개변수:
    - df: pandas DataFrame, 실제 레이블과 예측 확률을 포함합니다.
    - label2id: dict, 레이블과 해당 ID의 매핑.
    - n: int, 플롯에 포함할 레이블의 수.
    - plot_top: bool, True이면 상위 N개의 레이블을 플롯하고, False이면 하위 N개의 레이블을 플롯합니다.
    - save_dir: str, 플롯을 저장할 디렉토리 경로. None이면 플롯을 화면에 표시합니다.
    - plot_name: str, 플롯의 제목.
    - target_order: list, 레이블의 표시 순서를 지정합니다.
    """
    # 기본 target_order 정의
    label_list = list(label2id.keys())
    
    # target_order가 None이면 label_list 사용
    if target_order is None:
        target_order = label_list
    
    roc_aucs = []
    lines = []
    labels = []

    # 각 레이블에 대한 ROC AUC 계산
    for label in label_list:
        true_binary_label = (df['label'] == label).astype(int)
        prediction_prob = df[label]
        fpr, tpr, _ = roc_curve(true_binary_label, prediction_prob)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append((label, roc_auc))

    # ROC AUC를 기준으로 레이블 정렬
    roc_aucs.sort(key=lambda x: x[1], reverse=plot_top)
    
    # N개의 레이블 선택
    n = min(n, len(roc_aucs))
    selected_labels = [label for label, _ in roc_aucs[:n]]
    selected_roc_aucs = roc_aucs[:n]

    # 플롯 설정
    plt.figure(figsize=(14, 12))

    # 선택된 레이블에 대한 ROC 곡선 계산 및 플롯
    for label, roc_auc in selected_roc_aucs:
        true_binary_label = (df['label'] == label).astype(int)
        prediction_prob = df[label]
        fpr, tpr, _ = roc_curve(true_binary_label, prediction_prob)
        
        # ROC 곡선 플롯 및 라인과 레이블 저장
        line, = plt.plot(fpr, tpr, lw=2, label=f'{label} (area = {roc_auc:.3f})')
        lines.append(line)
        labels.append(label)

    # 대각선 선 그리기
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    # 플롯 마무리
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title(plot_name, fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)
    plt.gcf().set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  # 테두리 색상 설정
        spine.set_linewidth(2)

    # 범례 설정
    legend = plt.legend(lines, [f'{label} (area = {dict(selected_roc_aucs)[label]:.3f})' for label in labels], loc="lower right", fontsize=20)
    legend.get_frame().set_facecolor('white')  # 범례 배경을 흰색으로 설정
    legend.get_frame().set_edgecolor('black')

    # 플롯 저장 또는 표시
    if save_dir:
        # 저장할 파일명 설정
        if plot_top:
            filename_modifier = 'top'
        else:
            filename_modifier = 'bottom'
        plt.savefig(os.path.join(save_dir, f'roc_curve_{filename_modifier}_{n}.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return selected_roc_aucs


def plot_cm_from_df(df, n=20, plot_top=True, save_dir=None):
    """
    데이터프레임에서 혼동행렬을 그립니다. 각 클래스의 정확도를 기준으로 상위 또는 하위 N개의 레이블을 선택합니다.

    매개변수:
    - df: 실제 레이블과 예측 확률을 포함한 pandas DataFrame.
    - n: int, 플롯에 포함할 레이블의 수.
    - plot_top: bool, True이면 상위 N개의 레이블을 플롯하고, False이면 하위 N개의 레이블을 플롯합니다.
    - save_dir: str, 플롯을 저장할 디렉토리 경로. None이면 플롯을 화면에 표시합니다.
    """
    # Label Encoder 초기화 및 실제 레이블 인코딩
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])
    class_labels = le.classes_  # 클래스 레이블 목록

    # 예측 레이블은 가장 높은 확률을 가진 열입니다
    # 마지막 두 열('label'과 'encoded_label')은 제외합니다
    predictions = df.iloc[:, :-2].idxmax(axis=1)
    encoded_predictions = le.transform(predictions)

    # 전체 혼동행렬 계산
    conf_matrix = confusion_matrix(
        df['encoded_label'], encoded_predictions, labels=np.arange(len(class_labels))
    )

    # 클래스별 지원 수 계산 (각 클래스의 실제 인스턴스 수)
    support = conf_matrix.sum(axis=1)

    # 클래스별 정확도 계산
    per_class_accuracy = np.divide(
        conf_matrix.diagonal(), support, out=np.zeros_like(support, dtype=float), where=support != 0
    )

    # 정확도를 기준으로 레이블 정렬
    sorted_indices = np.argsort(per_class_accuracy)
    if plot_top:
        # 상위 N개의 레이블을 위해 내림차순으로 정렬
        sorted_indices = sorted_indices[::-1]
        title_modifier = "Top"
        filename_modifier = "top"
    else:
        # 하위 N개의 레이블은 오름차순 유지
        title_modifier = "Bottom"
        filename_modifier = "bottom"

    # N개의 인덱스 선택
    n = min(n, len(class_labels))
    selected_indices = sorted_indices[:n]

    # 인덱스를 클래스 레이블로 매핑하여 표시
    selected_labels = [class_labels[i] for i in selected_indices]

    # 선택한 레이블에 대한 혼동행렬 추출
    conf_matrix_selected = conf_matrix[np.ix_(selected_indices, selected_indices)]

    # 혼동행렬 정규화
    conf_matrix_norm = conf_matrix_selected.astype('float') / conf_matrix_selected.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # 0으로 나누는 경우 처리

    # 정규화된 혼동행렬 플롯
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_norm, display_labels=selected_labels)
    disp.plot(cmap="Blues", values_format='.2f', ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"{title_modifier} {n} normalized confusion matrix", fontsize=16)
    plt.grid(False)
    plt.tight_layout()

    # 플롯 저장 또는 표시
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f'cm_normalized_{filename_modifier}.png'), bbox_inches='tight'
        )
    else:
        plt.show()
    plt.close()

    # 비정규화된 혼동행렬 플롯
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_selected, display_labels=selected_labels)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"{title_modifier} {n} confusion matrix", fontsize=16)
    plt.grid(False)
    plt.tight_layout()

    # 플롯 저장 또는 표시
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'cm_{filename_modifier}.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()
