import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             roc_auc_score,
                             f1_score)

def calc_IoU(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def calc_mAP(predictions, targets, IoU_threshold=0.5):
    precision_sum = 0
    for prediction, target in zip(predictions, targets):
        boxes = target["boxes"]
        labels = target["labels"]
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]

        correct_pred = 0

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            for box, label in zip(boxes, labels):
                if pred_label == label:
                    IoU = calc_IoU(pred_box, box)
                    if IoU >= IoU_threshold:
                        correct_pred += 1
                        break

        if (len(pred_labels) == 0 and correct_pred == 0):
            # handle negative sample
            precision_sum += 1
        elif (len(pred_labels) != 0):
            precision = correct_pred / len(pred_labels)
            precision_sum += precision

    mAP = precision_sum / len(predictions)
    return mAP

def show_object_detection_image(image, boxes, labels, output_dir):
    fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.imshow(image, cmap='gray')
    
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        print(box)
        rect = patches.Rectangle(
            (x_min, y_min),  
            x_max - x_min,  
            y_max - y_min,  
            linewidth=4,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            x_min, y_min - 5, label,
            color='red', fontsize=12,
            backgroundcolor='none'
        )
    
    plt.axis('off')  
    plt.show()
    plt.savefig(output_dir, bbox_inches='tight')
    print(f"Image saved to {output_dir}")
    plt.close()

def calc_classification_metrics(targs, preds, probs):
    cm = confusion_matrix(targs, preds)
    acc = accuracy_score(targs, preds)
    auc_roc = roc_auc_score(targs, probs)
    f1 = f1_score(targs, preds)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp+fn)
    spe = tn / (tn+fp)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Area under the ROC Curve: {auc_roc:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Sensitivity: {sen:.2f}")
    print(f"Specificity: {spe:.2f}")
