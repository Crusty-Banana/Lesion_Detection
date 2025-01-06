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