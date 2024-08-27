import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm

def bbox_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def calculate_ap(recalls, precisions):
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

def evaluate_detections(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    true_positives = {i: [] for i in range(num_classes)}
    false_positives = {i: [] for i in range(num_classes)}
    false_negatives = {i: 0 for i in range(num_classes)}
    
    for detections, annotations in zip(all_detections, all_annotations):
        for class_id in range(num_classes):
            class_detections = detections[detections[:, -1] == class_id]
            class_annotations = annotations[annotations[:, -1] == class_id]
            
            detected = torch.zeros(len(class_annotations))
            
            for detection in class_detections:
                if len(class_annotations) == 0:
                    false_positives[class_id].append(1)
                    true_positives[class_id].append(0)
                    continue

                ious = bbox_iou(detection[:4].unsqueeze(0), class_annotations[:, :4])
                best_iou, best_annotation = ious.max(0)

                if best_iou >= iou_threshold:
                    if not detected[best_annotation]:
                        true_positives[class_id].append(1)
                        false_positives[class_id].append(0)
                        detected[best_annotation] = 1
                    else:
                        true_positives[class_id].append(0)
                        false_positives[class_id].append(1)
                else:
                    true_positives[class_id].append(0)
                    false_positives[class_id].append(1)
            
            false_negatives[class_id] += torch.sum(1 - detected).item()
    
    # Calculate metrics
    ap = {}
    precision = {}
    recall = {}
    f1_score = {}
    
    for class_id in range(num_classes):
        tp = torch.tensor(true_positives[class_id])
        fp = torch.tensor(false_positives[class_id])
        fn = false_negatives[class_id]
        
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / (tp_cumsum + fn + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        ap[class_id] = calculate_ap(recalls.numpy(), precisions.numpy())
        precision[class_id] = precisions[-1].item()
        recall[class_id] = recalls[-1].item()
        f1_score[class_id] = 2 * (precision[class_id] * recall[class_id]) / (precision[class_id] + recall[class_id] + 1e-6)
    
    mAP = np.mean(list(ap.values()))
    
    return {
        'mAP': mAP,
        'AP': ap,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score
    }

def calculate_metrics(net, val_loader, num_classes, device):
    net.eval()
    all_detections = []
    all_annotations = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = net(images)
            
            # Process outputs and targets
            all_detections.extend(outputs)
            all_annotations.extend(targets)
    
    metrics = evaluate_detections(all_detections, all_annotations, num_classes)
    return metrics