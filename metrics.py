import numpy as np
from collections import defaultdict
import torch

def bbox_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
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
    true_positives = defaultdict(list)
    false_positives = defaultdict(list)
    false_negatives = defaultdict(int)
    
    for image_id in all_annotations.keys():
        detections = all_detections[image_id]
        annotations = all_annotations[image_id]
        
        for class_id in range(num_classes):
            class_detections = detections[detections[:, -1] == class_id]
            class_annotations = annotations[annotations[:, -1] == class_id]
            
            detected = np.zeros(len(class_annotations))
            
            for detection in class_detections:
                best_iou = 0
                best_annotation = -1
                
                for idx, annotation in enumerate(class_annotations):
                    iou = bbox_iou(detection[:4], annotation[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_annotation = idx
                
                if best_annotation >= 0 and best_iou >= iou_threshold:
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
            
            false_negatives[class_id] += np.sum(1 - detected)
    
    # Calculate metrics
    ap = defaultdict(float)
    precision = defaultdict(float)
    recall = defaultdict(float)
    f1_score = defaultdict(float)
    
    for class_id in range(num_classes):
        tp = np.array(true_positives[class_id])
        fp = np.array(false_positives[class_id])
        fn = false_negatives[class_id]
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (tp_cumsum + fn + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        ap[class_id] = calculate_ap(recalls, precisions)
        precision[class_id] = np.mean(precisions)
        recall[class_id] = np.mean(recalls)
        f1_score[class_id] = 2 * (precision[class_id] * recall[class_id]) / (precision[class_id] + recall[class_id] + 1e-6)
    
    mAP = np.mean(list(ap.values()))
    
    return {
        'mAP': mAP,
        'AP': dict(ap),
        'Precision': dict(precision),
        'Recall': dict(recall),
        'F1-score': dict(f1_score)
    }

def calculate_metrics(net, val_loader, num_classes, device):
    net.eval()
    all_detections = defaultdict(list)
    all_annotations = defaultdict(list)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            outputs = net(images)
            
            # Process outputs and targets
            for batch_idx, (output, target) in enumerate(zip(outputs, targets)):
                image_id = i * val_loader.batch_size + batch_idx
                all_detections[image_id] = output.cpu().numpy()
                all_annotations[image_id] = target.cpu().numpy()
    
    metrics = evaluate_detections(all_detections, all_annotations, num_classes)
    return metrics