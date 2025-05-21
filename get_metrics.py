import numpy as np

def get_confusion_matrix_old(gt, pred, num_classes):
    """
    手动计算混淆矩阵（计算效率太慢了，亟需优化）
    :param gt: 真实标签(Pytorch tensor, 大小为N×H×W,其中N为batch size,H为图像高度,W为图像宽度)
    :param pred: 预测标签(Pytorch tensor,大小为N×H×W,其中N为batch size,H为图像高度,W为图像宽度)
    :param num_classes: 类别数
    :return: 混淆矩阵
    """

    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for sample_idx in range(gt.shape[0]):
        # 遍历每一个样本
        for i in range(gt.shape[1]):
            for j in range(gt.shape[2]):
                # 遍历每一个像素
                true_label = gt[sample_idx, i, j]
                pred_label = pred[sample_idx, i, j]
                # 更新混淆矩阵
                confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix

def get_confusion_matrix(gt, pred, num_classes):
    """
    手动计算混淆矩阵
    :param gt: 真实标签(Pytorch tensor, 大小为N×H×W,其中N为batch size,H为图像高度,W为图像宽度)
    :param pred: 预测标签(Pytorch tensor, 大小为N×H×W,其中N为batch size,H为图像高度,W为图像宽度)
    :param num_classes: 类别数
    :return: 混淆矩阵
    """

    gt_flat = gt.cpu().numpy().ravel().astype(np.int32)
    pred_flat = pred.cpu().numpy().ravel().astype(np.int32)
    confusion_matrix = np.bincount(num_classes * gt_flat + pred_flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    return confusion_matrix

def get_overall_accuracy(confusion_matrix):
    """
    计算整体精度
    :param confusion_matrix: 混淆矩阵
    :return: 整体精度
    """
    # 计算总样本数
    total_samples = np.sum(confusion_matrix)
    # 计算正确预测的样本数
    correct_predictions = np.sum(np.diag(confusion_matrix))
    # 计算整体精度
    overall_accuracy = correct_predictions / total_samples

    return overall_accuracy

def get_precision(confusion_matrix):
    """
    计算精确率
    :param confusion_matrix: 混淆矩阵
    :return: 精确率
    """
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    precision = np.nan_to_num(precision)
    
    return precision

def get_recall(confusion_matrix):
    """
    计算召回率
    :param confusion_matrix: 混淆矩阵
    :return: 召回率    
    """
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    recall = np.nan_to_num(recall)
    
    return recall

def get_f1_score(confusion_matrix):
    """
    计算F1-score        
    :param confusion_matrix: 混淆矩阵
    :return: F1-score
    """
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    f1_score = 2 * precision * recall / (precision + recall)
    f1_score = np.nan_to_num(f1_score)
    
    return f1_score

def get_iou_score(confusion_matrix):
    """
    计算IoU-score(交并比)
    :param confusion_matrix: 混淆矩阵
    :return: IoU-score
    """
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    iou_score = intersection / (union + 1e-7)
    iou_score = np.nan_to_num(iou_score)

    return iou_score

def get_miou_score(confusion_matrix):
    """
    计算mIoU-score(平均交并比)
    :param confusion_matrix: 混淆矩阵
    :return: mIoU-score
    """
    iou_score = get_iou_score(confusion_matrix)
    miou_score = np.mean(iou_score)

    return miou_score





