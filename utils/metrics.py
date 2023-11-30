from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_additional_metrics(predicted, target):
    # Flatten tensors to convert them into 1D arrays
    predicted_flat = predicted.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    precision = precision_score(target_flat, predicted_flat, average='binary')
    recall = recall_score(target_flat, predicted_flat, average='binary')
    f1 = f1_score(target_flat, predicted_flat, average='binary')

    return precision, recall, f1


def iou(predicted, target):
    predicted = predicted > 0.5  # Convert to binary: 1 if > 0.5 else 0
    target = target > 0.5  # Ensure target is also binary

    intersection = (predicted & target).float().sum()  # Intersection
    union = (predicted | target).float().sum()         # Union

    if union == 0:
        return 1  # Avoid division by 0; if both are 0, IoU is 1 by definition
    else:
        return intersection / union
    