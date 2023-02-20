from sklearn.metrics import confusion_matrix

from autodefer.utils import thresholding as t


def add_fp_fn(df, label_col, pred_col, drop_fp=False, drop_fn=False):
    if not drop_fp:
        df['FP'] = ((df[label_col] == 0) & (df[pred_col] == 1)).astype(int)
    if not drop_fn:
        df['FN'] = ((df[label_col] == 1) & (df[pred_col] == 0)).astype(int)

    return df

def calc_tpr_at_fpr(y_score, y_true, target_fpr):
    y_pred = t.predict_with_threshold(
        y_score=y_score,
        threshold=t.calc_threshold_at_fpr(
            y_true=y_true, y_score=y_score, fpr=target_fpr
        )
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=[0, 1]).ravel()

    fpr = fp / (tn + fp)
    tpr = tp / (tp + fn)

    if fpr > target_fpr:  # classifiers with uniform scores around the threshold ruin the thresholding
        tpr = 0

    return tpr
