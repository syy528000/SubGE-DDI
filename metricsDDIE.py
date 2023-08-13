import csv
import sys
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support, classification_report
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def ddie_compute_metrics(task_name, preds, labels, every_type=False, output_dir=None):
        label_list = ('Mechanism', 'Effect', 'Advise', 'Int.')
        p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
        result = {"Precision": p,"Recall": r,"microF": f}


        """ROC Curve"""
        one_hot = np.eye(max(preds) + 1)
        preds = one_hot[preds]
        labels = one_hot[labels]
        n_classes = preds.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:,i],preds[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        """micro"""
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        """macro"""
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        """AUC"""
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        """ROC"""
        lw = 2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                label="micro-average ROC curve (AUC = {0:0.2f})"
                    ''.format(roc_auc["micro"]),
                color="deeppink", linestyle=":", linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (AUC = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','lawngreen','gold'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (AUC = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir,"roc_curve.png"))

        result["micro_auc"] = roc_auc["micro"]
        result["macro_auc"] = roc_auc["macro"]

        # if every_type:
            # result = classification_report(y_pred=preds,y_true=labels,target_names=label_list,output_dict=True)
            # for i, label_type in enumerate(label_list):
                # p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0,1], average='micro')
                # result[label_type + ' Precision'] = p
                # result[label_type + ' Recall'] = r
                # result[label_type + ' F'] = f
        return result

    def pretraining_compute_metrics(task_name, preds, labels, every_type=False):
        acc = accuracy_score(y_pred=preds, y_true=labels)
        result = {
            "Accuracy": acc,
        }
        return result
