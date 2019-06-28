'''
Created on 7 Jan 2018

@author: peter
'''
import io

from IPython.display import Image
from sklearn.metrics import classification_report, classification, regression
from sklearn.tree import export_graphviz
import numpy as np
import math

try:
  import pydotplus
except ImportError:
  pydotplus = None


def plotDecisionTree(decisionTree, feature_names=None, class_names=None, impurity=True, pdfFile=None):
    """ Create a plot of the decision tree and show in the Jupyter notebooke """
    if pydotplus is None:
      return 'You need to install pydotplus to visualize decision trees'
    dot_data = io.StringIO()
    export_graphviz(decisionTree, feature_names=feature_names, class_names=class_names, impurity=impurity,
                    out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    if pdfFile is not None:
      graph.write_pdf(pdfFile)
    return Image(graph.create_png())


def printConfusionMatrix(y_true, y_pred, class_names=None):
    """ Print a confusion matrix similar to R's confusionMatrix """
    confMatrix = classification.confusion_matrix(y_true, y_pred)
    accuracy = classification.accuracy_score(y_true, y_pred)

    print('Confusion Matrix (Accuracy {:.4f})\n'.format(accuracy))
    _printConfusionMatrix(confMatrix, class_names)

def classificationSummary(y_true, y_pred, class_names=None):
    """ Provide a comprehensive summary of classification performance similar to R's confusionMatrix """
    confMatrix = classification.confusion_matrix(y_true, y_pred)
    TP = confMatrix[0, 0]
    FP = confMatrix[1, 0]
    TN = confMatrix[1, 1]
    FN = confMatrix[0, 1]
    N = TN + TP + FN + FP
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    prevalence = (TP + FN) / N 
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    BAC = (sensitivity + specificity) / 2
    
    metrics = [
        ('Accuracy', classification.accuracy_score(y_true, y_pred)),
        ('95% CI', None),
        ('No Information Rate', None),
        ('P-Value [Acc > NIR]', None),
        (None, None),
        ('Kappa', classification.cohen_kappa_score(y_true, y_pred)),
        ("Mcnemar's Test P-Value", None),
        (None, None),
        ('Sensitivity', sensitivity),
        ('Specificity', specificity),
        ('Pos Pred Value', PPV),
        ('Neg Pred Value', NPV),
        ('Prevalence', prevalence),
        ('Detection Rate', None),
        ('Detection Prevalence', None),
        ('Balanced Accuracy', BAC),
        ]

    print('Confusion Matrix and Statistics\n')
    _printConfusionMatrix(confMatrix, class_names)
    if len(set(y_true)) < 5:
        print(classification_report(y_true, y_pred, digits=4))
    
    fmt1 = '{{:>{}}} : {{:.3f}}'.format(max(len(m[0]) for m in metrics if m[0] is not None))
    fmt2 = '{{:>{}}} : {{}}'.format(max(len(m[0]) for m in metrics if m[0] is not None))
    for metric, value in metrics:
        if metric is None:
            print()
        elif value is None:
            pass
            # print(fmt2.format(metric, 'missing'))
        else:
            print(fmt1.format(metric, value))


def regressionSummary(y_true, y_pred, timeSeries=False):
    """ print regression performance metrics """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_res = y_true - y_pred
    metrics = [
        ('Mean Error (ME)', sum(y_res) / len(y_res)),
        ('Root Mean Squared Error (RMSE)', math.sqrt(regression.mean_squared_error(y_true, y_pred))),
        ('Mean Absolute Error (MAE)', sum(abs(y_res)) / len(y_res)),
        ('Mean Percentage Error (MPE)', 100 * sum(y_res / y_true) / len(y_res)),
        ('Mean Absolute Percentage Error (MAPE)', 100 * sum(abs(y_res / y_true) / len(y_res))),
    ]
    if timeSeries:
        metrics.extend([
        ('Mean Absolute Scaled Error (MASE)', None),
        ('Autocorrelation of errors at lag 1 (ACF1)', None),
        ])
    fmt1 = '{{:>{}}} : {{:.4f}}'.format(max(len(m[0]) for m in metrics))
    fmt2 = '{{:>{}}} : {{}}'.format(max(len(m[0]) for m in metrics))
    print('\nRegression statistics\n')
    for metric, value in metrics:
        if value is None:
            print(fmt2.format(metric, 'missing'))
        else:
            print(fmt1.format(metric, value))


def _printConfusionMatrix(cm, labels):
    """ pretty print confusion matrixes """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    # Convert the confusion matrix and labels to strings
    cm = [[str(i) for i in row] for row in cm]
    labels = [str(i) for i in labels]

    # Determine the width for the first label column and the individual cells    
    prediction = 'Reference'
    labelWidth = max(len(s) for s in labels)
    cmWidth = max(max(len(s) for row in cm for s in row), labelWidth) + 1
    labelWidth = max(labelWidth, len(prediction))
    
    # Construct the format statements
    fmt1 = '{{:>{}}}'.format(labelWidth)
    fmt2 = '{{:>{}}}'.format(cmWidth) * len(labels)

    # And print the confusion matrix    
    print(fmt1.format(' ') + 'Prediction')
    print(fmt1.format(prediction), end='')
    print(fmt2.format(*labels))
    
    for cls, row in zip(labels, cm):
        print(fmt1.format(cls), end='')
        print(fmt2.format(*row))
