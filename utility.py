import tensorflow as tf
import numpy as np
from sklearn import metrics


def evaluate(model, test_dataset=None):
    y_score = model.predict(test_dataset)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}")
    y_pred = np.rint(y_score)  # to have 0 or 1
    y_true = tf.concat([labels_batch for data_batch, labels_batch in test_dataset], axis=0)
    print("Classification report: ")
    print(type(y_true))
    print(type(y_pred))
    print(y_pred.argmax(axis=1))
    print(metrics.classification_report(y_true.numpy().argmax(axis=1), y_pred.argmax(axis=1), digits=4))
    metrics.ConfusionMatrixDisplay.from_predictions(y_true.numpy().argmax(axis=1), y_pred.argmax(axis=1))

    # ROC curve
    '''fpr,tpr,th = metrics.roc_curve(y_true,y_score)
      roc_auc = metrics.roc_auc_score(y_true,y_score)
    
      plt.figure()
      plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.0])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('ROC curve')
      plt.legend(loc="lower right")
      plt.show()'''
