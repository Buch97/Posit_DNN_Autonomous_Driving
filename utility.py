import tensorflow as tf
import numpy as np
from sklearn import metrics


def evaluate(model, x_test, y_test):
    y_score = model.predict(x_test)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)
    correct_predictions = np.sum(np.equal(y_pred, y_true))
    accuracy = correct_predictions / len(y_true)
    print("accuracy: " + str(accuracy))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}")

    print("Classification report: ")
    print(type(y_true))
    print(type(y_pred))
    #print(y_pred.argmax(axis=1))
    print(metrics.classification_report(y_true, y_pred, digits=4))
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

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
