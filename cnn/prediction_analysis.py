from cnn.load_data_std import get_data
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

X_validation, y_validation = get_data(False)

model = load_model('cnn_train_model.h5')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
y_pred = model.predict(X_validation)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validation, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10))

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_prob_errors = y_pred[errors]
Y_true_classes_errors = Y_true[errors]
X_validation_errors = X_validation[errors]


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1


# Probabilities of the wrong predicted numbers
Y_pred_maxProb_errors = np.max(Y_pred_prob_errors, axis=1)

# Predicted probabilities of the true values in the error set
Y_true_prob_errors = np.diagonal(np.take(Y_pred_prob_errors, Y_true_classes_errors, axis=1))

# Difference between the probability of the predicted label and the true label
deltaProb_pred_true_errors = Y_pred_maxProb_errors - Y_true_prob_errors

# Sorted list of the delta prob errors
sorted_delaProb_errors = np.argsort(deltaProb_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_delaProb_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_validation_errors, Y_pred_classes_errors, Y_true_classes_errors)
