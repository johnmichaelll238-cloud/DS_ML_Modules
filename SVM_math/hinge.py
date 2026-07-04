import numpy as np
def hinge_loss(y_true, y_pred):
    #Compute the margins
    margins = y_true * y_pred
    losses = np.maximum(0, 1 - margins)
    #Return average loss
    return np.mean(losses)


y_true = np.array([1, -1, 1])
y_pred = np.array([0.8, 0.3, -0.2])

mean = hinge_loss(y_true, y_pred)
print(mean)