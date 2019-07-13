# Find Best Learning Rate On Plateau 
A Keras callback to automatically adjust the learning rate when it stops improving.

Learning rate finder is based off this code here: https://github.com/surmenok/keras_lr_finder which implements [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) by Lesie Smith but doesn't wrap it up completely.

This method works on the principal that any particular learning rate is only good until it stops improving your loss. At that point you need to select a new learning rate, typically this is done by a decay function (as in `keras.callbacks.ReduceLROnPlateau`). The method used here is to simply search for the best learning rate by training for a few epochs, increasing the learning rate as we go and selecting the one that gives the largest learning rate improvement. The process is something like this:

1. Train a model for a large number of epochs
2. If the model's loss fails to improve for `n` epochs:
 1. Take a snapshot of the model
 2. Set training rate to `min_lr` and train for a batch
 3. Increase the learning rate exponentially toward `max_lr` after every batch.
 4. Once candidate learning rates have been exhausted, select `new_lr` as the learning rate that gave the steepest negative gradient in loss.
 5. Set model's learning rate to `new_lr` and continue training as normal


I initially tried performing this process every `n` epochs, but it later occurred to me that I should do it only when loss stops improving (a la ReduceLROnPlateau), which is what this repository does.
