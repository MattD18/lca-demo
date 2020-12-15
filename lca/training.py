# Module to handle LCA calculation during training
import time
import numpy as np
import tensorflow as tf

def train_and_calculate_lca(model, num_params, loss_fn, optimizer, x_train, y_train, x_val, y_val, 
                            batch_size=227,
                            num_epochs=4):
    '''
    Trains the model via mini-batch gradient descent while gathering needed
    objects for lca calculation

    Paramters:
    ----------
    model : tf.keras.Model
    num_params : int
        number of params in model, used to verify correct number of weights
        are stored for lca calculation
    loss_fn : tf.keras.losses.Loss
    optimizer : tf.keras.optimizers.Optimizer
    x_train : np.ndarray
    y_train : np.ndarray
    x_val : np.ndarray 
    y_val : np.ndarray
    batch_size : int (default = 227)
    num_epochs : int (default = 4)

    Returns:
    ----------
    output : dict

    '''
    # Prepare the training dataset.
    training_size = x_train.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=training_size).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    training_loss_metric = tf.keras.metrics.Mean()
    training_loss_list = []
    train_acc_list = []
    val_acc_list = []
    step_loss_list = []

    # Store weights as rows in # iterations x # parameter matrix
    step_parameters_over_time = []

    # store grads as rows in # iterations x # parameter matrix
    step_grads_over_time = []


    training_start = time.time()
    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            
            # Store weights at start of step
            step_weights = np.array([])
            for layer in model.weights:
                layer_weights = np.copy(layer.numpy())
                # flatten weights 
                layer_weights = layer_weights.flatten()
                # add to iteration weight
                step_weights = np.concatenate([step_weights, layer_weights], axis=0)
            assert step_weights.shape[0] == num_params
            step_parameters_over_time.append(step_weights)
            
            ## TAKE GRADIENT STEP BASED ON MINIBATCH
            
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)
            # Update loss metric.
            training_loss_metric.update_state(loss_value)
            
            # Log every 20 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
            
            ## RECORD GRADIENT ON FULL TRAINING SET FOR LCA
            with tf.GradientTape() as full_tape:
                # Run the forward pass of the layer on the entire training set
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                full_logits = model(x_train, training=False)
                # Compute the loss value on the entire training set.
                full_loss_value = loss_fn(y_train, full_logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            full_grads = full_tape.gradient(full_loss_value, model.trainable_weights)
            
            # save full training loss
            step_loss_list.append(np.copy(full_loss_value.numpy()))
            # save full training loss gradient
            step_grads = np.array([])
            # assumes grads are being concatenated in same order as weights,
            # otherwise LCA will be wrongly allocated
            for layer in full_grads:
                layer_grads = np.copy(layer)
                # flatten weights 
                layer_grads = layer_grads.flatten()
                # add to iteration weight
                step_grads = np.concatenate([step_grads, layer_grads], axis=0)
            assert step_grads.shape[0] == num_params
            step_grads_over_time.append(step_grads)
        
        # Display metrics at the end of each epoch and add to list
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_list.append(train_acc)
        # save training loss to list
        training_loss_result = training_loss_metric.result()
        training_loss_list.append(training_loss_result)
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        training_loss_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_list.append(val_acc)
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    training_end = time.time()

    print(training_end - training_start)

    print('Calculating LCAs')
    step_lcas_over_time = _calculate_lcas(step_parameters_over_time, step_grads_over_time)

    # gather output
    output = {
        'train_acc' : train_acc_list,
        'val_acc' : val_acc_list,
        'train_loss' : step_loss_list,
        'lcas' : step_lcas_over_time
    }

    return output


def _calculate_lcas(step_parameters_over_time, step_grads_over_time):
    '''
    calculate lcas from objects collected during training

    Parameters:
    ----------
    step_parameters_over_time : list
    step_grads_over_time : list

    Returns:
    ---------
    step_lcas_over_time : list
    '''
    num_steps = len(step_parameters_over_time)
    step_lcas_over_time = []
    for i in range(num_steps - 1):
        theta_t0 = step_parameters_over_time[i]
        theta_t1 = step_parameters_over_time[i+1]
        delta_theta = theta_t1 - theta_t0
        grad_theta = step_grads_over_time[i]
        summands = grad_theta * delta_theta
        step_lcas_over_time.append(summands)
        
    print(len(step_lcas_over_time))
    step_lcas_over_time = np.stack(step_lcas_over_time)
    print(step_lcas_over_time.shape)

    return step_lcas_over_time