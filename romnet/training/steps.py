import sys
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils  import losses_utils



#=======================================================================================================================================
def train_step(net, data):
    """Custom train step."""

    x, y, indx = data_adapter.expand_1d(data)


    # Gradient descent step
    losses = {}
    with tf.GradientTape() as tape:
        for i, data_id in enumerate(net.data_ids):
            losses[data_id] = _get_loss(i, x[i], y[i], tf.squeeze(indx[i]), net, data_id)

        # Regularization loss
        reg_loss = _get_reg_loss(net)

        # Calculate total loss
        tot_loss, tot_loss_weighted = _get_tot_loss(net, losses, reg_loss)

    
    # Collect metrics to return
    if net.pde_loss_weights is None:
        metric = {}
    else:
        # Add weighted total loss
        metric = {'tot_loss_weighted': tot_loss_weighted}
    metrics = _get_metrics(tot_loss, losses, reg_loss=reg_loss)


    # Update parameters
    net.optimizer.minimize( tot_loss_weighted, net.trainable_variables, tape=tape )

    return {**metric, **metrics}

#=======================================================================================================================================



#=======================================================================================================================================
def test_step(net, data):
    """Custom test step."""

    x, y, indx = data_adapter.expand_1d(data)

    losses = {}
    for i, data_id in enumerate(net.data_ids_valid):
        losses[data_id] = _get_loss(i, x[i], y[i], indx[i], net, data_id, training=False)

    # Calculate total loss
    total_loss = _get_tot_loss_test(net, losses)

    # Collect metrics to return
    metrics = _get_metrics_test(total_loss, losses)

    return metrics

#=======================================================================================================================================



#=======================================================================================================================================
def _get_loss(i, x, y, indx, net, data_id, training=True):
            
    if data_id == 'res':
        y_pred = net.residual(x, training=training)
        if isinstance(y_pred, (list,tuple)):
            y = [ tf.zeros_like(y_pred_i) for y_pred_i in y_pred ]
        else:
            y = tf.zeros_like(y_pred)
    else:
        y_pred = net(x, training=training)
        if (net.fROM_anti):
            y_pred = net.fROM_anti(y_pred)

    if ((net.attention_mask is None) or (data_id == 'pts') or (not training)):
        attention_mask = None
    else:
        attention_mask = tf.gather(net.attention_mask[i], indx, axis=0)         
    
    return net.compiled_loss[data_id](y, y_pred, attention_mask=attention_mask)

#=======================================================================================================================================



#=======================================================================================================================================
def _get_reg_loss(net):

    if net.losses:
        reg_losses = losses_utils.cast_losses_to_common_dtype(net.losses)
        reg_loss   = tf.math.add_n(reg_losses)
        return reg_loss

#=======================================================================================================================================



#=======================================================================================================================================
def _get_tot_loss(net, losses, reg_loss):
    
    total_loss          = reg_loss 
    total_loss_weighted = reg_loss 

    for data_id, loss in losses.items():

        if net.pde_loss_weights is not None:
            total_loss_weighted += net.pde_loss_weights[data_id] * loss 
        else:
            total_loss_weighted += loss 
        
        total_loss += loss 

    return total_loss, total_loss_weighted

#=======================================================================================================================================



#=======================================================================================================================================
def _get_tot_loss_test(net, losses):
    
    total_loss = 0.
    for data_id, loss in losses.items():

        total_loss += loss 

    return total_loss

#=======================================================================================================================================



#=======================================================================================================================================
def _get_metrics(total_loss, losses, reg_loss):
    metrics             = {}
    metrics['tot_loss'] = total_loss 
    for data_id, loss in losses.items():
        metrics[data_id + '_loss'] = loss 
    if reg_loss is not None:
        metrics['reg_loss'] = reg_loss  
    return metrics

#=======================================================================================================================================



#=======================================================================================================================================
def _get_metrics_test(total_loss, losses):
    metrics             = {}
    metrics['tot_loss'] = total_loss 
    for data_id, loss in losses.items():
        metrics[data_id + '_loss'] = loss 
    return metrics

#=======================================================================================================================================