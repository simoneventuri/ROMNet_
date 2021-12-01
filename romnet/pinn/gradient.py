import tensorflow as tf



#===============================================================================
def get_grad_fn(ode_order=0):
    return order_0 if ode_order == 0 else order_n

#===============================================================================



#===============================================================================
def order_0(ode_cls, order, net, other_vars, ind_vars, training):
    assert order == 0

    x = [other_vars] + ind_vars
    x = tf.concat(x, axis=1)

    y_ = net(x, training=training)
    if isinstance(y_, (list,tuple)):
        y_ = tf.concat(y, axis=1)

    if (net.fROM_anti):
        y = net.fROM_anti(y_)
    else:
        y = y_

    return [y]

#===============================================================================



#===============================================================================
def order_n(ode_cls, order, net, other_vars, ind_vars, training):
    ### TODO: Generalize for cases with more independent variables

    if order == 1:
        grad_fn = order_0
    else:
        grad_fn = order_n
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(ind_vars)
        grads = grad_fn(ode_cls, order-1, net, other_vars, ind_vars, training)

        # Get `n-1`-th order gradients
        y     = tf.split(grads[-1], grads[-1].shape[1], axis=1)

    # Evaluate `n`-th order gradients
    dy = tf.concat([tape.gradient(i, ind_vars[0]) for i in y], axis=1)

    # Collect all the gradients
    grads.append(dy)

    return grads

#===============================================================================

