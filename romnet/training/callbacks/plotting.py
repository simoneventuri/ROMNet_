import os
import copy
import numpy as np
import pandas as pd
import tensorflow.keras.callbacks as CB

from tqdm import tqdm
from tensorflow.python.util.tf_export import keras_export


#=======================================================================================================================================
@keras_export('keras.callbacks.Plotter')
class Plotter(CB.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        test_data=None,
        train_dir=None,
        output_ops=None,
        var_names=None,
        time_idx=-1,
        fig_name=None,
        labels=None,
        scales=None,
        limit_y=False,
        log_plot=False,
        freq=100
    ):
        super(Plotter, self).__init__()

        self.test_data = test_data

        self.train_dir = train_dir
        self.path = {
            'train': self.train_dir+'plots/training',
            'test':  self.train_dir+'plots/testing'
        }
        for k, v in self.path.items():
            if not os.path.exists(v):
                os.makedirs(v)

        self.output_ops = output_ops

        self.var_names = var_names
        self.labels = labels
        self.time_idx = time_idx

        self.scales = scales
        self.log_plot = log_plot
        self.limit_y = limit_y
        
        self.freq = freq
        self.fig_name = fig_name

    # def on_epoch_end(self, epoch, logs=None):
    #     if epoch % self.freq == 0:
    #         self.training(limit_y=self.limit_y)
    #         if self.test_data is not None:
    #             self.testing()
    #         print("")

    # # def training(self, scale='log', limit_y=False):
    # #     history = pd.DataFrame.from_dict(self.model.history.history)
    # #     path = self.path['train']
    # #     # Learning rate
    # #     post.training.plot_lr(self.train_dir+'/lr.csv', path)
    # #     # Losses
    # #     if 'loss' in history:
    # #         post.training.plot_loss(
    # #             history, path, scale=scale, limit_y=limit_y
    # #         )
    # #     elif 'tot_loss' in history:
    # #         post.training.plot_composite_loss(
    # #             history, path, scale=scale, limit_y=limit_y
    # #         )
    # #         if any([ c.endswith('weight') for c in history ]):
    # #             post.training.plot_loss_weights(history, path)
    # #     # Metrics
    # #     if self.model.compiled_metrics:
    # #         for metric in self.model.compiled_metrics.metrics:
    # #             if metric.name in history:
    # #                 post.training.plot_metric(
    # #                     history,
    # #                     path,
    # #                     metric.name,
    # #                     scale=scale,
    # #                     limit_y=limit_y
    # #                 )

    # # def testing(self):
    # #     for i, test_i in enumerate(tqdm(copy.deepcopy(self.test_data))):
    # #         # x, y_true = self.unpack_test_data(test_i)
    # #         x, y_true = test_i
    # #         y_pred = self.model.predict(x)
    # #         if self.output_ops:
    # #             y_true, y_pred = list(map(self.output_ops, [y_true, y_pred]))

    # #         path = self.path['test'] + '/test_'+str(i+1)
    # #         if not os.path.exists(path):
    # #             os.makedirs(path)
    # #         if isinstance(y_pred, (list,tuple)):
    # #             if self.fig_name == None:
    # #                 self.fig_name = [
    # #                     'fig'+str(i) for i in range(1,len(y_true)+1)
    # #                 ]
    # #             for j in range(len(y_pred)):
    # #                 kwargs = dict(
    # #                     y_true=y_true[j],
    # #                     y_pred=y_pred[j],
    # #                     var_names=self.var_names[j],
    # #                     time_idx=self.time_idx,
    # #                     fig_name=self.fig_name[j],
    # #                     labels=self.labels[j],
    # #                     scales=self.scales[j]
    # #                 )
    # #                 self.testing_plot_fn(path, x, kwargs)
    # #         else:
    # #             kwargs = dict(
    # #                 y_true=y_true,
    # #                 y_pred=y_pred,
    # #                 var_names=self.var_names,
    # #                 time_idx=self.time_idx,
    # #                 fig_name=self.fig_name,
    # #                 labels=self.labels,
    # #                 scales=self.scales
    # #             )
    # #             self.testing_plot_fn(path, x, kwargs)

    # # def testing_plot_fn(self, path, x, kwargs):
    # #     post.testing.plot_test(path, x, **kwargs)
    # #     if self.log_plot:
    # #         kwargs['scales'] = ['log','log']
    # #         post.testing.plot_test(path, x,**kwargs)

#=======================================================================================================================================