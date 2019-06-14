from IPython.display import display
from pymongo import MongoClient
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib.ticker import MaxNLocator
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import warnings
import io
import os
import gc
import imp
with open('../.scripts/visualize_network.py', 'rb') as fp:
    visualize_network = imp.load_module('.scripts', fp, '../.scripts/visualize_network.py', ('.py', 'rb', imp.PY_SOURCE))

image_dir = '../.images/IntroductieDeepLearning'
model_dir = '../.data/IntroductieDeepLearning'

layout = widgets.Layout(width='30%')

conv_layers_widget = widgets.HBox([widgets.Label('Aantal convolutionele lagen:', layout=layout), widgets.IntSlider(value=1, min=1, max=3)])
ff_layers_widget = widgets.HBox([widgets.Label('Aantal feedforward lagen:', layout=layout), widgets.IntSlider(value=1, min=1, max=2)])
ff_input_widget = widgets.HBox([widgets.Label('Aantal neuronen in eerste feedforward laag:', layout=layout), widgets.Dropdown(options=[16, 64, 256, 1024])])
learning_rate_widget = widgets.HBox([widgets.Label('learning rate:', layout=layout), widgets.SelectionSlider(value=0.01, options=[0.1, 0.01, 0.001, 0.0001])])
epochs_widget = widgets.HBox([widgets.Label('Aantal epochs:', layout=layout), widgets.IntSlider(value=20, min=1, max=20)])

client = MongoClient('mongodb://database:27017/')
db = client.models
models_collection = db.models

model_attr = {
    'base_name': 'conv_base',
    'conv_layers': 1,
    'regularization': False,
    'ff_layers': 1,
    'ff_input': 16,
    'optimizer': 'sgd',
    'learning_rate': 0.01,
    'epoch': 50
}

model = None


def update_model(attr, value):
    if attr in model_attr:
        model_attr[attr] = value
        get_model()
    else:
        warnings.warn('Het model heeft geen attribuut ' + attr + '.')


def get_model():
    name = (model_attr['base_name'] if model_attr['base_name'] == 'VGG19_base' else model_attr['base_name'] + '_' + str(model_attr['conv_layers']) + '-cl') + ('_regularization_' if model_attr['regularization'] else '_no-regularization_') + str(model_attr['ff_layers']) + '-ffl_' + str(model_attr['ff_input']) + '-ffn_' + model_attr['optimizer'] + '_lr-' + "{:.0E}".format(model_attr['learning_rate'])

    global model
    model = models_collection.find_one({'name': name, 'epoch': (model_attr['epoch'] - 1)})


get_model()
conv_layers_widget.children[1].observe(lambda change: update_model('conv_layers', change['new']), names='value')
ff_layers_widget.children[1].observe(lambda change: update_model('ff_layers', change['new']), names='value')
ff_input_widget.children[1].observe(lambda change: update_model('ff_input', change['new']), names='value')
learning_rate_widget.children[1].observe(lambda change: update_model('learning_rate', change['new']), names='value')
#epochs_widget.children[1].observe(lambda change: update_model('epoch', change['new']), names='value')


def kies_netwerk_parameters():
    display(conv_layers_widget)
    display(ff_layers_widget)
    display(ff_input_widget)


def kies_training_parameters():
    display(learning_rate_widget)


def toon_netwerk():
    im = visualize_network.get_network_image(base=model_attr['base_name'], n_conv_layers=model_attr['conv_layers'], dropout=model_attr['regularization'], n_ff_layers=model_attr['ff_layers'], n_ff_input=model_attr['ff_input'])
    display(im)


def toon_voorspellingen():
    fig, big_axes = plt.subplots(figsize=(14, 5), nrows=2, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25)

    big_axes[0].set_title("Afbeeldingen met stoma")
    big_axes[1].set_title("Afbeeldingen zonder stoma")
    big_axes[0].set_axis_off()
    big_axes[1].set_axis_off()

    axes = []
    images = model['stoma_patches'] + model['no_stoma_patches']
    for i, image in enumerate(images):
        im = Image.open(os.path.join(image_dir, image['name']))
        ax = fig.add_subplot(2, 6, i + 1)
        ax.imshow(im)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        axes.append(ax)

        xlabel = "Pred: {:f}".format(float(image['prediction']))
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.close()

    def change_threshold(thr=0.5):
        for i, image in enumerate(images):
            if (float(image['prediction']) < thr and image in model['stoma_patches']) or (float(image['prediction']) > thr and image in model['no_stoma_patches']):
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('r')
            else:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('g')

        display(fig)

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.05, continuous_update=False))


def toon_slechte_voorspellingen():
    fig, big_axes = plt.subplots(figsize=(14, 5), nrows=2, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25)

    big_axes[0].set_title("Afbeeldingen met stoma (false negative)")
    big_axes[1].set_title("Afbeeldingen zonder stoma (false positive)")
    big_axes[0].set_axis_off()
    big_axes[1].set_axis_off()

    axes = []
    images = model['false_negatives'] + model['false_positives']
    for i, image in enumerate(images):
        im = Image.open(os.path.join(image_dir, image['name']))
        ax = fig.add_subplot(2, 6, i + 1)
        ax.imshow(im)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        axes.append(ax)

        xlabel = "Pred: {:f}".format(float(image['prediction']))
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.close()

    def change_threshold(thr=0.5):
        for i, image in enumerate(images):
            if (float(image['prediction']) < thr and image in model['false_negatives']) or (float(image['prediction']) > thr and image in model['false_positives']):
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('r')
            else:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor('g')

        display(fig)

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.05, continuous_update=False))


def toon_test_resultaten():
    print('Test loss:\t' + str(model['test_loss']))
    print('Test accuracy:\t' + str(model['test_acc']))


def toon_grafiek():
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    for temp_model in models_collection.find({'name': model['name'], 'epoch': {"$lte": model['epoch']}}).sort('epoch', 1):
        loss.append(temp_model['loss'])
        val_loss.append(temp_model['val_loss'])
        acc.append(temp_model['acc'])
        val_acc.append(temp_model['val_acc'])

    fig, ax = plt.subplots(figsize=(14, 5), nrows=1, ncols=2)

    title_1 = ('Training loss: \t\t' + str(model['loss']) + '\nValidation loss:\t' + str(model['val_loss'])).expandtabs()
    ax[0].set_title(title_1, loc='left')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].plot(range(1, len(loss) + 1), loss, 'b', label='Training loss')
    ax[0].plot(range(1, len(val_loss) + 1), val_loss, 'g', label='Validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    title_2 = ('Training accuracy: \t' + str(model['acc']) + '\nValidation accuracy:\t' + str(model['val_acc'])).expandtabs()
    ax[1].set_title(title_2, loc='left')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].plot(range(1, len(acc) + 1), acc, 'b', label='Training accuracy')
    ax[1].plot(range(1, len(val_acc) + 1), val_acc, 'g', label='Validation accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()


def vind_stomata():
    fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=3)
    im_objects = []
    points = []
    annotations = []
    for i, im_object in enumerate(model['full_images']):
        im_objects.append(im_object)

        im = Image.open(os.path.join(image_dir, im_object['name']))
        im_r = np.array(im)

        ax[i].imshow(im_r)
        points_im, = ax[i].plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
        points.append(points_im)
        annotation = ax[i].annotate('', (0, 0), (0, -30), xycoords='axes fraction', textcoords='offset points', fontsize=14, va='top')
        annotations.append(annotation)

    plt.close()

    def change_threshold(thr=0.5):
        for i, im_object in enumerate(im_objects):
            x_points = [x[0] for x in im_object['points'][str(int(thr * 100))]]
            y_points = [x[1] for x in im_object['points'][str(int(thr * 100))]]
            points[i].set_xdata(x_points)
            points[i].set_ydata(y_points)
            text = 'precision: ' + str(im_object['precision'][str(int(thr * 100))]) + '\nrecall: ' + str(im_object['recall'][str(int(thr * 100))])
            annotations[i].set_text(text)
        display(fig)

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.05, continuous_update=False))


def laad_referentie_model():
    global model_attr
    model_attr = {
        'base_name': 'conv_base',
        'conv_layers': 3,
        'regularization': True,
        'ff_layers': 1,
        'ff_input': 64,
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'epoch': 50
    }
    get_model()
    print('Gelukt!')


def misleid_netwerk():
    fig, ax = plt.subplots(nrows=1, ncols=len(model['adverserial_images']), squeeze=False)
    im_objects = []
    points = []
    for i, im_object in enumerate(model['adverserial_images']):
        im_objects.append(im_object)

        im = Image.open(os.path.join(image_dir, im_object['name']))
        im_r = np.array(im)

        ax[0][i].imshow(im_r)
        points_im, = ax[0][i].plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
        points.append(points_im)

    plt.close()

    def change_threshold(thr=0.5):
        for i, im_object in enumerate(im_objects):
            x_points = [x[0] for x in im_object['points'][str(int(thr * 100))]]
            y_points = [x[1] for x in im_object['points'][str(int(thr * 100))]]
            points[i].set_xdata(x_points)
            points[i].set_ydata(y_points)
        display(fig)

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.05, continuous_update=False))


print('Import succesvol, je kan beginnen!')
