from IPython.display import display
from scripts import visualize_network
from pymongo import MongoClient
from PIL import Image
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import os

image_dir = './images'

layout = widgets.Layout(width='30%')

conv_base = widgets.HBox([widgets.Label('Aantal convolutionele lagen:', layout=layout), widgets.IntSlider(value=1, min=1, max=3)])
ff_layers = widgets.HBox([widgets.Label('Aantal feedforward lagen:', layout=layout), widgets.IntSlider(value=1, min=1, max=2)])
ff_input = widgets.HBox([widgets.Label('Aantal neuronen in eerste feedforward laag:', layout=layout), widgets.Dropdown(options=[16, 64, 256, 1024])])
learning_rate = widgets.HBox([widgets.Label('learning rate:', layout=layout), widgets.SelectionSlider(options=[0.01, 0.001, 0.0001])])
epochs = widgets.HBox([widgets.Label('Aantal epochs:', layout=layout), widgets.IntSlider(value=10, min=1, max=20)])

client = MongoClient()
db = client.models
models_collection = db.models


def kies_netwerk():
    display(conv_base)
    display(ff_layers)
    display(ff_input)
    display(learning_rate)
    display(epochs)


def toon_netwerk(regularization=False):
    n_conv_layers = int(conv_base.children[1].value)
    n_ff_layers = int(ff_layers.children[1].value)
    n_ff_input = int(ff_input.children[1].value)
    im = visualize_network.get_network_image(n_conv_layers=n_conv_layers, dropout=regularization, n_ff_layers=n_ff_layers, n_ff_input=n_ff_input)
    display(im)


def get_model(base_name='conv_base', regularization=False, optimizer='sgd'):
    name = (base_name if base_name == 'VGG19_base' else base_name + '_' + str(conv_base.children[1].value) + '-cl') + ('_regularization_' if regularization else '_no-regularization_') + str(ff_layers.children[1].value) + '-ffl_' + str(ff_input.children[1].value) + '-ffn_' + optimizer + '_lr-' + "{:.0E}".format(float(learning_rate.children[1].value))

    epoch = int(epochs.children[1].value)

    model = models_collection.find_one({'name': name, 'epoch': (epoch - 1)})
    return model


def toon_accuracy(base_name='conv_base', regularization=False, optimizer='sgd'):
    model = get_model(base_name, regularization, optimizer)
    print('Training accuracy:\t' + str(model['acc']))
    print('Validation accuracy:\t' + str(model['val_acc']))
    print('Test accuracy:\t\t' + str(model['test_acc']))


def toon_afbeeldingen(base_name='conv_base', regularization=False, optimizer='sgd'):
    model = get_model(base_name, regularization, optimizer)

    fig, big_axes = plt.subplots(figsize=(7, 5), nrows=2, ncols=1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25)

    big_axes[0].set_title("Afbeeldingen met stoma")
    big_axes[1].set_title("Afbeeldingen zonder stoma")
    big_axes[0].set_xticks([])
    big_axes[1].set_yticks([])
    big_axes[0]._frameon = False
    big_axes[1]._frameon = False

    images = model['stoma_patches'] + model['no_stoma_patches']
    for i, image in enumerate(images):
        im = Image.open(os.path.join(image_dir, image['name']))
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(im)

        xlabel = "Pred: {:f}".format(float(image['prediction']))
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def toon_accuracy_grafiek(base_name='conv_base', regularization=False, optimizer='sgd'):
    acc = []
    val_acc = []

    name = (base_name if base_name == 'VGG19_base' else base_name + '_' + str(conv_base.children[1].value) + '-cl') + ('_regularization_' if regularization else '_no-regularization_') + str(ff_layers.children[1].value) + '-ffl_' + str(ff_input.children[1].value) + '-ffn_' + optimizer + '_lr-' + "{:.0E}".format(float(learning_rate.children[1].value))
    for model in models_collection.find({'name': name, 'epoch': {"$lt": epochs.children[1].value}}).sort('epoch', 1):
        acc.append(model['acc'])
        val_acc.append(model['val_acc'])

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, len(acc) + 1), acc, 'b', label='Training accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'g', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def vind_stomata(base_name='conv_base', regularization=False, optimizer='sgd'):
    model = get_model(base_name, regularization, optimizer)

    fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=3)
    im_objects = []
    points = []
    for i, im_object in enumerate(model['full_images']):
        im_objects.append(im_object)

        im = Image.open(os.path.join(image_dir, im_object['name']))
        im_r = np.array(im)

        ax[i].imshow(im_r)
        points_im, = ax[i].plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
        points.append(points_im)

    plt.close()

    def change_threshold(thr=50):
        for i, im_object in enumerate(im_objects):
            x_points = [x[0] for x in im_object['points'][str(thr)]]
            y_points = [x[1] for x in im_object['points'][str(thr)]]
            points[i].set_xdata(x_points)
            points[i].set_ydata(y_points)
        display(fig)

    widgets.interact(change_threshold, thr=(5, 95, 5))


print('Import succesvol, je kan beginnen!')
