from IPython.display import display
from pymongo import MongoClient
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib.ticker import MaxNLocator
from multiprocessing import Process, Queue
from ipyupload import FileUpload
from keras.models import load_model
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
learning_rate_widget = widgets.HBox([widgets.Label('learning rate:', layout=layout), widgets.SelectionSlider(options=[0.01, 0.001, 0.0001])])
epochs_widget = widgets.HBox([widgets.Label('Aantal epochs:', layout=layout), widgets.IntSlider(value=10, min=1, max=20)])

upload_widget = FileUpload(accept='image/*')

client = MongoClient()
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
    'epoch': 10
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


conv_layers_widget.children[1].observe(lambda change: update_model('conv_layers', change['new']), names='value')
ff_layers_widget.children[1].observe(lambda change: update_model('ff_layers', change['new']), names='value')
ff_input_widget.children[1].observe(lambda change: update_model('ff_input', change['new']), names='value')
learning_rate_widget.children[1].observe(lambda change: update_model('learning_rate', change['new']), names='value')
epochs_widget.children[1].observe(lambda change: update_model('epoch', change['new']), names='value')


def kies_netwerk():
    display(conv_layers_widget)
    display(ff_layers_widget)
    display(ff_input_widget)
    display(learning_rate_widget)
    display(epochs_widget)


def toon_netwerk():
    im = visualize_network.get_network_image(n_conv_layers=model_attr['conv_layers'], dropout=model_attr['regularization'], n_ff_layers=model_attr['ff_layers'], n_ff_input=model_attr['ff_input'])
    display(im)


def toon_loss():
    print('Training loss:\t\t' + str(model['loss']))
    print('Validation loss:\t' + str(model['val_loss']))
    print('Test loss:\t\t' + str(model['test_loss']))


def toon_accuracy():
    print('Training accuracy:\t' + str(model['acc']))
    print('Validation accuracy:\t' + str(model['val_acc']))
    print('Test accuracy:\t\t' + str(model['test_acc']))


def toon_afbeeldingen():
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


def toon_loss_grafiek():
    loss = []
    val_loss = []

    for temp_model in models_collection.find({'name': model['name'], 'epoch': {"$lte": model['epoch']}}).sort('epoch', 1):
        loss.append(temp_model['loss'])
        val_loss.append(temp_model['val_loss'])

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, len(loss) + 1), loss, 'b', label='Training loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'g', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def toon_accuracy_grafiek():
    acc = []
    val_acc = []

    for temp_model in models_collection.find({'name': model['name'], 'epoch': {"$lte": model['epoch']}}).sort('epoch', 1):
        acc.append(temp_model['acc'])
        val_acc.append(temp_model['val_acc'])

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, len(acc) + 1), acc, 'b', label='Training accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'g', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def vind_stomata():
    fig, ax = plt.subplots(figsize=(14, 12), nrows=1, ncols=3)
    im_objects = []
    points = []
    annotations = []
    for i, im_object in enumerate(model['full_images']):
        if im_object['name'] == 'kat':
            continue
        im_objects.append(im_object)

        im = Image.open(os.path.join(image_dir, im_object['name']))
        im_r = np.array(im)

        ax[i].imshow(im_r)
        points_im, = ax[i].plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
        points.append(points_im)
        annotation = ax[i].annotate('', (0, 0), (0, -30), xycoords='axes fraction', textcoords='offset points', fontsize=14, va='top')
        annotations.append(annotation)

    plt.close()

    def change_threshold(thr=50):
        for i, im_object in enumerate(im_objects):
            x_points = [x[0] for x in im_object['points'][str(thr)]]
            y_points = [x[1] for x in im_object['points'][str(thr)]]
            points[i].set_xdata(x_points)
            points[i].set_ydata(y_points)
            text = 'precision: ' + str(im_object['precision'][str(thr)]) + '\nrecall: ' + str(im_object['recall'][str(thr)])
            annotations[i].set_text(text)
        display(fig)

    widgets.interact(change_threshold, thr=widgets.IntSlider(value=50, min=5, max=95, step=5, continuous_update=False))


def laad_referentie_model():
    global model_attr
    model_attr = {
        'base_name': 'VGG19_base',
        'conv_layers': 2,
        'regularization': True,
        'ff_layers': 2,
        'ff_input': 1024,
        'optimizer': 'sgd',
        'learning_rate': 0.001,
        'epoch': 20
    }
    get_model()


def geef_gekozen_afbeelding():
    if not upload_widget.value:
        print('Je hebt nog geen file gekozen.')
        return None

    first_key = next(iter(upload_widget.value))

    if not upload_widget.value[first_key]['metadata']['type'].startswith('image'):
        print('Kies een afbeelding.')
        return None

    image = Image.open(io.BytesIO(upload_widget.value[first_key]['content']))
    return image


def vind_stomata_subproces(image, q):
    model_file = load_model(os.path.join(model_dir, model['name'] + '_model.h5'))

    shift = 10
    offset = 60
    bandwidth = offset

    im_r = np.array(image)

    stomata_punten = {}
    for thr in range(5, 100, 5):
        stomata_punten[str(thr)] = []

    no_x_shifts = (np.shape(im_r)[0] - 2 * offset) // shift
    no_y_shifts = (np.shape(im_r)[1] - 2 * offset) // shift

    for x in np.arange(no_x_shifts + 1):
        for y in np.arange(no_y_shifts + 1):
            x_c = x * shift + offset
            y_c = y * shift + offset

            im_r_crop = im_r[x_c - offset:x_c + offset, y_c - offset:y_c + offset, :]
            im_r_crop = im_r_crop.astype('float32')
            im_r_crop /= 255

            y_model = model_file.predict(np.expand_dims(im_r_crop, axis=0))

            for thr in range(5, 100, 5):
                if y_model[0][0] > thr / 100.:
                    stomata_punten[str(thr)].append([x_c, y_c])

    for thr in range(5, 100, 5):
        if stomata_punten[str(thr)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(stomata_punten[str(thr)])
                stomata_punten[str(thr)] = [[x[1], x[0]] for x in ms.cluster_centers_]  # Because cluster_centers is inverted

    q.put(stomata_punten)


def vind_stomata_eigen_afbeelding():
    image = geef_gekozen_afbeelding()
    if image is None:
        return

    # We voeren dit uit in een appart proces omdat de gpu memory dan wordt vrijgegeven
    q = Queue()
    p = Process(target=vind_stomata_subproces, args=(image, q))
    p.start()
    p.join()
    stomata_punten = q.get()

    fig, ax = plt.subplots()
    ax.imshow(image)
    points_im, = ax.plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
    plt.close()

    def change_threshold(thr=50):
        x_points = [x[0] for x in stomata_punten[str(thr)]]
        y_points = [x[1] for x in stomata_punten[str(thr)]]
        points_im.set_xdata(x_points)
        points_im.set_ydata(y_points)
        display(fig)

    widgets.interact(change_threshold, thr=widgets.IntSlider(value=50, min=5, max=95, step=5, continuous_update=False))


def misleid_netwerk():
    image_path = os.path.join(image_dir, 'kat.jpg')
    image = Image.open(image_path)

    for full_image in model['full_images']:
        if full_image['name'] != 'kat':
            continue
        stomata_punten = full_image['points']

    fig, ax = plt.subplots()
    ax.imshow(image)
    points_im, = ax.plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
    plt.close()

    def change_threshold(thr=50):
        x_points = [x[0] for x in stomata_punten[str(thr)]]
        y_points = [x[1] for x in stomata_punten[str(thr)]]
        points_im.set_xdata(x_points)
        points_im.set_ydata(y_points)
        display(fig)

    widgets.interact(change_threshold, thr=widgets.IntSlider(value=50, min=5, max=95, step=5, continuous_update=False))


print('Import succesvol, je kan beginnen!')
