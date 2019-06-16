from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from keras.models import load_model
from keras import backend as K
from ipyupload import FileUpload
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
import tensorflow as tf
import numpy as np
import warnings
import GPUtil
import io
import os

reference_model = 'detecting_stomata_model_VGG19FT.h5'
model_dir = '../.data/IntroductieDeepLearning'

upload_widget = FileUpload(accept='image/*')

image_crop = None


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


def toon_eigen_afbeelding():
    image = geef_gekozen_afbeelding()
    if image is None:
        return

    im_r = np.array(image)

    height, width, depth = im_r.shape
    rect_width = width
    rect_height = height
    if rect_width > 800:
        rect_width = 800
    if rect_height > 800:
        rect_height = 800

    fig = plt.figure()
    plt.imshow(im_r)
    rect = patches.Rectangle((0, 0), rect_width, rect_height, linewidth=-1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.close()

    def choose_regio(x, y):
        global image_crop
        image_crop = im_r[y: y + rect_height, x: x + rect_width, :]
        rect.set_xy((x, y))
        display(fig)

    max_x = max(width - 800, 0)
    max_y = max(height - 800, 0)
    widgets.interact(choose_regio, x=widgets.IntSlider(min=0, max=max_x, continuous_update=False), y=widgets.IntSlider(min=0, max=max_y, continuous_update=False))


def vind_stomata_subproces(im_r, q):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    K.set_session(sess)

    model_file = load_model(os.path.join(model_dir, reference_model))

    shift = 10
    offset = 60
    bandwidth = offset

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
                if y_model[0][1] > thr / 100.:
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
    global image_crop
    if image_crop is None:
        return

    GPUs = GPUtil.getGPUs()
    available_gpu_ids = []
    for gpu in GPUs:
        if gpu.memoryFree > 1000:
            available_gpu_ids.append(gpu.id)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        for id in available_gpu_ids:
            if str(id) not in os.environ["CUDA_VISIBLE_DEVICES"].split(','):
                available_gpu_ids.remove(id)

    if not available_gpu_ids:
        print('Niet genoeg GPU memory beschikbaar, probeer straks opnieuw.')
        return

    # We voeren dit uit in een appart proces omdat de gpu memory dan wordt vrijgegeven
    q = Queue()
    p = Process(target=vind_stomata_subproces, args=(image_crop, q))
    p.start()
    p.join()
    stomata_punten = q.get()

    fig, ax = plt.subplots()
    ax.imshow(image_crop)
    points_im, = ax.plot([], [], '+c', alpha=0.6, markeredgewidth=3, markersize=12)
    plt.close()

    def change_threshold(thr=0.5):
        x_points = [x[0] for x in stomata_punten[str(int(thr * 100))]]
        y_points = [x[1] for x in stomata_punten[str(int(thr * 100))]]
        points_im.set_xdata(x_points)
        points_im.set_ydata(y_points)
        display(fig)
        print('Aantal stomata: ' + str(len(stomata_punten[str(int(thr * 100))])))

    widgets.interact(change_threshold, thr=widgets.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.05, continuous_update=False))
