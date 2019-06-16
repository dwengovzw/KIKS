from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import os

im_height = 250

image_dir = '../.images/IntroductieDeepLearning'
style_dir = '../.style'

font_location = os.path.join(style_dir, 'BebasNeue-Regular.ttf')

stoma_im = Image.open(os.path.join(image_dir, 'stoma.jpg'))
conv_layer_im = Image.open(os.path.join(image_dir, 'conv_layer.jpg'))
tussen_layer_im = Image.open(os.path.join(image_dir, 'tussen_layer.jpg'))
ff_layer_im = Image.open(os.path.join(image_dir, 'feedforward_layer.jpg'))
last_layer_im = Image.open(os.path.join(image_dir, 'last_layer.jpg'))


def add_padding(im, height):
    im_r = np.asarray(im)
    padding_height = int((height - im_r.shape[0]) / 2)
    new_im_r = np.vstack(
        [np.full(shape=((padding_height if ((height - im_r.shape[0]) % 2 == 0) else padding_height + 1), im_r.shape[1], im_r.shape[2]), fill_value=255),
         im_r,
         np.full(shape=(padding_height, im_r.shape[1], im_r.shape[2]), fill_value=255)])
    new_im = Image.fromarray(new_im_r.astype(np.uint8))
    return new_im


def resize_keep_ratio(im, height):
    new_width = int(im.size[0] * (height / im.size[1]))
    new_im = im.resize((new_width, height), Image.ANTIALIAS)
    return new_im


def preprocess_image(im):
    new_im = resize_keep_ratio(im, int(im_height * 0.7))
    return new_im


def calculate_point(full_im_width, im_size, width_perc, height_perc, extra_width=0, extra_height=0):
    added_width = full_im_width
    added_height = math.ceil((im_height - im_size[1]) / 2)

    point = [int(im_size[0] * width_perc) + added_width + extra_width, int(im_size[1] * height_perc) + added_height + extra_height]
    return point


def get_layers_list(base, n_conv_layers, dropout, n_ff_layers, n_ff_input):
    layers = []
    if base == 'VGG19_base':
        layers.append({'name': 'conv', 'filters': 64, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 64, 'maxpool': True})
        layers.append({'name': 'conv', 'filters': 128, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 128, 'maxpool': True})
        layers.append({'name': 'conv', 'filters': 256, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 256, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 256, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 256, 'maxpool': True})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': True})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': False})
        layers.append({'name': 'conv', 'filters': 512, 'maxpool': True})
    elif base == 'conv_base':
        filters = 32
        for i in range(n_conv_layers):
            layers.append({'name': 'conv', 'filters': filters, 'maxpool': True})
            filters *= 2

    neurons = n_ff_input
    for i in range(n_ff_layers):
        layers.append({'name': 'feedforward', 'neurons': neurons, 'dropout': dropout})
        neurons = int(neurons / 2)

    return layers


def get_network_image(base='conv_base', n_conv_layers=1, dropout=False, n_ff_layers=1, n_ff_input=16):
    layers = get_layers_list(base, n_conv_layers, dropout, n_ff_layers, n_ff_input)

    font = ImageFont.truetype(font_location, size=int(im_height * (12 / 200)))

    conv_layer_im_local = preprocess_image(conv_layer_im)
    stoma_im_local = preprocess_image(stoma_im)
    ff_layer_im_local = preprocess_image(ff_layer_im)
    tussen_layer_local = preprocess_image(tussen_layer_im)

    full_im_r = np.hstack([np.asarray(add_padding(stoma_im_local, im_height)), np.asarray(add_padding(tussen_layer_local, im_height))])
    full_im = Image.fromarray(full_im_r)

    square_width = int(im_height * (10 / 200))
    square_left_top = [stoma_im_local.size[0] * 0.2, stoma_im_local.size[1] * 0.35 + math.ceil((im_height - stoma_im_local.size[1]) / 2)]
    square_right_bottom = [square_left_top[0] + square_width, square_left_top[1] + square_width]

    for layer in layers:
        full_im_width = full_im.size[0]
        if layer['name'] == 'conv':
            dest_point = calculate_point(full_im_width, conv_layer_im_local.size, 0.6, 0.7)

            demension_location = calculate_point(full_im_width, conv_layer_im_local.size, 0.60, 1, extra_height=5)

            full_im_r = np.hstack([np.asarray(full_im), np.asarray(add_padding(conv_layer_im_local, im_height)), np.asarray(add_padding(tussen_layer_local, im_height))])
            full_im = Image.fromarray(full_im_r)

            draw = ImageDraw.Draw(full_im)
            draw.rectangle([tuple(square_left_top), tuple(square_right_bottom)], outline=0)
            draw.line([square_left_top[0] + (square_right_bottom[0] - square_left_top[0]), square_left_top[1], dest_point[0],dest_point[1]], fill=0, width=2)
            draw.line([square_right_bottom[0], square_right_bottom[1], dest_point[0], dest_point[1]], fill=0, width=2)
            draw.text(demension_location, str(layer['filters']), 0, font=font)

            square_left_top = calculate_point(full_im_width, conv_layer_im_local.size, 0.7, 0.8)
            square_right_bottom = [square_left_top[0] + square_width, square_left_top[1] + square_width]

            if layer['maxpool']:
                pooling_location = calculate_point(full_im_width + conv_layer_im_local.size[0], tussen_layer_local.size, 0, 0.25)
                draw.text(pooling_location, 'max\npooling', 0, font=font)
                conv_layer_im_local = resize_keep_ratio(conv_layer_im_local, int(conv_layer_im_local.size[1] * 0.7))

        elif layer['name'] == 'feedforward':
            demension_location = calculate_point(full_im_width, ff_layer_im_local.size, 0, 1, extra_height=5)

            full_im_r = np.hstack([np.asarray(full_im), np.asarray(add_padding(ff_layer_im_local, im_height)), np.asarray(add_padding(tussen_layer_local, im_height))])
            full_im = Image.fromarray(full_im_r)

            draw = ImageDraw.Draw(full_im)
            draw.text(demension_location, str(layer['neurons']), 0, font=font)
            if layer['dropout']:
                dropout_location = calculate_point(full_im_width + ff_layer_im_local.size[0], tussen_layer_local.size, 0, 0.35, extra_width=2)
                draw.text(dropout_location, 'dropout', 0, font=font)

            ff_layer_im_local = ff_layer_im_local.resize((ff_layer_im_local.size[0], int(ff_layer_im_local.size[1] / 2)), Image.ANTIALIAS)

    full_im_r = np.hstack([np.asarray(full_im), np.asarray(add_padding(last_layer_im, im_height))])
    full_im = Image.fromarray(full_im_r)

    return full_im
