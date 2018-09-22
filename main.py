import numpy as np
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization, Add
from keras.layers import Input
from keras.models import Model

from VIPPruning import VIPPruning

def rebuild_net(model=None, layer_filters=[]):
    n_discarded_filters = 0
    total_filters = 0
    model = model
    inp = (model.inputs[0].shape.dims[1].value,
           model.inputs[0].shape.dims[2].value,
           model.inputs[0].shape.dims[3].value)

    H = Input(inp)
    inp = H
    idx_previous = []

    for i in range(1, len(model.layers)):

        layer = model.get_layer(index=i)
        config = layer.get_config()

        if isinstance(layer, MaxPooling2D):
            H = MaxPooling2D.from_config(config)(H)

        if isinstance(layer, Dropout):
            H = Dropout.from_config(config)(H)

        if isinstance(layer, Activation):
            H = Activation.from_config(config)(H)

        if isinstance(layer, BatchNormalization):
            weights = layer.get_weights()
            weights[0] = np.delete(weights[0], idx_previous)
            weights[1] = np.delete(weights[1], idx_previous)
            weights[2] = np.delete(weights[2], idx_previous)
            weights[3] = np.delete(weights[3], idx_previous)
            H = BatchNormalization(weights=weights)(H)

        elif isinstance(layer, Conv2D):
            weights = layer.get_weights()

            n_filters = weights[0].shape[3]
            total_filters = total_filters + n_filters

            idxs = [item for item in layer_filters if item[0] == i][0][1]

            weights[0] = np.delete(weights[0], idxs, axis=3)
            weights[1] = np.delete(weights[1], idxs)
            n_discarded_filters += len(idxs)
            if len(idx_previous) != 0:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)

            config['filters'] = weights[1].shape[0]
            H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       # config=config['config'],
                       # scale=config['scale'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)

        idx_previous = idxs
    print('Percentage of discarded filters {}'.format(n_discarded_filters / float(total_filters)))
    return Model(inp, H)

def generate_conv_model(model):

    model = model.get_layer(index=1)

    inp = (model.inputs[0].shape.dims[1].value,
           model.inputs[0].shape.dims[2].value,
           model.inputs[0].shape.dims[3].value)

    H = Input(inp)
    inp = H

    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        config = layer.get_config()

        if isinstance(layer, MaxPooling2D):
            H = MaxPooling2D.from_config(config)(H)

        if isinstance(layer, Dropout):
            H = Dropout.from_config(config)(H)

        if isinstance(layer, Activation):
            H = Activation.from_config(config)(H)

        if isinstance(layer, BatchNormalization):
            weights = layer.get_weights()
            H = BatchNormalization(weights=weights)(H)

        elif isinstance(layer, Conv2D):
            weights = layer.get_weights()

            config['filters'] = weights[1].shape[0]
            H = Conv2D(activation=config['activation'],
                       activity_regularizer=config['activity_regularizer'],
                       bias_constraint=config['bias_constraint'],
                       bias_regularizer=config['bias_regularizer'],
                       data_format=config['data_format'],
                       dilation_rate=config['dilation_rate'],
                       filters=config['filters'],
                       kernel_constraint=config['kernel_constraint'],
                       kernel_regularizer=config['kernel_regularizer'],
                       kernel_size=config['kernel_size'],
                       name=config['name'],
                       padding=config['padding'],
                       strides=config['strides'],
                       trainable=config['trainable'],
                       use_bias=config['use_bias'],
                       weights=weights
                       )(H)

    return Model(inp, H)

def frozen_conv2D(model, trainable=False):
    #Frozen the convolutional layers
    try:
        model.get_layer(index=1).trainable = trainable
        for layer in model.get_layer(index=1).layers:
            layer.trainable = trainable
    except:
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    return model

def insert_fully(cnn_model, input_shape=(32, 32, 3), num_classes=10):

    inp = Input(input_shape)
    H = Model(cnn_model.input, cnn_model.layers[-1].output)
    H = H(inp)

    H = Flatten()(H)
    H = Dense(512)(H)
    H = Activation('relu')(H)
    H = Dropout(0.5)(H)
    H = Dense(num_classes)(H)
    H = Activation('softmax')(H)

    model = keras.models.Model(inp, H)
    return model

def compute_flops(model):
    total_flops =0
    flops_per_layer = []

    for layer_idx in range(1, len(model.layers)):
        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) is True:
            _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

            _, _, _, previous_layer_depth = layer.input_shape
            kernel_H, kernel_W = layer.kernel_size

            flops = output_map_H*output_map_W*previous_layer_depth*current_layer_depth*kernel_H*kernel_W
            total_flops += flops
            flops_per_layer.append(flops)


    # if total_flops / 1e9 > 1:  # for Giga Flops
    #     print(total_flops / 1e9, '{}'.format('GFlops'))
    # else:
    #     print(total_flops / 1e6, '{}'.format('MFlops'))
    return total_flops, flops_per_layer

def count_filters(model):
    n_filters = 0
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) == True:
            config = layer.get_config()
            n_filters+=config['filters']

    return n_filters

if __name__ == '__main__':
    np.random.seed(12227)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    input = Input((32, 32, 3))
    H = Conv2D(32, (3,3), padding='same')(input)
    H = Activation('relu')(H)
    H = Conv2D(32, (3, 3))(H)
    H = Activation('relu')(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)

    H = Conv2D(64, (3, 3), padding='same')(H)
    H = Activation('relu')(H)
    H = Conv2D(64, (3, 3))(H)
    H = Activation('relu')(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)

    H = Flatten()(H)
    H = Dense(512)(H)
    H = Activation('relu')(H)
    H = Dropout(0.5)(H)
    H = Dense(10)(H)
    H = Activation('softmax')(H)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    cnn_model = keras.models.Model([input], H)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    cnn_model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=2)
    y_pred = cnn_model.predict(X_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    n_params = cnn_model.count_params()
    n_filters = count_filters(cnn_model)
    flops, _ = compute_flops(cnn_model)
    print('Original Network. #Parameters [{}] #Filters [{}] FLOPS [{}] Accuracy [{:.4f}]'.format(n_params, n_filters, flops, acc))

    iterations = 5
    p = 0.05

    #We need to remove the layers (i.e., fully connected) from flatten layer
    #to avoid that the rebuild process adds extra layers.
    for i in range(0, 6):
        cnn_model.layers.pop()

    cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    for i in range(0, iterations):

        pruning_method = VIPPruning(n_comp=2, model=cnn_model, representation='max', percentage_discard=p)
        #pruning_method = VIPPruning(n_comp=2, model=cnn_model, representation=MaxPooling2D(pool_size=(2, 2), name='vip_net'), percentage_discard=p)

        idxs = pruning_method.idxs_to_prune(X_train, y_train, [])
        cnn_model = rebuild_net(cnn_model, idxs)

        # First, we fine-tune the Fully Connected Layers only
        cnn_model = frozen_conv2D(cnn_model, False)
        model = insert_fully(cnn_model=cnn_model, num_classes=10)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=2)

        # Finally, we fine-tune the entire network (Fully Connected Layers and Convolutional Layers)
        model = frozen_conv2D(model, True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=45, batch_size=128, verbose=2)

        y_pred = model.predict(X_test)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

        n_params  = model.count_params()
        n_filters = count_filters(model.get_layer(index=1))
        flops, _ = compute_flops(model.get_layer(index=1))
        print('Iteration [{}] #Parameters [{}] #Filters [{}] FLOPS [{}] Accuracy [{:.4f}]'.format(i, n_params, n_filters, flops, acc))

        cnn_model = generate_conv_model(model=model)