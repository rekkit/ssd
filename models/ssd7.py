# general imports
import numpy as np
from custom_layers.AnchorBox import AnchorBoxes

# Keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

class SSD7(object):
    def __init__(
        self,
        image_size,
        n_classes,
        mode="training",
        min_scale=0.1,
        max_scale=0.9,
        scales=None,
        aspect_ratios_global=[0.5, 1, 2],
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=True,
        steps=None,
        offsets=None,
        clip_boxes=False,
        variances=[1, 1, 1, 1],
        coordinates="centroids",
        norm_coordinates=False,
        mean_to_subtract=None,
        std_to_divide_by=None
    ):
        # initialize members
        self.img_h, self.img_w, self.img_channels = image_size
        self.img_shape = image_size  # also keep the tuple for brevity's sake
        self.n_classes = n_classes + 1  # +1 because we also have a background class
        self.mode = mode,
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.steps = steps
        self.offsets = offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coordinates = coordinates
        self.norm_coordinates = norm_coordinates
        self.mean_to_subtract = mean_to_subtract
        self.std_to_divide_by = std_to_divide_by
        self.predictor_sizes = None
        self.model = None

        # internal variables
        self.__allowed_modes = ["training"]
        self.__allowed_coordinates = ["centroids", "corners", "minmax"]

        # since the architecture of the NN is fixed for now, we can also set the number of predictor layers
        self.n_layers = 4

        # sanity checks
        if self.mode not in self.__allowed_modes:
            raise ValueError(
                "The parameter 'mode' can take on the values {}, but the value {} was passed".format(
                    ", ".join(self.__allowed_modes[:-1]) + " and {}".format(self.__allowed_modes[-1])
                    if len(self.__allowed_modes) > 1
                    else self.__allowed_modes[0],
                    self.mode
                )
            )

        if self.scales is None and (self.min_scale is None or self.max_scale is None):
            raise ValueError(
                "If the parameter 'scales' is None, the parameters 'min_scale' and 'max_scale' must both be numbers."
            )
        if self.scales:
            if len(self.scales) != self.n_layers + 1:
                raise ValueError(
                    "The parameter 'scales' must either be None or an iterable of length {}.".format(
                        self.n_layers + 1
                    )
                )
        else:
            self.scales = np.linspace(self.min_scale, self.max_scale, self.n_layers + 1)

        if self.aspect_ratios_global is None and self.aspect_ratios_per_layer is None:
            raise ValueError(
                "The parameters 'aspect_ratios_global' and 'aspect_ratios_per_layer' can not both be None at the same "
                "time"
            )

        if (self.steps is not None) and (len(self.steps) != self.n_layers):
            raise ValueError("You must provide at least one step value for each predictor layer.")

        if (self.offsets is not None) and (len(self.offsets) != self.n_layers):
            raise ValueError("You must provide at least one offset value for each predictor layer.")

        # compute the anchor box parameters
        if self.aspect_ratios_per_layer:
            self.aspect_ratios = self.aspect_ratios_per_layer
        else:
            self.aspect_ratios = [self.aspect_ratios_global] * self.n_layers

        # get the number of anchor boxes that we will have in each layer
        self.n_boxes = []
        for ar in self.aspect_ratios:
            self.n_boxes.append(len(ar))
            if (1 in ar) and self.two_boxes_for_ar1:
                self.n_boxes[-1] += 1

        # steps and offsets
        if self.steps is None:
            self.steps = [None] * self.n_layers

        if self.offsets is None:
            self.offsets = [None] * self.n_layers

    def identity_layer(self, tensor):
        return tensor

    def input_mean_normalize(self, tensor):
        return tensor - np.array(self.mean_to_subtract)

    def input_std_normalize(self, tensor):
        return tensor / np.array(self.std_to_divide_by)

    def initialize_network(self):
        x = Input(shape=self.img_shape)

        x1 = Lambda(function=self.identity_layer, output_shape=self.img_shape, name="identity_layer")(x)

        # normalize if necessary
        if self.mean_to_subtract is not None:
            x1 = Lambda(self.input_mean_normalize, output_shape=self.img_shape, name="input_mean_normalization")(x1)
        if self.std_to_divide_by is not None:
            x1 = Lambda(self.input_std_normalize, output_shape=self.img_shape, name="input_std_normalization")(x1)

        # the actual neural network
        conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="same", name='conv1')(x1)
        conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
        conv1 = ELU(name='elu1')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

        conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", name='conv2')(pool1)
        conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
        conv2 = ELU(name='elu2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

        conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name='conv3')(pool2)
        conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
        conv3 = ELU(name='elu3')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

        conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name='conv4')(pool3)
        conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
        conv4 = ELU(name='elu4')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

        conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", name='conv5')(pool4)
        conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
        conv5 = ELU(name='elu5')(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

        conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", name='conv6')(pool5)
        conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
        conv6 = ELU(name='elu6')(conv6)
        pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

        conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name='conv7')(pool6)
        conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
        conv7 = ELU(name='elu7')(conv7)

        # now we add predictor layers on top of each of the last four layers of the NN
        # classes
        classes4 = Conv2D(filters=self.n_boxes[0] * self.n_classes, kernel_size=(3, 3), padding="same")(conv4)
        classes5 = Conv2D(filters=self.n_boxes[1] * self.n_classes, kernel_size=(3, 3), padding="same")(conv5)
        classes6 = Conv2D(filters=self.n_boxes[2] * self.n_classes, kernel_size=(3, 3), padding="same")(conv6)
        classes7 = Conv2D(filters=self.n_boxes[3] * self.n_classes, kernel_size=(3, 3), padding="same")(conv7)

        # anchor box offsets
        boxes4 = Conv2D(filters=self.n_boxes[0] * 4, kernel_size=(3, 3), padding="same")(conv4)
        boxes5 = Conv2D(filters=self.n_boxes[1] * 4, kernel_size=(3, 3), padding="same")(conv5)
        boxes6 = Conv2D(filters=self.n_boxes[2] * 4, kernel_size=(3, 3), padding="same")(conv6)
        boxes7 = Conv2D(filters=self.n_boxes[3] * 4, kernel_size=(3, 3), padding="same")(conv7)

        # anchor box placeholders
        anchors4 = AnchorBoxes(
            img_h=self.img_h, img_w=self.img_w, this_scale=self.scales[0], next_scale=self.scales[1],
            aspect_ratios=self.aspect_ratios, two_boxes_for_ar1=self.two_boxes_for_ar1, steps=self.steps,
            offsets=self.offsets, variances=self.variances, coordinates=self.coordinates,
            norm_coordinates=self.norm_coordinates
        )(boxes4)

        anchors5 = AnchorBoxes(
            img_h=self.img_h, img_w=self.img_w, this_scale=self.scales[1], next_scale=self.scales[2],
            aspect_ratios=self.aspect_ratios, two_boxes_for_ar1=self.two_boxes_for_ar1, steps=self.steps,
            offsets=self.offsets, variances=self.variances, coordinates=self.coordinates,
            norm_coordinates=self.norm_coordinates
        )(boxes5)

        anchors6 = AnchorBoxes(
            img_h=self.img_h, img_w=self.img_w, this_scale=self.scales[2], next_scale=self.scales[3],
            aspect_ratios=self.aspect_ratios, two_boxes_for_ar1=self.two_boxes_for_ar1, steps=self.steps,
            offsets=self.offsets, variances=self.variances, coordinates=self.coordinates,
            norm_coordinates=self.norm_coordinates
        )(boxes6)

        anchors7 = AnchorBoxes(
            img_h=self.img_h, img_w=self.img_w, this_scale=self.scales[3], next_scale=self.scales[4],
            aspect_ratios=self.aspect_ratios, two_boxes_for_ar1=self.two_boxes_for_ar1, steps=self.steps,
            offsets=self.offsets, variances=self.variances, coordinates=self.coordinates,
            norm_coordinates=self.norm_coordinates
        )(boxes7)

        # now we have to reshape all of these tensors to be three-dimensional:
        # (batch_size, filter_map_h * filter_map_w * n_channels, n_classes)
        classes4 = Reshape(target_shape=(-1, self.n_classes))(classes4)
        classes5 = Reshape(target_shape=(-1, self.n_classes))(classes5)
        classes6 = Reshape(target_shape=(-1, self.n_classes))(classes6)
        classes7 = Reshape(target_shape=(-1, self.n_classes))(classes7)

        # (batch_size, filter_map_h * filter_map_w * n_channels, 4 (predicted bounding boxes))
        boxes4 = Reshape(target_shape=(-1, 4))(boxes4)
        boxes5 = Reshape(target_shape=(-1, 4))(boxes5)
        boxes6 = Reshape(target_shape=(-1, 4))(boxes6)
        boxes7 = Reshape(target_shape=(-1, 4))(boxes7)

        # (batch_size, filter_map_h * filter_map_w * n_channels, 4 + 4 (anchors + variances))
        anchors4 = Reshape(target_shape=(-1, 8))(anchors4)
        anchors5 = Reshape(target_shape=(-1, 8))(anchors5)
        anchors6 = Reshape(target_shape=(-1, 8))(anchors6)
        anchors7 = Reshape(target_shape=(-1, 8))(anchors7)

        # concatenate the tensors
        # for each predictive layer we have class and offset predictions, as well as anchor boxes
        # we want to join all these along their dimension
        classes_concat = Concatenate(axis=1, name="classes_concat")(
            [
                classes4, classes5, classes6, classes7
            ]
        )

        boxes_concat = Concatenate(axis=1, name="boxes_concat")(
            [
                boxes4, boxes5, boxes6, boxes7
            ]
        )

        anchors_concat = Concatenate(axis=1, name="anchors_concat")(
            [
                anchors4, anchors5, anchors6, anchors7
            ]
        )

        # apply the softmax function to the classes
        classes_softmax = Activation("softmax", name="classes_softmax")(classes_concat)

        # now define the 'predictions' tensor
        predictions = Concatenate(axis=2, name="predictions")(
            [
                classes_softmax, boxes_concat, anchors_concat
            ]
        )

        # save the output shape
        self.predictor_sizes = np.array(
            [
                classes4._keras_shape[1:3],
                classes5._keras_shape[1:3],
                classes6._keras_shape[1:3],
                classes7._keras_shape[1:3]
            ]
        )

        # define the model
        self.model = Model(inputs=x, outputs=predictions)

    def get_model(self, return_predictor_sizes):
        if return_predictor_sizes:
            return self.model, self.predictor_sizes
        else:
            return self.model

    def compile_model(self, optimizer, loss):
        self.model.compile(
            optimizer=optimizer,
            loss=loss
        )
