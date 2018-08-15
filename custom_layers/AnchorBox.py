import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
from bounding_box_utils.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    def __init__(
        self,
        img_h,
        img_w,
        this_scale,
        next_scale,
        aspect_ratios,
        two_boxes_for_ar1,
        steps,
        offsets,
        variances,
        coordinates="centroid",
        norm_coordinates=False,
        **kwargs
    ):
        # sanity checks
        if this_scale < 0 or next_scale < 0 or this_scale > 1:
            raise ValueError(
                "The parameter 'this_scale' must belong to the interval [0, 1] while the parameter 'next_scale' must "
                "be > 0. You passed this_scale = {} and next_scale = {}".format(
                    this_scale,
                    next_scale
                )
            )

        # initialize members
        self.img_h = img_h
        self.img_w = img_w
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.steps = steps
        self.offsets = offsets

        self.variances = variances
        self.coordinates = coordinates
        self.norm_coordinates = norm_coordinates

        # calculate the number of anchor boxes for each cell
        self.n_boxes = len(self.aspect_ratios) + (1 if (self.two_boxes_for_ar1 and 1 in self.aspect_ratios) else 0)

        # initialize
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        # the shorter side of the image is used to compute the width and height of the anchor boxes using the scale and
        # aspect ratios
        size = min(self.img_h, self.img_w)

        # compute the anchor box widths / heights for each of the aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            height = size * self.this_scale / np.sqrt(ar)
            width = size * self.this_scale * np.sqrt(ar)
            wh_list.append([height, width])

        if (1 in self.aspect_ratios) and self.two_boxes_for_ar1:
            height = width = size * np.sqrt(self.this_scale * self.next_scale)
            wh_list.append([height, width])

        wh_list = np.array(wh_list)

        # get the shape of the input tensor
        batch_size, feature_map_h, feature_map_w, n_channels = x._keras_shape

        # compute the grid points
        step_h = None
        step_w = None
        if self.steps is None:
            step_h = self.img_h / feature_map_h
            step_w = self.img_w / feature_map_w
        else:
            # otherwise steps are specified. We have two acceptable inputs here:
            # 1. we are given both step_h and step_w
            # 2. we are given a single value which we apply to both step_h and step_w
            if isinstance(self.steps, (list, tuple)):
                if len(self.steps) == 2:
                    step_h, step_w = self.steps
                else:
                    raise ValueError(
                        "Expected two values to be contained in the input 'steps', received {}.".format(
                            len(self.steps)
                        )
                    )
            elif isinstance(self.steps, (int, float)):
                step_h = step_w = self.steps

        offset_h = None
        offset_w = None
        if self.offsets is None:
            offset_h = offset_w = 0.5
        else:
            if isinstance(self.offsets, (list, tuple)):
                if len(self.offsets) == 2:
                    offset_h = offset_w = self.offsets
                else:
                    raise ValueError(
                        "Expected two values to be contained in the input 'offsets', received {}.".format(
                            len(self.offsets)
                        )
                    )
            elif isinstance(self.offsets, (int, float)):
                offset_h = offset_w = self.offsets

        # we have the offsets and step sizes. Time to generate the grid.
        cy = np.linspace(start=offset_h * step_h, stop=(offset_h + feature_map_h - 1) * step_h, num=feature_map_h)
        cx = np.linspace(start=offset_w * step_w, stop=(offset_w + feature_map_w - 1) * step_w, num=feature_map_w)

        # create the grid and extend the dimensions of each of the axes so we can tile them to get the anchor boxes
        # tensor
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # now create the array that's going to hold the anchor boxes
        boxes_tensor = np.zeros((feature_map_h, feature_map_w, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        # normalize coordinates if necessary
        if self.norm_coordinates:
            # convert to corners
            boxes_tensor = convert_coordinates(boxes_tensor, conversion_type="%s2corners" % self.coordinates)

            # normalize
            boxes_tensor[:, :, :, [0, 2]] /= self.img_w
            boxes_tensor[:, :, :, [1, 3]] /= self.img_h

            # convert back to centroid format
            boxes_tensor = convert_coordinates(boxes_tensor, conversion_type="corners2%s" % self.coordinates)

        # create a tensor to hold the variances
        variances_tensor = np.zeros_like(boxes_tensor)  # shape (fm_h, fm_w, n_boxes, 4)
        variances_tensor += self.variances

        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # now add a dimension to boxes_tensor. We need to have a copy of the 'current' boxes_tensor for each element
        # in the batch
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(
            K.constant(boxes_tensor, dtype="float32"),
            (K.shape(x)[0], 1, 1, 1, 1)
        )

        return boxes_tensor
