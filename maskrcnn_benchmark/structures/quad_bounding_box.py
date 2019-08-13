# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class QuadBoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, quad_bbox, image_size, mode="xyxy"):
        device = quad_bbox.device if isinstance(quad_bbox, torch.Tensor) else torch.device("cpu")
        quad_bbox = torch.as_tensor(quad_bbox, dtype=torch.float32, device=device)
        if quad_bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(quad_bbox.ndimension())
            )
        if quad_bbox.size(-1) != 8:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 8, got {}".format(quad_bbox.size(-1))
            )
        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        self.device = device
        self.quad_bbox = quad_bbox
        self.bbox = self.quad_bbox_to_bbox()
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def quad_bbox_to_bbox(self):
        bbox = torch.zeros((self.quad_bbox.shape[0], 4))
        if self.quad_bbox.shape[0] == 0:
            return bbox.to(self.device)
        bbox[:, 0], _ = torch.min(self.quad_bbox[:, 0::2], 1)
        bbox[:, 1], _ = torch.min(self.quad_bbox[:, 1::2], 1)
        bbox[:, 2], _ = torch.max(self.quad_bbox[:, 0::2], 1)
        bbox[:, 3], _ = torch.max(self.quad_bbox[:, 1::2], 1)
        return bbox.to(self.device)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        if mode == self.mode:
            return self


    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.quad_bbox * ratio
        else:
            ratio_width, ratio_height = ratios
            scaled_box = self.quad_bbox
            scaled_box[:, 0::2] *= ratio_width
            scaled_box[:, 1::2] *= ratio_height
        bbox = QuadBoxList(scaled_box, size, mode=self.mode)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox


    # def transpose(self, method):
    #     """
    #     Transpose bounding box (flip or rotate in 90 degree steps)
    #     :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
    #       :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
    #       :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
    #       :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
    #     """
    #     if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
    #         raise NotImplementedError(
    #             "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
    #         )
    #
    #     image_width, image_height = self.size
    #     xmin, ymin, xmax, ymax = self._split_into_xyxy()
    #     if method == FLIP_LEFT_RIGHT:
    #         TO_REMOVE = 1
    #         transposed_xmin = image_width - xmax - TO_REMOVE
    #         transposed_xmax = image_width - xmin - TO_REMOVE
    #         transposed_ymin = ymin
    #         transposed_ymax = ymax
    #     elif method == FLIP_TOP_BOTTOM:
    #         transposed_xmin = xmin
    #         transposed_xmax = xmax
    #         transposed_ymin = image_height - ymax
    #         transposed_ymax = image_height - ymin
    #
    #     transposed_boxes = torch.cat(
    #         (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
    #     )
    #     bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
    #     # bbox._copy_extra_fields(self)
    #     for k, v in self.extra_fields.items():
    #         if not isinstance(v, torch.Tensor):
    #             v = v.transpose(method)
    #         bbox.add_field(k, v)
    #     return bbox.convert(self.mode)

    # def crop(self, box):
    #     """
    #     Cropss a rectangular region from this bounding box. The box is a
    #     4-tuple defining the left, upper, right, and lower pixel
    #     coordinate.
    #     """
    #     xmin, ymin, xmax, ymax = self._split_into_xyxy()
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
    #     cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
    #     cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
    #     cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
    #
    #     # TODO should I filter empty boxes here?
    #     if False:
    #         is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
    #
    #     cropped_box = torch.cat(
    #         (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
    #     )
    #     bbox = BoxList(cropped_box, (w, h), mode="xyxy")
    #     # bbox._copy_extra_fields(self)
    #     for k, v in self.extra_fields.items():
    #         if not isinstance(v, torch.Tensor):
    #             v = v.crop(box)
    #         bbox.add_field(k, v)
    #     return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = QuadBoxList(self.quad_bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = QuadBoxList(self.quad_bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.quad_bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.quad_bbox[:, 0::2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.quad_bbox[:, 1::2].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = QuadBoxList(self.quad_bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = QuadBoxList([[0, 0, 10, 10, 0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
