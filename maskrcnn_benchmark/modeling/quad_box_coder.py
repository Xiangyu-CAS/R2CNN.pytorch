# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class QuadBoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (8-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes [x1,y1,x2,y2,x3,y3,x4,y4]
            proposals (Tensor): boxes to be encoded   [x1,y1,x2,y2]
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        wx1, wy1, wx2, wy2, wx3, wy3, wx4, wy4 = self.weights

        dx1 = wx1 * (reference_boxes[:, 0] - ex_ctr_x) / ex_widths
        dy1 = wy1 * (reference_boxes[:, 1] - ex_ctr_y) / ex_heights
        dx2 = wx2 * (reference_boxes[:, 2] - ex_ctr_x) / ex_widths
        dy2 = wy2 * (reference_boxes[:, 3] - ex_ctr_y) / ex_heights
        dx3 = wx3 * (reference_boxes[:, 4] - ex_ctr_x) / ex_widths
        dy3 = wy3 * (reference_boxes[:, 5] - ex_ctr_y) / ex_heights
        dx4 = wx4 * (reference_boxes[:, 6] - ex_ctr_x) / ex_widths
        dy4 = wy4 * (reference_boxes[:, 7] - ex_ctr_y) / ex_heights

        targets = torch.stack((dx1, dy1, dx2, dy2,
                               dx3, dy3, dx4, dy4), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
            boxes (Tensor): reference boxes. [x1, y1, x2, y2]
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx1, wy1, wx2, wy2, wx3, wy3, wx4, wy4 = self.weights
        dx1 = rel_codes[:, 0::8] / wx1
        dy1 = rel_codes[:, 1::8] / wy1
        dx2 = rel_codes[:, 2::8] / wx2
        dy2 = rel_codes[:, 3::8] / wy2
        dx3 = rel_codes[:, 4::8] / wx3
        dy3 = rel_codes[:, 5::8] / wy3
        dx4 = rel_codes[:, 6::8] / wx4
        dy4 = rel_codes[:, 7::8] / wy4

        pred_boxes = torch.zeros_like(rel_codes)

        pred_boxes[:, 0::8] = dx1 * widths[:, None] + ctr_x[:, None]
        pred_boxes[:, 1::8] = dy1 * heights[:, None] + ctr_y[:, None]
        pred_boxes[:, 2::8] = dx2 * widths[:, None] + ctr_x[:, None]
        pred_boxes[:, 3::8] = dy2 * heights[:, None] + ctr_y[:, None]
        pred_boxes[:, 4::8] = dx3 * widths[:, None] + ctr_x[:, None]
        pred_boxes[:, 5::8] = dy3 * heights[:, None] + ctr_y[:, None]
        pred_boxes[:, 6::8] = dx4 * widths[:, None] + ctr_x[:, None]
        pred_boxes[:, 7::8] = dy4 * heights[:, None] + ctr_y[:, None]

        return pred_boxes
