import torch
import torchvision
import os, json
from PIL import Image

from maskrcnn_benchmark.structures.quad_bounding_box import QuadBoxList

class IcdarDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "text",)

    def __init__(self, root, ann_file, use_difficult=True, transforms=None):
        self.img_dir = root
        self.ann_file = ann_file
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self.load_annotations()

    def load_annotations(self):
        path = os.path.join(self.ann_file)
        with open(path, 'r') as f:
            self.ids = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        anno = self.ids[idx]
        boxes = [obj["bbox"] for obj in anno['objs'] if self.keep_difficult or not obj['isDifficult']]
        boxes = torch.as_tensor(boxes).reshape(-1, 8)
        target = QuadBoxList(boxes, [anno['width'], anno['height']], mode="xyxy")
        classes = [obj["category_id"] for obj in anno['objs']]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target = target.clip_to_image(remove_empty=False)

        img = Image.open(os.path.join(self.img_dir, anno['img_name'])).convert("RGB")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, idx):
        anno = self.ids[idx]
        return {"height": anno['height'], "width": anno['width']}
