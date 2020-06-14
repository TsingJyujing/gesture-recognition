import time

import numpy as np
from PIL import Image, ImageDraw

from gesture_recognizer.pretrained.blazepalm import BlazePalm


def box_test():
    im = Image.open("../data/414076003_highres.jpg").resize((256, 256), box=[110, 0, 536, 426])
    input_data = np.expand_dims(im, axis=0)[:, :, :, :3]
    input_data = (np.float32(input_data) - 127.5) / 127.5
    model = BlazePalm("../model/BlazePalm.tflite", np.loadtxt("../model/BlazePalmAnchors.csv", delimiter=","))

    boxes = model.get_box(input_data, prob_threshold=0.5)

    txt = Image.new('RGBA', im.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt)
    for b in boxes:
        d.rectangle([b.x, b.y, b.x + b.width, b.y + b.height])
        d.point([(p[0], p[1]) for p in b.keypoints])
        for j, p in enumerate(b.keypoints):
            d.text([p[0], p[1]], f"{j}")
    out = Image.alpha_composite(im.convert("RGBA"), txt)

    print(model.extract_tensor(280))


def batch_test(batch_size):
    im = Image.open("../data/414076003_highres.jpg").resize((256, 256), box=[110, 0, 536, 426])
    input_data = np.expand_dims(im, axis=0)[:, :, :, :3]
    input_data = (np.float32(input_data) - 127.5) / 127.5
    batch_input = np.repeat(input_data, batch_size, axis=0)
    model = BlazePalm(
        "../model/BlazePalm.tflite",
        np.loadtxt("../model/BlazePalmAnchors.csv", delimiter=","),
        batch_size=batch_size
    )
    start_tick = time.time()
    model._inference(batch_input)
    print(f"Used {time.time() - start_tick}s infer {batch_size} images")


if __name__ == '__main__':
    box_test()
