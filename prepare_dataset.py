import os
from glob import glob
import cv2
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def process_single_image(line, output_file):
    try:
        line = line.strip().replace('\n', '')
        txt_path = line.replace(".jpeg", ".txt").replace(".jpg", ".txt").replace(".png", '.txt')
        image = cv2.imread(line)
        if image is None:
            return None, f"Error: Unable to read image {line}"

        image_h, image_w, _ = image.shape
        with open(txt_path, 'r') as f:
            annotation = [[float(y) for y in x.split()] for x in f.read().strip().split('\n')]

        annotation = np.array(annotation)
        annotation_shape = annotation.shape[0]
        classes = annotation[:, 0]
        center_points = annotation[:, 1:3]
        sizes = annotation[:, 3:5]
        coordinates = annotation[:, 5:]
        coordinates = coordinates[:, [i for i in range(15) if (i + 1) % 3 != 0]]

        coordinates[:, ::2] *= image_w
        coordinates[:, 1::2] *= image_h

        sizes[:, ::2] *= image_w
        sizes[:, 1::2] *= image_h

        center_points[:, ::2] *= image_w
        center_points[:, 1::2] *= image_h

        write_text = f"# {line} \n"

        for index in range(annotation_shape):
            lt, rt, cp, lb, rb = coordinates[index].reshape(5, 2)
            w, h = sizes[index]
            _cp_x, _cp_y = center_points[index]
            cls_ = classes[index]
            in_txt = f"{lt[0]} {lt[1]} {w} {h} {lt[0]} {lt[1]} {rt[0]} {rt[1]} {cp[0]} {cp[1]} {lb[0]} {lb[1]} {rb[0]} {rb[1]} {int(cls_)}\n"
            write_text += in_txt

        return write_text, None
    except Exception as e:
        return None, f"Error processing {line}: {str(e)}"


def prepare_dataset(input_file, output_file, num_cores=128):
    with open(input_file, 'r') as f:
        content = f.readlines()

    total_images = len(content)
    print(f"Total images to process: {total_images}")

    pool = mp.Pool(num_cores)
    process_func = partial(process_single_image, output_file=output_file)

    results = []
    errors = []
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for result, error in pool.imap_unordered(process_func, content):
            if error:
                errors.append(error)
            elif result:
                results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    print("Writing results to file...")
    with open(output_file, 'w') as outTxt:
        for result in results:
            outTxt.write(result)

    print(f'Annotations for {len(results)} images recorded')

    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors[:10]:  # Print first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more.")

        # Optionally, write errors to a file
        with open('processing_errors.log', 'w') as error_file:
            for error in errors:
                error_file.write(f"{error}\n")
        print("Full error log written to 'processing_errors.log'")


if __name__ == '__main__':
    original_data = '/mnt/data/all_filenames.txt'
    out_file_path = '/mnt/data/retina_label.txt'
    prepare_dataset(input_file=original_data, output_file=out_file_path, num_cores=128)