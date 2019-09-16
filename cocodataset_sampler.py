import os
import sys
import random
from random import shuffle
from pycocotools.coco import COCO
import json
import argparse

random.seed(123)
HERE = os.path.dirname(os.path.realpath(__file__))


class CocoDatasetFilter:
    @staticmethod
    def run(coco, image_ids_to_keep):
        dataset = {'info': coco.dataset['info'],
                   'images': [image for image in coco.dataset['images'] if image['id'] in image_ids_to_keep],
                   'licenses': coco.dataset['licenses']
                   }

        if 'annotations' in coco.dataset:
            dataset['annotations'] = [sample for sample in coco.dataset['annotations'] if
                                      sample['image_id'] in image_ids_to_keep]

        if 'categories' in coco.dataset:
            dataset['categories'] = coco.dataset['categories']
        return dataset


class CocoDatasetAnnotationsWriter():
    def __init__(self, directory):
        self.directory = directory

        if not os.path.exists(directory):
            raise Exception("Directory {} does not exist".format(directory))
        if len(os.listdir(directory)) != 0:
            raise Exception("Directory {} is not empty".format(directory))

        self.annotations_dir = os.path.join(self.directory, 'annotations')
        os.mkdir(self.annotations_dir)

    def run(self, coco, filename):
        annotations_file = os.path.join(self.annotations_dir, filename)
        with open(annotations_file, 'w') as f:
            f.write(json.dumps(coco))

    def generate_cp_image_file(self, coco, orig_image_dir, to_dir, name):
        cp_images_file = os.path.join(self.directory, name)
        with open(cp_images_file, 'w') as f:
            for image in coco['images']:
                f.write('cp {}/{} {}/{}/{}\n'.format(orig_image_dir, image['file_name'], self.directory, to_dir,
                                                     image['file_name']))


class CocoDatasetImageIdSampler:
    def __init__(self, coco, n_per_category):
        self.coco = coco
        self.categories = [c['id'] for c in coco.dataset['categories']]
        self.n_per_category = n_per_category

    @staticmethod
    def get_image_ids_to_keep_from_test(coco_test, n):
        tmp = [image['id'] for image in coco_test.dataset['images']]
        shuffle(tmp)
        return set(tmp[:n])

    @staticmethod
    def _take_n_not_from_x_not_in_y(n, x, y):
        i = n
        taken = set()
        while len(taken) != n:
            current = set(x[:i])
            taken = current.difference(y)
            i += 1
            if i >= len(x):
                raise Exception("Cannot take {} elements from x that are not in y yet.".format(n))
        return taken

    def get_image_ids_to_keep(self):
        image_ids_by_category_ids = dict()
        for c in self.categories:
            image_ids_by_category_ids[c] = set()

        for instance in self.coco.dataset['annotations']:
            category_id = instance['category_id']
            image_ids_by_category_ids[category_id].add(instance['image_id'])

        image_ids_to_keep = set()
        for category_id, image_ids in image_ids_by_category_ids.items():
            image_ids_as_list = list(image_ids)
            shuffle(image_ids_as_list)
            first_n = CocoDatasetImageIdSampler._take_n_not_from_x_not_in_y(self.n_per_category, image_ids_as_list,
                                                                            image_ids_to_keep)
            image_ids_to_keep = image_ids_to_keep.union(set(first_n))

        return image_ids_to_keep


def run_for_2014(annotations_dir, image_dir, out_dir):
    instances_val = os.path.join(annotations_dir, 'instances_val2014.json')
    instances_train = os.path.join(annotations_dir, 'instances_train2014.json')
    captions_val = os.path.join(annotations_dir, 'captions_val2014.json')
    captions_train = os.path.join(annotations_dir, 'captions_train2014.json')
    person_keypoints_val = os.path.join(annotations_dir, 'person_keypoints_val2014.json')
    person_keypoints_train = os.path.join(annotations_dir, 'person_keypoints_train2014.json')
    orig_image_test = os.path.join(image_dir, 'test2014')
    orig_image_val = os.path.join(image_dir, 'val2014')
    orig_image_train = os.path.join(image_dir, 'train2014')
    image_info_test = os.path.join(annotations_dir, 'image_info_test2014.json')

    coco_writer = CocoDatasetAnnotationsWriter(out_dir)

    coco_test = COCO(image_info_test)
    image_ids_to_keep_from_test = CocoDatasetImageIdSampler.get_image_ids_to_keep_from_test(coco_test, 100)

    filtered_image_info_test = CocoDatasetFilter.run(coco_test, image_ids_to_keep_from_test)
    coco_writer.run(filtered_image_info_test, 'image_info_test2014_10_per_category.json')
    coco_writer.generate_cp_image_file(filtered_image_info_test, orig_image_test, 'test2014', 'cp_image_test2014.sh')

    coco_val = COCO(instances_val)
    sampler_val = CocoDatasetImageIdSampler(coco_val, 10)
    image_ids_to_keep_from_val = sampler_val.get_image_ids_to_keep()

    filtered_instances_val = CocoDatasetFilter.run(coco_val, image_ids_to_keep_from_val)
    filtered_captions_val = CocoDatasetFilter.run(COCO(captions_val), image_ids_to_keep_from_val)
    filtered_person_keypoints_val = CocoDatasetFilter.run(COCO(person_keypoints_val),
                                                          image_ids_to_keep_from_val)

    coco_writer.run(filtered_instances_val, 'instances_val2014_10_per_category.json')
    coco_writer.run(filtered_captions_val, 'captions_val2014_10_per_category.json')
    coco_writer.run(filtered_person_keypoints_val, 'person_keypoints_val2014_10_per_category.json')

    coco_writer.generate_cp_image_file(filtered_instances_val, orig_image_val, 'val2014', 'cp_image_val2014.sh')

    coco_train = COCO(instances_train)
    sampler_train = CocoDatasetImageIdSampler(coco_train, 10)
    image_ids_to_keep_from_train = sampler_train.get_image_ids_to_keep()

    filtered_instances_train = CocoDatasetFilter.run(coco_train, image_ids_to_keep_from_train)
    filtered_captions_train = CocoDatasetFilter.run(COCO(captions_train), image_ids_to_keep_from_train)
    filtered_person_keypoints_train = CocoDatasetFilter.run(COCO(person_keypoints_train),
                                                            image_ids_to_keep_from_train)

    coco_writer.run(filtered_instances_train, 'instances_train2014_10_per_category.json')
    coco_writer.run(filtered_captions_train, 'captions_train2014_10_per_category.json')
    coco_writer.run(filtered_person_keypoints_train, 'person_keypoints_train2014_10_per_category.json')

    coco_writer.generate_cp_image_file(filtered_instances_train, orig_image_train, 'train2014', 'cp_image_train2014.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cocodataset-sampler')

    parser.add_argument('--annotations-dir', dest='annotations_dir', type=str,
                        help='where to find the annotations\' json files')
    parser.add_argument('--image-dir', dest='image_dir', type=str,
                        help='where to find the image files')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        help='where to write to')

    args = parser.parse_args()
    print(args)
    if (args.annotations_dir is None or args.image_dir is None or args.out_dir is None):
        parser.print_help()

    annotations_dir = args.annotations_dir
    run_for_2014(annotations_dir=args.annotations_dir, image_dir=args.image_dir, out_dir=args.out_dir)
