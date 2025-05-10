import os
import urllib.request
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from glob import glob
import shutil
import cv2                      
import numpy as np              
from sklearn.model_selection import train_test_split 


url = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
filename = "bccd.zip"
extract_dir = "bccd_dataset"
dataset_folder_name = "BCCD_Dataset-master"
urllib.request.urlretrieve(url, filename)
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(dir)
bccd_root = os.path.join(extract_dir, dataset_folder_name)
os.listdir(dir)

#Обрезка объектов из изображений BCCD и сохранение в папку classification_data
import cv2
import xml.etree.ElementTree as ET
from glob import glob

images_path = 'bccd_dataset/BCCD_Dataset-master/BCCD/JPEGImages'
annotations_path = 'bccd_dataset/BCCD_Dataset-master/BCCD/Annotations'
output_path = 'classification_data'
# Создаем директории для каждого класса
classes = ['WBC', 'RBC', 'Platelets']
os.makedirs(output_path, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(output_path, cls), exist_ok=True)
# получаем списки файлов изображений и аннотаций
image_files = sorted(glob(os.path.join(images_path, '*.jpg')))
annotation_files = sorted(glob(os.path.join(annotations_path, '*.xml')))
# цикл по всем изображениям и соответствующим аннотациям
for img_path in image_files:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    annot_file = os.path.join(annotations_path, filename + '.xml')
    if not os.path.exists(annot_file):
        continue
    # конвертация BGR->RGB для корректного отображения
    image = cv2.imread(img_path)
    if image is None:
        continue
    # парсим XML-аннотацию и обрезаем каждый объект
    tree = ET.parse(annot_file)
    root = tree.getroot()

    for idx, obj in enumerate(root.findall('object')):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        # координаты ограничивающего прямоугольника
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        cropped = image[ymin:ymax, xmin:xmax]
        if cropped.size == 0:
            continue
        # сохраняем обрезанное изображение
        save_path = os.path.join(output_path, cls_name, f'{filename}_{idx}.jpg')
        cv2.imwrite(save_path, cropped)


# Разделение папки classification_data на train и test по 80/20
source_dir = 'classification_data'
train_dir = 'train'
test_dir = 'test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))


#для 7ой лабоарторной
# Подготовка масок из Pascal VOC аннотаций
images_dir      = 'bccd_dataset/BCCD_Dataset-master/BCCD/JPEGImages'
annotations_dir = 'bccd_dataset/BCCD_Dataset-master/BCCD/Annotations'
masks_dir       = 'masks'
classes         = ['WBC','RBC','Platelets']
os.makedirs(masks_dir, exist_ok=True)
for xml_file in glob(os.path.join(annotations_dir,'*.xml')):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    fn = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
    img_path = os.path.join(images_dir, fn)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h,w),dtype=np.uint8)
    for obj in root.findall('object'):
        cls = obj.find('name').text
        idx = classes.index(cls)+1
        bb = obj.find('bndbox')
        x1,y1 = int(bb.find('xmin').text), int(bb.find('ymin').text)
        x2,y2 = int(bb.find('xmax').text), int(bb.find('ymax').text)
        mask[y1:y2, x1:x2] = idx
    cv2.imwrite(os.path.join(masks_dir, fn.replace('.jpg','.png')), mask)

#для 8ой лабораторной

import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

voc_ann_dir = 'BCCD/Annotations'
voc_img_dir = 'BCCD/JPEGImages'
output_dir = 'dataset_yolo'
classes = ['WBC', 'RBC', 'Platelets']

# Создаем структуру папок
for split in ['train', 'val']:
    os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)


voc_ann_dir = 'bccd_dataset/BCCD_Dataset-master/BCCD/Annotations'
voc_img_dir = 'bccd_dataset/BCCD_Dataset-master/BCCD/JPEGImages'

# Получаем список файлов
all_files = [f[:-4] for f in os.listdir(voc_ann_dir) if f.endswith('.xml')]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

def convert_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    result = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [int(xmlbox.find(tag).text) for tag in ['xmin','ymin','xmax','ymax']]
        xc = (b[0] + b[2]) / 2 / w
        yc = (b[1] + b[3]) / 2 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        result.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return result

# Конвертация и копирование изображений
from shutil import copy2
for split, filelist in zip(['train','val'], [train_files, val_files]):
    for fname in filelist:
        img_src = os.path.join(voc_img_dir, fname + '.jpg')
        label_txt = convert_annotation(os.path.join(voc_ann_dir, fname + '.xml'))
        with open(f"{output_dir}/labels/{split}/{fname}.txt", 'w') as f:
            f.write('\n'.join(label_txt))
        copy2(img_src, f"{output_dir}/images/{split}/{fname}.jpg")
