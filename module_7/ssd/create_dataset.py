import json
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def create_data_lists(annotations_dir, image_dir, output_folder):
    """
    Создает JSON списки, необходимые для обучения SSD (sgrvinod format).
    """
    print(f"Reading XML annotations from {annotations_dir}...")
    
    # Словарь меток (background = 0, RBC = 1, WBC = 2, Platelets = 3)
    label_map = {'background': 0, 'rbc': 1, 'wbc': 2, 'platelets': 3}

    # Находим все xml файлы системно-независимо
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    xml_files.sort() # Для детерминированного сплита
    
    # Бьем на train (80%) и temp (20%)
    train_xmls, temp_xmls = train_test_split(xml_files, test_size=0.2, random_state=42)
    # Бьем temp (20%) пополам на val (10%) и test (10%)
    val_xmls, test_xmls = train_test_split(temp_xmls, test_size=0.5, random_state=42)

    def process_split(split_xmls, split_name):
        split_images = []
        split_objects = []
        n_objects = 0

        for xml_file in split_xmls:
            xml_path = os.path.join(annotations_dir, xml_file)
            
            parsedXML = ET.parse(xml_path)
            root = parsedXML.getroot()
            
            filename = xml_file.replace('.xml', '.jpg')
            img_path = os.path.abspath(os.path.join(image_dir, filename))
            
            boxes = []
            labels = []

            for node in root.iter('object'):
                label_name = node.find('name').text.lower()
                
                if label_name not in label_map:
                    continue
                
                # Координаты в XML начинаются с 1
                xmin = int(node.find('bndbox/xmin').text) - 1
                ymin = int(node.find('bndbox/ymin').text) - 1
                xmax = int(node.find('bndbox/xmax').text) - 1
                ymax = int(node.find('bndbox/ymax').text) - 1
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_map[label_name])
                n_objects += 1

            if len(boxes) == 0:
                continue

            split_images.append(img_path)
            split_objects.append({'boxes': boxes, 'labels': labels})

        # Сохраняем в JSON
        with open(os.path.join(output_folder, f'{split_name}_images.json'), 'w') as j:
            json.dump(split_images, j)
        with open(os.path.join(output_folder, f'{split_name}_objects.json'), 'w') as j:
            json.dump(split_objects, j)

        print(f"[{split_name}] Сохранено {len(split_images)} картинок и {n_objects} объектов.")

    # Обрабатываем train, val и test
    process_split(train_xmls, 'TRAIN')
    process_split(val_xmls, 'VAL')
    process_split(test_xmls, 'TEST')

    # Сохраняем label_map
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)

if __name__ == '__main__':
    annotations_dir = os.path.join('BCCD_Dataset', 'BCCD', 'Annotations')
    img_dir = os.path.join('BCCD_Dataset', 'BCCD', 'JPEGImages')
    out_dir = os.path.join('.', 'data')
    
    os.makedirs(out_dir, exist_ok=True)
    create_data_lists(annotations_dir, img_dir, out_dir)
