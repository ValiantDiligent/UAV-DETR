"""
Author: Huaxiang Zhang
Date: 2025-01-17
Version: 1.0

Description:
This script is a sample file for evaluating and validating COCO formatted annotation and prediction files.
It performs the following tasks:
- Verifies that the image IDs in the annotation and prediction files match.
- Verifies that the categories in the annotation and prediction files are consistent.
- Checks for common issues in the prediction file, such as invalid bounding box formats, missing or invalid scores, and incorrect category IDs.
- Verifies the content of the annotation file, including checking the number of annotations, images, and categories.
- Provides examples of a few annotation and prediction entries to help with debugging.
- Attempts to run the COCO evaluation using pycocotools to evaluate the predicted bounding boxes against the ground truth annotations.

Note:
This script is intended as a simple example to demonstrate how to handle COCO evaluation using Python and pycocotools.
It was generated with the help of ChatGPT.


"""
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def check_image_ids(anno_json, pred_json):
    anno = load_json(anno_json)
    pred = load_json(pred_json)
    
    anno_image_ids = set([img['id'] for img in anno['images']])
    pred_image_ids = set([ann['image_id'] for ann in pred])
    
    missing_in_pred = anno_image_ids - pred_image_ids
    missing_in_anno = pred_image_ids - anno_image_ids
    
    print(f"Total images in annotations: {len(anno_image_ids)}")
    print(f"Total images in predictions: {len(pred_image_ids)}")
    print(f"Images in annotations but not in predictions: {len(missing_in_pred)}")
    print(f"Images in predictions but not in annotations: {len(missing_in_anno)}")
    
    if missing_in_pred:
        print("Some images in annotations are missing in predictions:")
        for img_id in list(missing_in_pred)[:10]:  # 打印前10个
            print(img_id)
    
    if missing_in_anno:
        print("Some images in predictions are missing in annotations:")
        for img_id in list(missing_in_anno)[:10]:
            print(img_id)

def check_categories(anno_json, pred_json):
    anno = load_json(anno_json)
    pred = load_json(pred_json)
    
    anno_categories = set([cat['id'] for cat in anno['categories']])
    pred_categories = set([ann['category_id'] for ann in pred])
    
    missing_in_pred = anno_categories - pred_categories
    missing_in_anno = pred_categories - anno_categories
    
    print(f"Total categories in annotations: {len(anno_categories)}")
    print(f"Total categories in predictions: {len(pred_categories)}")
    print(f"Categories in annotations but not in predictions: {len(missing_in_pred)}")
    print(f"Categories in predictions but not in annotations: {len(missing_in_anno)}")
    
    if missing_in_pred:
        print("Some categories in annotations are missing in predictions:")
        for cat_id in list(missing_in_pred):
            print(cat_id)
    
    if missing_in_anno:
        print("Some categories in predictions are missing in annotations:")
        for cat_id in list(missing_in_anno):
            print(cat_id)

def check_predictions(pred_json):
    pred = load_json(pred_json)
    print(f"Total predictions: {len(pred)}")
    
    # 检查边界框格式
    invalid_bbox = [ann for ann in pred if not (isinstance(ann.get('bbox'), list) and len(ann['bbox']) == 4)]
    print(f"Predictions with invalid bbox format: {len(invalid_bbox)}")
    
    # 检查得分是否存在且在合理范围内
    invalid_scores = [ann for ann in pred if 'score' not in ann or not (0 <= ann['score'] <= 1)]
    print(f"Predictions with invalid or missing scores: {len(invalid_scores)}")
    
    # 检查类别ID是否为正整数
    invalid_cat_ids = [ann for ann in pred if not isinstance(ann.get('category_id'), int) or ann['category_id'] <= 0]
    print(f"Predictions with invalid category_ids: {len(invalid_cat_ids)}")

def check_annotations(anno_json):
    anno = load_json(anno_json)
    print(f"Total annotations: {len(anno.get('annotations', []))}")
    print(f"Total images: {len(anno.get('images', []))}")
    print(f"Total categories: {len(anno.get('categories', []))}")

def print_sample_predictions(pred_json, num_samples=5):
    pred = load_json(pred_json)
    print(f"\n打印 {num_samples} 个预测条目的示例:")
    for idx, ann in enumerate(pred[:num_samples]):
        print(f"条目 {idx + 1} 数据类型: {type(ann)}")
        print(json.dumps(ann, indent=2))

def print_sample_annotations(anno_json, num_samples=5):
    anno = load_json(anno_json)
    print(f"\n打印 {num_samples} 个注释条目的示例:")
    for idx, ann in enumerate(anno['annotations'][:num_samples]):
        print(f"条目 {idx + 1} 数据类型: {type(ann)}")
        print(json.dumps(ann, indent=2))

def main(anno_json, pred_json):
    print("检查注释文件内容...")
    check_annotations(anno_json)
    
    print("\n检查图像ID匹配情况...")
    check_image_ids(anno_json, pred_json)
    
    print("\n检查类别ID匹配情况...")
    check_categories(anno_json, pred_json)
    
    print("\n检查预测文件内容...")
    check_predictions(pred_json)
    
    # 打印示例预测条目
    print_sample_predictions(pred_json)
    
    print_sample_annotations(anno_json)
    
    # 尝试进行COCO评估
    print("\n尝试进行COCO评估...")
    try:
        coco_gt = COCO(anno_json)
        coco_dt = coco_gt.loadRes(pred_json)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"COCO评估过程中出现错误: {e}")


if __name__ == '__main__':
    anno_json = '/mnt/RTdetr/RTDETR-main/dataset/vaste/data_test.json'
    pred_json = '/mnt/RTdetr/RTDETR-main/runs/val/vaste/exp_so_r502/predictions.json'
    main(anno_json, pred_json)
