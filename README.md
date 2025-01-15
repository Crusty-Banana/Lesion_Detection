1. Classify normal or abnormal

- Input: Image
- Output: Prob of being abnormal (We called this output: P)
- CE Loss, SGD optimizer
- Models: DenseNet-169, DenseNet-201, DenseNet-121. (Done)
- Details: 
    - Then get ensemble of these model (average of probability of being abnormal). 
    - Image is abnormal if P >= threshold c. 
    - c is choosen to maximize Youden index J(c) = q(c) + r(c) − 1. q is sensitivity, r is specificity
- Sub-Tasks:
    - Code trainable model
    - Code evaluation for those model
    - Code Youden Index if possible

2. Lesion Detector 

- Input: Image
- Output: localize 7 important lesions: osteophytes, disc space narrowing, surgical implant, foraminal stenosis, spondylolysthesis, vertebral collapse, and other lesions. (bounding box, confidence level of bounding box and class) (We called this output: box, confidence and class)
- SGD optimizer, bounding box regression loss, region-level classification loss (both loss are jointly minimize)
- Models: 
    - Faster R CNN (method: Anchor-based two-stage detectors)
    - RetinaNet or EfficientDet (method: one-stage detectors)
    - Sparse R-CNN 
- Sub-Tasks:
    - Code trainable model
    - Code evaluation for those model

3. Decision Rule:

- if P >= c, all results of Lesion Detector are retained
- if P < c, only result with confidence > 0.5 are retained
- Sub-Tasks:
    - Code framework to integrate abnormal classifier and lesion detector
    - Remember to find c such that it maximize Youden Index.
  
4. Data:

- Columns: study_id,series_id,image_id,rad_id,lesion_type,xmin,ymin,xmax,ymax
- Important columns:
    - image_id: id of image
    - lesion_type: osteophytes, disc space narrowing, surgical implant, foraminal stenosis, spondylolysthesis, vertebral collapse, and other lesions.
    - xmin, ymin, xmax, ymax: bounding box.
- Faster RCNN:
    - Input: Image: RGB image. Normalized between 0 and 1.
    - Target:
        - Boxes: [[xmin, ymin, xmax, ymax], ...]
        - Labels: [3, ...]
        - Note: Some image have no findings.
- Sub-Tasks:
    - Code child dataset class for each type of model.
        - DenseNet
        - Faster R CNN (Done)
        - RetinaNet
        - EfficientDet
        - Sparse R-CNN
     
Get data:

XG9ebRtEFY3EPKs
wget -r -N -c -np --user crustybanana --ask-password https://physionet.org/files/vindr-spinexr/1.0.0/train_images/0c89242a97a3a080b70c3957728a1e89.dicom

# Command to run:

1. Train the models

```
python main.py --model FasterRCNN
python main.py --model RetinaNet
python main.py --model Densenet121
python main.py --model Densenet169
python main.py --model Densenet201
```

2. Inference model on 1 sample

```
python main.py --action inference --model FasterRCNN --model_path models/beta_model
python main.py --action inference --model RetinaNet --model_path models/retina_net_10epoch_model
python main.py --action inference --model Densenet121 --model_path models/densenet_121
python main.py --action inference --model Densenet169 --model_path models/densenet_169
python main.py --action inference --model Densenet201 --model_path models/densenet_201
```

3. Evaluate the model on the test set

```
python main.py --action test --model FasterRCNN --model_path models/beta_model
python main.py --action test --model RetinaNet --model_path models/retina_net_10epoch_model
python main.py --action test --model Densenet121 --model_path models/densenet_121
python main.py --action test --model Densenet169 --model_path models/densenet_169
python main.py --action test --model Densenet201 --model_path models/densenet_201
```
