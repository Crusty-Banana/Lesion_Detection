1. Classify normal or abnormal

- Input: Image
- Output: Prob of being abnormal (We called this output: P)
- CE Loss, SGD optimizer
- Models: DenseNet-169, DenseNet-201, DenseNet-121. 
- Details: 

Then get ensemble of these model (average of probability of being abnormal). 

Image is abnormal if P >= threshold c. 

c is choosen to maximize Youden index J(c) = q(c) + r(c) − 1. q is sensitivity, r is specificity


2. Lesion Detector 

- Input: Image
- Output: localize 7 important lesions: osteophytes, disc space narrowing, surgical implant, foraminal stenosis, spondylolysthesis, vertebral collapse, and other lesions. (bounding box, confidence level of bounding box and class) (We called this output: box, confidence and class)
- SGD optimizer, bounding box regression loss, region-level classification loss (both loss are jointly minimize)
- Models: 
    - Faster R CNN (method: Anchor-based two-stage detectors)
    - RetinaNet or EfficientDet (method: one-stage detectors)
    - Sparse R-CNN 
- Details:

3. Decision Rule:

- if P >= c, all results of Lesion Detector are retained
- if P < c, only result with confidence > 0.5 are retained

4. Data:

- Columns: study_id,series_id,image_id,rad_id,lesion_type,xmin,ymin,xmax,ymax
- Important columns:
    - image_id: id of image
    - lesion_type: osteophytes, disc space narrowing, surgical implant, foraminal stenosis, spondylolysthesis, vertebral collapse, and other lesions.
    - xmin, ymin, xmax, ymax: bounding box.