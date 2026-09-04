# Explainable Diabetic Retinopathy Screening

Deployment application foundation.

## Pipeline

Image Upload
-> Image Quality Assessment
-> DR Prediction
-> Probability Distribution
-> Grad-CAM Explanation
-> Screening Result

## Model

B0_CLASSWEIGHTED_FINETUNED_224

EfficientNetB0
224x224 input
5-class diabetic retinopathy grading

## Safety

This application is intended for screening support only.
It does not provide a medical diagnosis.
Clinical assessment by a qualified healthcare professional
is required.

The image-quality gate is advisory and is not clinically validated.
