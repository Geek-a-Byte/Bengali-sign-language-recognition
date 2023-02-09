## Abstract

<pre>
The sole form of communication available to the deaf and dumb (D&D) people is sign language. 
Every day, the D&D community faces challenges while attempting to communicate with the general public. 
Because they need to communicate their feelings through the interchange of visual cues to function normally. 
They typically require an interpreter to speak with others, although they might not always have access to one. 
The problem is also experienced by those who use Bengali Sign Language (BdSL), as more BdSL interpreters must be needed. 
In Bangladesh, studying sign language is difficult because there aren't many institutions dedicated to the subject, 
and there aren't enough online resources or aids. In order to address this issue, computer vision-based techniques 
for the automatic recognition of sign languages have recently been developed. However, there needs to be more credible 
works to support the acknowledgment of BdSL. In this article, we suggest a technique for automatically identifying BdSL 
alphabets. In BdSL, there are 38 different letter signs. In this study, we compare five pre-trained models for recognizing 
38 classes - Densenet121, VGG16, Mobilenet v3 small, Mobilenet v2, and Resnet50; we achieve an overall test accuracy of 
96.57%, 95.13%, 92.82%, 95.52%, 96.31% respectively.
</pre>

| Pre-trained Model Name | loss function    | optimizer | learning rate | step_size | gamma | no of epochs | Best epoch acc | training time | test_accuracy | precision | recall | f1-score |
| ---------------------- | ---------------- | --------- | ------------- | --------- | ----- | ------------ | -------------- | ------------- | ------------- | --------- | ------ | -------- |
| densenet121            | CrossEntropyLoss | SGD       | 0.006         | 3         | 0.1   | 5            | 0.963158       | 10m 45s       | 96.3158       | 96.48     | 96.32  | 96.31    |
| Resnet50               | CrossEntropyLoss | SGD       | 0.006         | 3         | 0.1   | 5            | 0.960526       | 10m 1s        | 96.0526       | 96.37     | 96.05  | 96.02    |
| mobilenet_v2           | CrossEntropyLoss | SGD       | 0.006         | 3         | 0.1   | 5            | 0.954605       | 4m 11s        | 95.4605       | 95.76     | 95.46  | 95.45    |
| vgg16                  | CrossEntropyLoss | SGD       | 0.001         | 7         | 0.1   | 5            | 0.942763       | 13m 12s       | 94.2763       | 94.67     | 94.28  | 94.28    |
| Mobilenet_V3_small     | CrossEntropyLoss | SGD       | 0.006         | 3         | 0.1   | 5            | 0.929605       | 2m 41s        | 92.9605       | 93.58     | 92.96  | 93.02    |

## load a model and test the accuracy

```py

device = torch.device("cuda")
model = models.vgg16(pretrained=True)
for param in model.parameters():
  param.requires_grad = True
  
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs,38)
# model_vgg = model_vgg.to(device)

model.load_state_dict(torch.load('BSL_Vgg16.pt'))
model.to(device)


model_vgg=model
predictions_list, labels_list = measure_test_acc(model_vgg)

```


