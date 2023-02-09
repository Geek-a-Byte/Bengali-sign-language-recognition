## Summary


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


