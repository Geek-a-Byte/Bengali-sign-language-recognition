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

## ``pre-trained models used``
![image](https://user-images.githubusercontent.com/59027621/205045598-a598e2f5-092a-41de-baf7-331412923a66.png)

will verify the metrics later
- https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

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
