import torch
from utils.helpers import *
import warnings
from PIL import Image
from torchvision import transforms


# from torchsummary import summary

def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor


def predict(imagepath, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './models/seedlingClassifier.pth'
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = load_model(model_path)
    model.eval()
    # summary(model, input_size=(3,244,244))
    if verbose:
        print("Model Loaded..")
    image = image_transform(imagepath)
    image1 = image[None, :, :, :]
    ps = torch.exp(model(image1))
    topconf, topclass = ps.topk(1, dim=1)
    if topclass.item() == 1:
        return {'name': str(imagepath), 'class': 'seedling', 'confidence': str(topconf.item())}
    else:
        return {'name': str(imagepath), 'class': 'no_seedling', 'confidence': str(topconf.item())}


print(predict('data/1606399760EGGPLANT3.jpeg'))
print(predict('data/1607132230EGGPLANT4.jpeg'))
print(predict('data/1606567393EGGPLANT4.jpeg'))
