import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork

if __name__ == '__main__':
    # load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    test_data = datasets.FashionMNIST(
        root="~/torch_vision_data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
