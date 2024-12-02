import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
from torchsummary import summary
import torch.optim as optim
import time
import torch.nn as nn

def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')


    AlexNet_Model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    AlexNet_Model.eval()

    AlexNet_Model.classifier[1] = nn.Linear(9216,4096)
    AlexNet_Model.classifier[4] = nn.Linear(4096,1024)
    AlexNet_Model.classifier[6] = nn.Linear(1024,10)

    AlexNet_Model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AlexNet_Model.to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.001, momentum=0.9)

    # 모델 요약 출력 (input size는 CIFAR-10 이미지 크기인 32x32x3)
    summary(AlexNet_Model, input_size=(3, 224, 224))


    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = AlexNet_Model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            #Time
            end_time = time.time()
            time_taken = end_time - start_time

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                print('Time:',time_taken)
                running_loss = 0.0

    print('Finished Training of AlexNet')

    #Testing Accuracy

        # Performance Evaluation
    def evaluate_model(model, testloader, classes):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = nn.functional.softmax(outputs, dim=1)
                all_preds.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Convert predictions and labels to numpy arrays
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate Top-1 and Top-3 accuracies
        top1_accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
        top3_accuracy = top_k_accuracy_score(all_labels, all_preds, k=3, labels=np.arange(len(classes)))

        # Generate Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, np.argmax(all_preds, axis=1))

        # Print results
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.yticks(np.arange(len(classes)), classes)
        plt.tight_layout()
        plt.show()

    # Call the evaluation function
    evaluate_model(AlexNet_Model, testloader, classes)


if __name__ == '__main__':
    main()



