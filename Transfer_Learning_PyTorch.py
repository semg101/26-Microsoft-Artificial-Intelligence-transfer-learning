'''
Functions to generate some image data

First, we'll generate our image data, which will consist of only 150 images. In reality, we'd use images of real objects; 
but we'll just generate some images of basic geometric shapes.
'''
# function to generate an image of random size and color
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw
    
    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw
    
    return np.array(img)

# function to create a dataset of images
def generate_image_data (classes, size, cases, img_dir):
    import os, shutil
    from PIL import Image
    
    if os.path.exists(img_dir):
        replace_folder = input("Image folder already exists. Enter Y to replace it (this can take a while!). \n")
        if replace_folder == "Y":
            print("Deleting old images...")
            shutil.rmtree(img_dir)
        else:
            return # Quit - no need to replace existing images
    os.makedirs(img_dir)
    print("Generating new images...")
    i = 0
    while(i < (cases - 1) / len(classes)):
        if (i%25 == 0):
            print("Progress:{:.0%}".format((i*len(classes))/cases))
        i += 1
        for classname in classes:
            img = Image.fromarray(create_image(size, classname))
            saveFolder = os.path.join(img_dir,classname)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            imgFileName = os.path.join(saveFolder, classname + str(i) + '.jpg')
            try:
                img.save(imgFileName)
            except:
                try:
                    # Retry (resource constraints in Azure notebooks can cause occassional disk access errors)
                    img.save(imgFileName)
                except:
                    # We gave it a shot - time to move on with our lives
                    print("Error saving image", imgFileName)
            
# Our classes will be circles, squares, and triangles
classnames = ['circle', 'square', 'triangle']

# All images will be 128x128 pixels
img_size = (128,128)

# We'll store the images in a folder named 'shapes'
folder_name = 'small_shapes'

# Generate 90 random images.
generate_image_data(classnames, img_size, 90, folder_name)

print("Image files ready in %s folder!" % folder_name)


'''

Setting up the Frameworks

Now that we have our data, we're ready to build a classifier from a pre-trained CNN. The first step is to install and configure the frameworks we want to use.

Note: For instructions on how to install the PyTorch and TorchVision packages on your own system, see https://pytorch.org/get-started/locally/
'''
# Install PyTorch
!pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
!pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
    
# Import PyTorch Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

print('Ready to train a model using PyTorch', torch.__version__)


'''

Downloading a trained model to user as a base

The resnet model is an CNN-based image classifier that has been pre-trained using a huge dataset containing thousands of images of many kinds of object. 
We'll download the trained model, excluding its final linear layer, and freeze the convolutional layers to retain the trained weights. 
Then we'll add a new linear layer that will map the features extracted by the convolutional layers to the classes of our shape images.
'''
model_resnet = torchvision.models.resnet18(pretrained=True)
for param in model_resnet.parameters():
    param.requires_grad = False

num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, len(classnames))

# Now print the full model, which will include the layers of the base model plus the linear layer we added
print(model_resnet)


'''

Loading and Preparing the Data

Before we can train the model to classify images based on our shape classes, we need to prepare the training data. 
PyTorch includes functions for loading and transforming data. We'll use these to create an iterative loader for training data, 
and a second iterative loader for test data (which we'll use to validate the trained model). The loaders will transform 
the image data to match the format used to train the original resnet CNN model, and finally convert the image data into tensors, 
which are the core data structure used in PyTorch.

Run the following cell to define the data loaders, and then load the first batch of 32 training images and display them along with their class labels.
'''
# Function to ingest data using training and test loaders
def load_dataset(data_path):
    
    # Resize to 256 x 256, center-crop to 224x224 (to match the resnet image size), and convert to Tensor
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    # Split into training (70%) and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 32-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False
    )
    
    # define a loader for the testing data we can iterate through in 32-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader


# Now load the images from the shapes folder
import os  
data_path = 'small_shapes/'

# Get the class names
classes = os.listdir(data_path)
classes.sort()
print(len(classes), 'classes:')
print(classes)

# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset(data_path)


'''

Training the Model

With the layers of the CNN defined, we're ready to train it using our image data. The weights used in the feature extraction layers from 
the base resnet model will not be changed by training, only the final linear layer that maps the features to our shape classes will be trained.

'''
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
            
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Get the predicted classes for this batch
            output = model(data)
            # calculate the loss and successful predictions for this batch
            test_loss += loss_criteria(output, target).item()
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate the average loss and total accuracy for this epoch
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return test_loss
    
    
# Now use the train and test functions to train and test the model    

device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)

# Create an instance of the model class and allocate it to the device
model_resnet = model_resnet.to(device)

# Use an "Adam" optimizer to adjust weights
# (see https://pytorch.org/docs/stable/optim.html#algorithms for details of supported algorithms)
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 3 epochs
epochs = 3
for epoch in range(1, epochs + 1):
        train_loss = train(model_resnet, device, train_loader, optimizer, epoch)
        test_loss = test(model_resnet, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)


'''

View the Loss History

We tracked average training and validation loss for each epoch. We can plot these to see where the levels of loss converged, 
and to detect overfitting (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase.
'''
%matplotlib inline
from matplotlib import pyplot as plt

plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()


'''

Evaluate Model Performance

We can see the final accuracy based on the test data, but typically we'll want to explore performance metrics in a little mode depth. 
Let's plot a confusion matrix to see how well the model is predicting each class.

'''
#Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

# Set the model to evaluate mode
model_resnet.eval()

# Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in test_loader:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model_resnet.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")
plt.show()

'''

Using the Trained Model

Now that we've trained the model, we can use it to predict the class of an image.
'''
# Function to predict the class of an image
def predict_image(classifier, image_array):
    from PIL import Image
    import numpy as np
    
    # Set the classifer model to evaluation mode
    classifier.eval()
    
    # These are the classes our model can predict
    class_names = ['circle', 'square', 'triangle']
    
    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Preprocess the imagees
    image_tensor = torch.stack([transformation(Image.fromarray(image)).float() for image in image_array])

    # Predict the class of each input image (using the CPU/GPU as available)
    predictions = classifier(image_tensor.to(device))
    
    predicted_classes = []
    # Convert the predictions to a numpy array on the CPU
    for prediction in predictions.cpu().data.numpy():
        # The prediction for each image is the probability for each class, e.g. [0.8, 0.1, 0.2]
        # So get the index of the highest probability
        class_idx = np.argmax(prediction)
        # Add add the corresponding class name to the results
        predicted_classes.append(class_names[class_idx])
    return np.array(predicted_classes)


# Now let's try it with a new image
from random import randint

# Create a random test image
img = create_image ((224,224), classes[randint(0, len(classes)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
image_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

predicted_classes = predict_image(model_resnet, image_array)
print(predicted_classes[0])