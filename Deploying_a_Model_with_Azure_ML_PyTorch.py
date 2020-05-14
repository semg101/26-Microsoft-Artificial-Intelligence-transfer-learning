#Use a classification model locally
'''
Let's start by verifying that the classification model you created previously works locally.

> Note: If you haven't used the 05a - Image Classification with a CNN (PyTorch).ipynb notebook to create a PyTorch image classifier, go and do it now - we'll wait!

First, we'll install the latest version of PyTorch and import the libraries we'll use to train and test the model locally.

# Install PyTorch
!pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
!pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
'''
# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("Libraries imported - ready to use PyTorch", torch.__version__)

# Other libraries we'll use
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline

'''
Next we'll create a function to generate new images, and a function to classify an image using a specified classification model.
'''

# Function to create a random image (of a square, circle, or triangle)
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
    elif shape == 'square':
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    else: # triangle
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    del draw
    
    return np.array(img)

# Function to predict the class of an image
def predict_image(classifier, image_array):
    import torch.utils.data as utils
    import numpy as np
    
    # Set the classifer model to evaluation mode
    classifier.eval()
    
    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the image
    image_tensor = torch.stack([transformation(image).float() for image in image_array])

    # Predict the class of the image
    predictions = classifier(image_tensor)
    
    classnames = ['circle', 'square', 'triangle']
    
    predicted_classes = []
    for prediction in predictions.data.numpy():
        class_idx = np.argmax(prediction)
        predicted_classes.append(classnames[class_idx])
    return np.array(predicted_classes)

print("Functions ready")


'''
Now you're ready to try to classify new images using the model you created in the 05a - Image Classification with a CNN (PyTorch).ipynb notebook. 
You'll need to define a class for this model, and load the weights you saved previously.
'''

# Create a neural net class
class Net(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        x = x.view(-1, 32 * 32 * 24)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
# Create a new model and load the weights
model_file = 'shape-classifier.pth'
model = Net()
model.load_state_dict(torch.load(model_file))
print("New model created from saved weights")

#Now you can use the model to predict the class of a new image.

# Now let's try it with a new image
from random import randint

# Create a random test image
classnames = ['circle', 'square', 'triangle']
img_size = (128,128)
img = create_image (img_size, classnames[randint(0, len(classnames)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
image_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]).astype('float32')

predicted_classes = predict_image(model, image_array)
print(predicted_classes[0])


'''
It looks as though we have a working model. Now we're ready to use Azure Machine Learning to deploy it as a web service.
Create an Azure Machine Learning workspace
'''
#Create an Azure Machine Learning workspace
'''
To use Azure Machine Learning, you'll need to create a workspace in your Azure subscription.

Your Azure subscription is identified by a subscription ID. To find this:

    Sign into the Azure portal at https://portal.azure.com.
    On the menu tab on the left, click 🔑 Subscriptions.
    View the list of your subscriptions and copy the ID for the subscription you want to use.
    Paste the subscription ID into the code below, and then run the cell to set the variable - you will use it later.
'''

# Replace YOUR_SUBSCRIPTION_ID in the following variable assignment:
SUBSCRIPTION_ID = 'YOUR_SUBSCRIPTION_ID'


'''
To deploy the model file as a web service, we'll use the Azure Machine Learning SDK.

> Note: the Azure Machine Learning SDK is installed by default in Azure Notebooks and the Azure Data Science Virtual Machine, but you may want to ensure that it's upgraded to the latest version. If you're using your own Python environment, you'll need to install it using the instructions in the Azure Machine Learning documentation*

!pip install azureml-sdk --upgrade
'''

import azureml.core
print(azureml.core.VERSION)

'''
To manage the deployment, we need an Azure ML workspace. Create one in your Azure subscription by running the following cell. 
The first time you run this you'll be prompted to sign into your Azure subscription by entering a code at a given URL, so just click 
the link that's displayed and enter the specified code.
'''

from azureml.core import Workspace
ws = Workspace.create(name='aml_workspace_pytorch', # or another name of your choosing
                      subscription_id=SUBSCRIPTION_ID,
                      resource_group='aml_resource_group_pytorch', # or another name of your choosing
                      create_resource_group=True,
                      location='eastus2' # or other supported Azure region
                     )

#Now that you have a workspace, you can save the configuration so you can reconnect to it later.

from azureml.core import Workspace

# Save the workspace config
ws.write_config()

# Reconnect to the workspace (if you're not already signed in, you'll be prompted to authenticate with a code as before)
ws = Workspace.from_config()

'''
Register the model in Azure ML

You've created a model and saved it locally. Now you can register this model in your Azure ML workspace, which will enable you to manage and deploy it.
'''

from azureml.core.model import Model

model = Model.register(model_path = "shape-classifier.pth",
                       model_name = "shape-classifier-pytorch",
                       tags = {'area': "shapes", 'type': "classifier"},
                       description = "PyTorch shape classifier",
                       workspace = ws)

'''
Create a scoring file

Your web service will need some Python code to load the input data, get the model, and generate and return a prediction. 
We'll save this code in a scoring file that will be deployed to the web service:
'''

%%writefile score_pytorch.py

# create a scoring script that loads and infers from the model
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from azureml.core.model import Model

def init():
    try:
        global model
        MODEL_NAME = 'shape-classifier-pytorch'
        # retieve the local path to the model using the model name
        MODEL_PATH = Model.get_model_path(MODEL_NAME)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Net()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

# REST API served by Azure ML supports json input
def run(json_data):
    try:
        data = np.array(json.loads(json_data)['data']).astype('float32')
        predictions = predict_image(model, data)
        return json.dumps(predictions.tolist())
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

# Function to predict
def predict_image(classifier, image_array):
    import torch
    import torch.utils.data as utils
    from torchvision import transforms
    import numpy
    
    classifier.eval()
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_tensor = torch.stack([transformation(image).float() for image in image_array])
    predictions = classifier(image_tensor)
    classnames = ['circle', 'square', 'triangle']
    predicted_classes = []
    for prediction in predictions.data.numpy():
        class_idx = np.argmax(prediction)
        predicted_classes.append(classnames[class_idx])
    return np.array(predicted_classes)
    
# Define the Net class as used for training so we can load the trained weights
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.2)
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        x = x.view(-1, 32 * 32 * 24)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


'''
Create an environment file

The web service will be hosted in a container, and the container will need to install any Python dependencies when it gets initialized. In this case, our scoring code requires the torch and torchvision Python libraries, so we'll create a .yml file that tells the container host to install these into the environment along with the default libraries used by Azure ML.
'''

from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("pytorch")
myenv.add_conda_package("torchvision")
myenv.add_channel("pytorch")

env_file = "env_pytorch.yml"

with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

with open(env_file,"r") as f:
    print(f.read())

'''
Deploy the web service

Now we're ready to deploy. We'll deploy the container a service named pytorch-shape-classifier. The deployment process includes the following steps:

    Define an inference configuration, which includes the scoring and environment files required to load and use the model.
    Define a deployment configuration that defines the execution environment in which the service will be hosted. In this case, an Azure Container Instance.
    Deploy the web service.
    Verify the status of the deployed service.

This will take some time. When deployment has completed successfully, you'll see a status of Healthy.
'''


from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(runtime= "python", 
                                   entry_script="score_pytorch.py",
                                   conda_file="env_pytorch.yml")

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "pytorch-shape-classifier"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)


'''
Use the web service

With the service deployed, now we can test it by using it to predict the shape of a new image.
'''

import json
from random import randint

# Create a random test image
img = create_image ((128,128), classnames[randint(0, len(classnames)-1)])
plt.imshow(img)

# Modify the image data to create an array of 1 image, matching the format of the training features
input_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# Convert the array to JSON format
input_json = json.dumps({"data": input_array.tolist()})

# Call the web service, passing the input data (the web service will also accept the data in binary format)
predictions = service.run(input_data = input_json)

# Get the predicted class - it'll be the first (and only) one.
classname = json.loads(predictions)[0]
print('The image is a', classname)

#You can also send a batch of images to the service, and get back a prediction for each one.

import json
from random import randint
import matplotlib.pyplot as plt
%matplotlib inline

# Create three random test images
fig = plt.figure(figsize=(6, 6))
images = []
i = 0
while(i < 3):  
    # Create a new image
    img = create_image((128,128), classnames[randint(0, len(classnames)-1)])
    # Plot the image
    a=fig.add_subplot(1,3,i + 1)
    imgplot = plt.imshow(img)
    # Add the image to an array to be submitted as a batch
    images.append(img.tolist())
    i += 1

# Convert the array to JSON format
input_json = json.dumps({"data": images})

# Call the web service, passing the input data
predictions = service.run(input_data = input_json)

# Get the predicted classes
print(json.loads(predictions))


'''
Using the Web Service from Other Applications

The code above uses the Azure ML SDK to connect to the containerized web service and use it to generate predictions from your image classification model. In production, the model is likely to be consumed by business applications that make HTTP requests to the web service.

Let's determine the URL to which these applications must submit their requests:
'''

endpoint = service.scoring_uri
print(endpoint)

'''
Now that we know the endpoint URI, an application can simply make an HTTP request, sending the image data in JSON (or binary) format, 
and receive back the predicted class(es).
'''

import requests
import requests
import json
from random import randint

# Create a random test image
img = create_image ((128,128), classnames[randint(0, len(classnames)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
image_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": image_array.tolist()})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
print(json.loads(predictions.content))


'''
Deleting the Service

When we're finished with the service, we can delete it to avoid incurring charges.
'''

service.delete()
print("Service deleted.")

#And if you're finished with the workspace, you can delete that too

rg = ws.resource_group
ws.delete()
print("Workspace deleted. You should delete the '%s' resource group in your Azure subscription." % rg)

