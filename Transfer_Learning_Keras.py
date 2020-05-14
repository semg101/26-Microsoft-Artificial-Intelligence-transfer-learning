'''
Functions to generate some image data

First, we'll create some functions to generate our image data. In reality, we'd use images of real objects; 
but we'll just generate a small number of images of basic geometric shapes.
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

Now that we have our data, we're ready to build a CNN. The first step is to import and configure the framework we want to use.

!pip install --upgrade keras
'''

import keras
from keras import backend as K

print('Keras version:',keras.__version__)

Preparing the Data

Before we can train the model, we need to prepare the data.

from keras.preprocessing.image import ImageDataGenerator

data_folder = 'small_shapes'
# Our source images are 128x128, but the base model we're going to use was trained with 224x224 images
pretrained_size = (224,224)
batch_size = 15

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values
                             validation_split=0.3) # hold back 30% of the images for validation

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    data_folder,
    target_size=pretrained_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    data_folder,
    target_size=pretrained_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

classnames = list(train_generator.class_indices.keys())
print("class names: ", classnames)

Downloading a trained model to use as a base

The VGG16 model is an image classifier that was trained on the ImageNet dataset - a huge dataset containing thousands of images of many kinds of object. We'll download the trained model, excluding its top layer, and set its input shape to match our image data.

Note: The keras.applications namespace includes multiple base models, some which may perform better for your dataset than others. I've chosen this model because it's fairly lightweight within the limited resources of the Azure Notebooks environment.

from keras import applications
#Load the base model, not including its final connected layer, and set the input shape to match our images
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=train_generator.image_shape)

Freeze the already trained layers and add a custom output layer for our classes

The existing feature extraction layers are already trained, so we just need to add a couple of layers so that the model output is the predictions for our classes.

from keras import Model
from keras.layers import Flatten, Dense

# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create layers for classification of our images
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(len(classnames), activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=prediction_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Now print the full model, which will include the layers of the base model plus the dense layer we added
print(model.summary())

Training the Model

With the layers of the CNN defined, we're ready to train the top layer using our image data. This will take a considerable amount of time on a CPU depending on the complexity of the base model.

# Train the model over 2 epochs using 15-image batches and using the validation holdout dataset for validation
num_epochs = 2
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = num_epochs)

View the Loss History

We tracked average training and validation loss for each epoch. We can plot these to see where the levels of loss converged, and to detect overfitting (which is indicated by a continued drop in training loss after validation loss has levelled out or started to increase.

from matplotlib import pyplot as plt

epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()


'''
Using the Trained Model

Now that we've trained the model, we can use it to predict the class of an image.
'''

def predict_image(classifier, image_array):
    import numpy as np
    
    # We need to format the input to match the training data
    # The generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    imgfeatures = image_array.astype('float32')
    imgfeatures /= 255
    
    # These are the classes our model can predict
    classes = ['circle', 'square', 'triangle']
    
    # Predict the class of each input image
    predictions = classifier.predict(imgfeatures)
    
    predicted_classes = []
    for prediction in predictions:
        # The prediction for each image is the probability for each class, e.g. [0.8, 0.1, 0.2]
        # So get the index of the highest probability
        class_idx = np.argmax(prediction)
        # And append the corresponding class name to the results
        predicted_classes.append(classes[int(class_idx)])
    # Return the predictions as a JSON
    return predicted_classes


from random import randint
from PIL import Image
import numpy as np
%matplotlib inline

# Create a random test image
img = create_image ((224,224), classnames[randint(0, len(classnames)-1)])
plt.imshow(img)

# Create an array of (1) images to match the expected input format
img_array = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# get the predicted clases
predicted_classes = predict_image(model, img_array)

# Display the prediction for the first image (we only submitted one!)
print(predicted_classes[0])

