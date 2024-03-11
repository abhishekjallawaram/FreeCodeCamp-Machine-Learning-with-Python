# Image Classification Challenge: Dogs vs Cats

**Note**: You're currently using Google Colaboratory, a cloud-based version of Jupyter Notebook. This document contains both text cells for documentation and code cells that can be executed. If you're new to Jupyter Notebook, consider watching this [3-minute introduction video](https://www.youtube.com/watch?v=inN8seMm7UI) before starting this challenge.

---

## Overview

In this challenge, you'll develop a convolutional neural network (CNN) to classify images of dogs and cats. Utilizing TensorFlow 2.0 and Keras, your goal is to achieve at least 63% accuracy in distinguishing between the two. Achieving 70% or higher will earn you extra credit!

The challenge involves partially provided code; you'll need to complete certain sections to progress. Follow the instructions in each text cell carefully to understand what's required in the corresponding code cell.

## Dataset Structure

Upon downloading the data, you'll find it organized as follows, with the `test` directory containing unlabeled images:

```
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```


## Instructions

1. **Import Libraries**: The first code cell will include necessary library imports.
2. **Download Data & Set Variables**: The second cell is for data download and setting key variables.
3. **Your Code**: Starting from the third cell, you'll be writing your own code to implement the CNN.

Feel free to adjust the epochs and batch size to fine-tune your model, though it's not a requirement for completing the challenge.

# Image Data Preparation

**Now it's your turn!** Your task is to correctly set up the variables below. Initially set to `None`, these variables should be assigned appropriate values as per the instructions.

## Task

You will create image generators for the training, validation, and test datasets using the `ImageDataGenerator` class. This tool will read and decode the images, converting them into floating-point tensors. Specifically, you'll use the `rescale` argument to adjust the pixel values from a range of 0-255 to a normalized range of 0-1. This normalization helps in improving the performance of the model by scaling the inputs.

### Instructions

1. **Initialize Image Generators**: For each of the datasets (train, validation, test), create an instance of `ImageDataGenerator` with the `rescale` parameter set to rescale the image tensors.

2. **Generate Data**: Utilize the `flow_from_directory` method of your image generators to set up the `*_data_gen` variables. Provide the necessary arguments including:
    - `batch_size`: The number of images to process in a single batch.
    - `directory`: The path to the target directory (for each dataset).
    - `target_size`: The dimensions to which all images found will be resized (`(IMG_HEIGHT, IMG_WIDTH)`).
    - `class_mode`: Defines the type of label arrays that are returned: `"binary"` for 1D binary labels, `"categorical"` for 2D one-hot encoded labels, etc.
    - For `test_data_gen`, ensure you set `shuffle=False` in the `flow_from_directory` call. This is crucial for maintaining the order of the test images, aligning with the expected order of predictions.

### Note on Test Data
- The `test_data_gen` requires careful attention due to its unique directory structure and the necessity to keep the image order fixed (hence `shuffle=False`). Observing the directory structure will aid in correctly setting up this generator.

### Expected Output

After executing your code, the output should confirm the successful preparation of your image datasets, resembling the following:

```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 50 images belonging to 1 classes.
```


This output indicates that your image data generators are correctly set up and ready for use in training and evaluating your model.

# Augmenting Training Data

Due to the limited number of training examples, there's a significant risk of overfitting. A common strategy to mitigate this issue is to augment the training data by applying random transformations to the existing images. This approach generates more diverse training examples, helping the model generalize better to unseen data.

## Task

Recreate your `train_image_generator` using the `ImageDataGenerator` class from TensorFlow's Keras API, incorporating several random transformations to augment your data.

### Instructions

1. **Initialize `ImageDataGenerator`**: Create a new instance of `ImageDataGenerator` for the training dataset. Ensure you include the `rescale` argument to normalize the image pixels values between 0 and 1.

2. **Apply Random Transformations**: Enhance the data generator by adding 4-6 random transformations as arguments. Some common transformations include:
    - `rotation_range`: A degree range within which to randomly rotate pictures.
    - `width_shift_range` and `height_shift_range`: Ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
    - `shear_range`: For randomly applying shearing transformations.
    - `zoom_range`: For randomly zooming inside pictures.
    - `horizontal_flip`: For randomly flipping half of the images horizontally — relevant when there are no assumptions of horizontal asymmetry (e.g., real-world pictures).
    - `fill_mode`: The strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

### Example

```python
train_image_generator = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    height_shift_range = 0.2,
    width_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2)
```

# Visualizing Data Augmentation

In this section, we'll visualize the impact of our data augmentation strategy on the training images. This step is crucial for understanding how the random transformations applied by `ImageDataGenerator` can lead to diverse training examples, thereby helping to prevent overfitting.

## What's Happening?

- **Data Generation**: The `train_data_gen` is set up using the newly configured `train_image_generator` with data augmentation. This process mirrors the previous setup but incorporates the random transformations we've added.
- **Image Visualization**: We will then display a single image from our dataset five times, each time applying a different set of transformations. This illustrates the variety of training examples that our model will see, without needing to manually expand the dataset.

## No Action Required

For this specific cell, **no action is required on your part**. Simply run the cell to execute the predefined operations:
- The creation of `train_data_gen` using the augmented `train_image_generator`.
- The visualization of augmented images.

This automatic process showcases the power of data augmentation in generating diverse training samples from a single image, effectively demonstrating how such transformations can enhance the model's ability to generalize from limited data.


# Building and Compiling the CNN Model

In this section, you will define and compile the convolutional neural network (CNN) model that predicts class probabilities. The model will be built using TensorFlow's Keras API, specifically employing the Sequential model framework to facilitate a linear stack of layers.

## Model Architecture

1. **Convolutional Layers**: Begin with a stack of `Conv2D` layers. These layers will help the model to learn the patterns in the images.
2. **Pooling Layers**: Follow each `Conv2D` layer with a `MaxPooling2D` layer. Pooling reduces the dimensionality of the data by downscaling, making the detection of features invariant to scale and orientation changes.
3. **Fully Connected Layer**: After the convolutional and pooling layers, flatten the network and add a fully connected (`Dense`) layer. This layer is activated by a ReLU activation function, serving as a classifier on top of the features extracted by the convolutions.

## Compiling the Model

- **Optimizer**: Choose an optimizer that will update the model's weights based on the observed data and the loss function. Common choices include `adam`, `sgd`, etc.
- **Loss Function**: Use a loss function suitable for a classification problem. For binary classification, `binary_crossentropy` is a common choice. For multi-class classification, consider `categorical_crossentropy`.
- **Metrics**: Include `accuracy` in the metrics to track the training and validation accuracy of each epoch. This provides insight into the performance of your model throughout the training process.

### Example Code

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
```

# Training the Model

With your convolutional neural network (CNN) model built and compiled, the next step is to train it using the dataset you've prepared. Training is accomplished by calling the `fit` method on your model, which adjusts the model parameters (weights) to minimize the loss and improve accuracy.

## Steps for Training

1. **Prepare Data**: Ensure your training (`train_data_gen`) and validation (`val_data_gen`) datasets are ready for use.
2. **Call `fit` Method**: Use the `fit` method on your model to start the training process. You'll need to provide it with the training data, validation data, and specify how many epochs to train for.

### Parameters to Specify

- **`x`**: Typically, you'll pass your training data generator as this argument.
- **`steps_per_epoch`**: This controls how many batches of samples to use for one epoch. Setting this to the total number of samples divided by the batch size is a common practice.
- **`epochs`**: The number of times the learning algorithm will work through the entire training dataset. Choose a number that balances between underfitting and overfitting.
- **`validation_data`**: Your validation data generator. It's used to evaluate the loss and any model metrics at the end of each epoch.
- **`validation_steps`**: Similar to `steps_per_epoch`, but for validation data. Set this to the total number of validation samples divided by the validation batch size.

### Example Code

```python
history = model.fit(
        x=train_data_gen,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size)
```

# Visualizing Model Performance

After training your model, it's essential to evaluate its performance by examining the accuracy and loss metrics. These metrics provide insight into how well your model is learning from the training data and generalizing to the validation data.

## What to Expect

By running the next cell, you will generate plots that depict the model's training and validation accuracy, as well as its training and validation loss over each epoch. These visualizations are crucial for identifying trends such as:
- **Overfitting**: If the training accuracy significantly exceeds the validation accuracy.
- **Underfitting**: If both training and validation accuracies are low or if the model performs poorly on both the training and validation sets.

## Steps

1. **Execute the Cell**: Simply run the next code cell to generate the visualizations. No modifications are needed.
2. **Interpret the Plots**: Look for the following indicators:
   - A closing gap between training and validation accuracy suggests good generalization.
   - A decreasing trend in loss over epochs indicates learning progress.

## Understanding the Outputs

- **Accuracy Plot**: Shows the comparison between the training accuracy and validation accuracy across epochs.
- **Loss Plot**: Illustrates how the training loss and validation loss change throughout the training process.

These plots are invaluable tools for tuning your model further. They can help you decide if you need to adjust the model architecture, data augmentation strategies, or training parameters (like the learning rate).

Now, go ahead and run the next cell to visualize your model's learning journey!

# Making Predictions with Your Model

Having trained your model, the next exciting step is to use it for predicting whether new, unseen images are of cats or dogs. This is a crucial test of your model's ability to generalize from the training data to real-world examples.

## Task

You will now use the `test_data_gen` dataset to predict the class (cat or dog) of each image. Your task involves generating predictions and then visualizing these images along with their predicted classes.

### Instructions

1. **Generate Predictions**: Use your trained model to predict the class of each image in the `test_data_gen`. The output should be a list of probabilities that indicate the likelihood of each image being a dog (or conversely, a cat).
2. **Convert Probabilities to Integers**: Depending on your model's final layer activation (e.g., softmax for multi-class classification), you may need to convert the output probabilities to a binary form (0 or 1) representing cats or dogs.
3. **Visualize Predictions**: Utilize the `plotImages` function provided in your environment. Pass in the test images along with their corresponding probabilities or class predictions.

### Expected Outcome

Upon running your final cell, you should see all 50 test images displayed with labels indicating the model's confidence level (as a percentage) that each image is a cat or a dog. This visualization not only showcases your model's predictive power but also aligns with the accuracy metrics observed in the previous steps.

### Note on Model Accuracy

- The accuracy of your predictions will reflect the performance metrics observed during training and validation. Enhancing the training dataset size or diversity could potentially increase model accuracy.

### Example Code Snippet

```python
# Assuming 'model' is your trained model and 'test_data_gen' is your test data generator
probabilities = model.predict(test_data_gen)
# Convert probabilities to binary predictions, 1 for dog and 0 for cat, based on a threshold (e.g., 0.5)
predictions = [1 if prob > 0.5 else 0 for prob in probabilities]

# Now call the plotting function with test images and their predicted labels
plotImages(test_images, predictions)

