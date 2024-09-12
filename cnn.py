import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# def CNNForOCR(modelName,imgShape):
#     '''
#     Optimizer: Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, Optimizer, RMSprop, SGD
#     '''
#     h = imgShape[0]
#     w = imgShape[1]

#     if modelName == '32_10':
#         num_classes = 10

#         # 1 st layer must set input shape
#         model = keras.models.Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(h, w, 1)))
#         model.add(MaxPooling2D(pool_size=(2, 2)))  # stride = kernel_size

#         model.add(Dropout(0.25))
#         # convert to 1D array for feed to neural network
#         model.add(Flatten())
#         # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dense(num_classes, activation='softmax'))

#         # model.add(Dropout(0.5))
#         # model.add(Dense(num_classes, activation='softmax'))

#         model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.adam(),
#                 metrics=['accuracy'])
    
#     elif modelName == '28_10_adam':
#         # ref: https://towardsdatascience.com/digit-recognizer-using-cnn-55c65ca7f9e5
#         num_classes = 10

#         # 1 st layer must set input shape
#         model = Sequential()
#         model.add(Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(h, w, 1)))
#         model.add(MaxPooling2D(pool_size=(2, 2)))  # stride = kernel_size
#         # convert to 1D array for feed to neural network
#         model.add(Flatten())
#         model.add(Dense(128, activation=tf.nn.relu))
#         model.add(Dropout(0.2))
#         model.add(Dense(num_classes, activation=tf.nn.softmax))
        
#         # model.compile(loss='sparse_categorical_crossentropy',
#         #         optimizer=tf.keras.optimizers.Adam,
#         #         metrics=['accuracy'])
#         model.compile(loss='sparse_categorical_crossentropy',
#                     optimizer='adam',
#                     metrics=['accuracy'])

#     elif modelName == '32_10_sgd':
#         # ref https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
#         num_classes = 10

#         # 1 st layer must set input shape
#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(h, w, 1)))
#         model.add(MaxPooling2D(pool_size=(2, 2)))  # stride = kernel_size
#         # convert to 1D array for feed to neural network
#         model.add(Flatten())
#         # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dense(num_classes, activation='softmax'))
#         # tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#         #listClassOpt = dir(tf.keras.optimizers)
#         #sgdOpt = tf.keras.optimizers.SGD()
#         model.compile(loss='categorical_crossentropy',
#                 optimizer='SGD',
#                 metrics=['accuracy'])

#     elif modelName == '64_10': 
#         num_classes = 10

#         # 1 st layer must set input shape
#         model = Sequential()
#         model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(h, w, 1)))
#         model.add(MaxPooling2D(pool_size=(2, 2)))  # stride = kernel_size

#         # model.add(Dropout(0.25))
#         # convert to 1D array for feed to neural network
#         model.add(Flatten())
#         # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dense(num_classes, activation='softmax'))

#         # model.add(Dropout(0.5))
#         # model.add(Dense(num_classes, activation='softmax'))

#         model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])

#     elif modelName == 'mnist_colab':
#         # ref: https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=IWgBGmaplzcp
#         # Define the model architecture
        
#         num_classes = 10

#         model = Sequential([
#             keras.layers.InputLayer(input_shape=(h, w, 1)),
#             # keras.layers.Reshape(target_shape=(h, w, 1)),
#             keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
#             keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
#             keras.layers.MaxPooling2D(pool_size=(2, 2)),
#             keras.layers.Dropout(0.25),
#             keras.layers.Flatten(),
#             keras.layers.Dense(num_classes, activation = tf.nn.softmax)
#                             ])

#         # Define how to train the model
#         model.compile(optimizer='adam',
#                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                     metrics=['accuracy'])

#         # Train the digit classification model
#         # model.fit(train_images, train_labels, epochs=5)
#         # print(model.summary())

#     return model

def CNNForFeature(name,imgShape):
    '''
        - Create model without classification layer(Fully connected nn)
    '''
    if name == 'VGG-19_avg':
        '''
            keras.applications already implement state-of-the-art-model and 
            offer option to user to set whether use as feature extraction or classification
            - include_top = False >> not use Fully connected layer, output is featureMap 

            imgShape = (h,w,nCH)

        '''
        img_input = tf.keras.layers.Input(shape=imgShape)

        model = tf.keras.applications.VGG19(
            include_top=False, # include nn for classification or not
            input_tensor=img_input,
            input_shape=imgShape,
            pooling='avg')

    elif name == 'VGG-19_max':
        '''
            keras.applications already implement state-of-the-art-model and 
            offer option to user to set whether use as feature extraction or classification
            - include_top = False >> not use Fully connected layer, output is featureMap 

            imgShape = (h,w,nCH)

        '''
        img_input = tf.keras.layers.Input(shape=imgShape)

        model = tf.keras.applications.VGG19(
            include_top=False, # include nn for classification or not
            input_tensor=img_input,
            input_shape=imgShape,
            pooling='max')

    elif name == 'VGG-16':
        img_input = tf.keras.layers.Input(shape=imgShape)
        model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
            input_tensor=img_input,
            input_shape=imgShape,
            )
    
    else:
        ValueError('Model name was not recognite !')
        
    return model