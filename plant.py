import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Dense, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



# Define the U-Net model
def unet_model(input_size=(128, 128, 3), classes=12):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Bottleneck
    convm = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    convm = Conv2D(128, 3, activation='relu', padding='same')(convm)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(convm)
    up1 = concatenate([up1, conv1], axis=3)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    
    # Output for classification
    gap = GlobalAveragePooling2D()(conv2)
    output = Dense(classes, activation='softmax')(gap)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # using 20% of the data for validation

train_generator = train_datagen.flow_from_directory(
    '/home/cc/plant-seedlings-classification/train',  # 正确的训练数据路径
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training')


validation_generator = train_datagen.flow_from_directory(
    '/home/cc/plant-seedlings-classification/train',  # Corrected path to your training data
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation')


# Compile and train the model
model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('model-best.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50,
    callbacks=callbacks
)

# The 'train_dir' in ImageDataGenerator should be replaced with the actual path where your training images are stored.
