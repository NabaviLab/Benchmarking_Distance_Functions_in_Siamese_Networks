# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model, Model
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Concatenate
from keras import regularizers
import keras.backend as K
from keras.applications import ResNet50
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler

# %%
# Load the ResNet50 model
model_path = 'Backbone models'
resnet_model = load_model(model_path)
base_model = Model(resnet_model.input, resnet_model.layers[-2].output) # Vary the number of layers

# %%
# Define custom layers and functions
def custom_loss(y_true, y_pred):
    lambda_1 = 0.5
    lambda_2 = 0.5
    lambda_3 = 0.5
    entropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    l2_loss = tf.reduce_sum(tf.square(base_model.trainable_weights))
    l1_loss = tf.reduce_sum(tf.abs(base_model.trainable_weights))
    return entropy_loss + lambda_1 * l2_loss + lambda_2 * l1_loss

def pixel_wise_distance(vects):
    x, y = vects
    return K.abs(x - y)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = base_model(left_input)
    encoded_r = base_model(right_input)

    pixel_dist = Lambda(pixel_wise_distance)([encoded_l, encoded_r])
    euclidean_dist = Lambda(euclidean_distance)([encoded_l, encoded_r])

    concatenated = Concatenate()([encoded_l, encoded_r, pixel_dist, euclidean_dist])

    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    prediction = Dense(1, activation='sigmoid')(x)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net

def extract_identifier(filename):
    for id in ['LCC', 'LMLO', 'RCC', 'RMLO']:
        if id in filename:
            return id
    return None

def prepare_pairs_with_augmentation(data_path, image_size=(512, 512), augment=False):
    prior_images = []
    current_images = []
    labels = []

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for patient_folder in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_folder)
        try:
            sub_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
            prior_folder = next((f for f in sub_folders if f.endswith(('prior', 'P'))), None)
            current_folder = next((f for f in sub_folders if f.endswith(('current', 'C'))), None)
            
            if not prior_folder or not current_folder:
                continue

            prior_path = os.path.join(patient_path, prior_folder)
            current_path = os.path.join(patient_path, current_folder)
            
            prior_files = {extract_identifier(f): os.path.join(prior_path, f) for f in os.listdir(prior_path)}
            current_files = {extract_identifier(f): os.path.join(current_path, f) for f in os.listdir(current_path)}

            for identifier, current_img_path in current_files.items():
                if identifier and identifier in prior_files:
                    img_path_prior = prior_files[identifier]

                    img_prior = load_img(img_path_prior, target_size=image_size)
                    img_current = load_img(current_img_path, target_size=image_size)

                    img_prior = img_to_array(img_prior)
                    img_current = img_to_array(img_current)

                    if augment:
                        aug_iter_prior = datagen.flow(np.expand_dims(img_prior, 0), batch_size=1)
                        aug_iter_current = datagen.flow(np.expand_dims(img_current, 0), batch_size=1)
                        img_prior = next(aug_iter_prior)[0]
                        img_current = next(aug_iter_current)[0]

                    prior_images.append(img_prior)
                    current_images.append(img_current)

                    labels.append(1 if any(term in current_img_path for term in ['MASS', 'Calc', 'Arch']) else 0)
        except Exception as e:
            print(f"Error processing patient folder {patient_folder}: {e}")

    return np.array(prior_images), np.array(current_images), np.array(labels)

def print_image_pairs_for_verification(data_path, num_pairs=5):
    count = 0
    for patient_folder in sorted(os.listdir(data_path)):
        if count >= num_pairs:
            break
        patient_path = os.path.join(data_path, patient_folder)
        
        sub_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
        prior_folder = next((f for f in sub_folders if f.endswith(('prior', 'P'))), None)
        current_folder = next((f for f in sub_folders if f.endswith(('current', 'C'))), None)
        
        if not prior_folder or not current_folder:
            continue

        prior_path = os.path.join(patient_path, prior_folder)
        current_path = os.path.join(patient_path, current_folder)
        
        prior_files = {extract_identifier(f): os.path.join(prior_path, f) for f in os.listdir(prior_path)}
        current_files = {extract_identifier(f): os.path.join(current_path, f) for f in os.listdir(current_path)}

        for identifier, current_img_path in current_files.items():
            if identifier and identifier in prior_files and count < num_pairs:
                print(f"Pair {count+1}: Prior Image -> {prior_files[identifier]}, Current Image -> {current_img_path}")
                count += 1

def visualize_pair(prior_image, current_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(prior_image / 255)
    axs[0].title.set_text('Prior')
    axs[1].imshow(current_image / 255)
    axs[1].title.set_text('Current')
    plt.show()

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def cosine_scheduler(epoch, lr):
    initial_lr = 1e-2
    final_lr = 1e-5
    total_epochs = 25
    return final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / total_epochs))

# %%
# Main Code Block
input_shape = (1024, 1024, 3)
siamese_model = get_siamese_model(input_shape)

data_path = './Datasets/'
print_image_pairs_for_verification(data_path, num_pairs=5)

try:
    prior_images, current_images, labels = prepare_pairs_with_augmentation(data_path, augment=True)
except Exception as e:
    print(f"Error preparing image pairs: {e}")

if len(prior_images) > 0 and len(current_images) > 0:
    prior_train, prior_test, current_train, current_test, labels_train, labels_test = train_test_split(
        prior_images, current_images, labels, test_size=0.2, random_state=42)

    print(f"Loaded {len(prior_images)} prior images and {len(current_images)} current images.")

    visualize_pair(prior_images[0], current_images[0])

    siamese_model.summary()

    siamese_model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    lr_scheduler = LearningRateScheduler(cosine_scheduler)
    siamese_model.fit([prior_train, current_train], labels_train, batch_size=4, epochs=25, callbacks=[lr_scheduler])

    siamese_model.save('Save Model', include_optimizer=False)

    siamese_model.evaluate([prior_test, current_test], labels_test)
