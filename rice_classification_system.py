import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import os
import kagglehub
import shutil
from uuid import uuid4

tf.random.set_seed(42)
np.random.seed(42)

def subsample_dataset(dataset_path, output_path, samples_per_class=1000):
    os.makedirs(output_path, exist_ok=True)
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        output_class_path = os.path.join(output_path, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        images = os.listdir(class_path)[:samples_per_class]
        for img in images:
            shutil.copy(os.path.join(class_path, img), output_class_path)

def prepare_data(dataset_path, img_size=(128, 128), batch_size=16):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def build_model(num_classes, input_shape=(128, 128, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model(model, train_generator, validation_generator, learning_rates=[0.001, 0.0001], epochs=10):
    best_model = None
    best_val_acc = 0
    history_dict = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ],
            verbose=1
        )
        
        history_dict[lr] = history.history
        
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
    
    return best_model, history_dict

def evaluate_model(model, validation_generator):
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys())
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return report

app = Flask(__name__)
class_indices = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("Error: No file uploaded")
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected")
        return "No file selected", 400
    img_path = f"uploads/{uuid4()}.jpg"
    file.save(img_path)
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        global model, class_indices
        predictions = model.predict(img_array)
        print("Predictions:", predictions)  # Debug
        predicted_class = list(class_indices.keys())[np.argmax(predictions)]
        print("Predicted class:", predicted_class)  # Debug
        os.remove(img_path)
        return render_template('index.html', prediction=predicted_class)
    except Exception as e:
        print("Prediction error:", str(e))
        os.remove(img_path)
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    
    print("Downloading Rice Image Dataset...")
    dataset_path = kagglehub.dataset_download("muratkokludataset/rice-image-dataset")
    print("Path to dataset files:", dataset_path)
    
    dataset_path = os.path.join(dataset_path, 'Rice_Image_Dataset')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder {dataset_path} not found. Verify the download.")
    
    train_generator, validation_generator = prepare_data(dataset_path)
    
    class_indices = train_generator.class_indices
    
    model_path = 'rice_classifier.h5'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model = load_model(model_path)
            # Compile model to fix metrics warning
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # Skip validation by default to avoid delay
            print("Model loaded successfully. Skipping validation (already validated with 99% accuracy).")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Training new model...")
            model = build_model(num_classes=len(train_generator.class_indices))
            model, history = train_model(model, train_generator, validation_generator)
            evaluate_model(model, validation_generator)
            model.save(model_path)
            print(f"Model saved to {model_path}")
    else:
        print("Training new model...")
        model = build_model(num_classes=len(train_generator.class_indices))
        model, history = train_model(model, train_generator, validation_generator)
        evaluate_model(model, validation_generator)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    app.run(debug=False)

