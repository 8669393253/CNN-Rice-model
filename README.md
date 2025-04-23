Rice Classification System
Overview
The Rice Classification System is a web application designed to classify five varieties of rice—Arborio, Basmati, Ipsala, Jasmine, and Karacadag—using a deep learning model. The system leverages a MobileNetV2-based convolutional neural network (CNN) trained on the Rice Image Dataset, achieving near-perfect accuracy as evidenced by the confusion matrix (100% correct predictions on the test set). The frontend features a modern, premium design with a dark/light theme toggle, drag-and-drop functionality, and separate buttons for uploading images and viewing results. The backend, built with Flask, processes image uploads and returns predictions in real-time.
This project is currently running locally on http://localhost:5000 and has not been deployed to any hosting platform.
Features

Image Classification: Upload an image of rice grains to classify it into one of five varieties: Arborio, Basmati, Ipsala, Jasmine, or Karacadag.
Modern Frontend: A sleek, premium UI with Poppins font, gradient backgrounds, animations, and a dark/light theme toggle.
Drag-and-Drop Support: Upload images by dragging and dropping or clicking to select from your device.
Two-Button Workflow: Separate "Upload Image" and "Result" buttons for a clear user experience.
High Accuracy: The model achieves 100% accuracy on the test set, as shown in the confusion matrix.
Local Setup: Runs on your local machine using Flask and TensorFlow.

Project Structure
The project is organized as follows:

rice_classification_system.py: The main Flask application handling routing, image processing, and predictions.
rice_classifier.h5: The pre-trained MobileNetV2 model for rice classification.
templates/: Contains the frontend HTML file.
index.html: The main webpage with the UI and JavaScript logic.


static/: Contains static assets like CSS.
style.css: Stylesheet for the premium UI design.


uploads/: Temporary folder for storing uploaded images (created at runtime).
requirements.txt: Lists the Python dependencies needed to run the project.

Prerequisites
Before setting up the project, ensure you have the following installed on your system:

Python 3.8 or higher
pip (Python package manager)
Git (optional, for cloning the project if hosted on GitHub)
A web browser (e.g., Chrome, Firefox) to access the application

Additionally, ensure you have at least 2 GB of free disk space for the virtual environment, dependencies, and the pre-trained model (rice_classifier.h5).
Environment Setup
Follow these steps to set up the project on your local machine:
1. Clone or Download the Project
If the project is hosted on GitHub, clone it to your local machine:

Open a terminal or command prompt.

Navigate to your desired directory (e.g., D:\Desktop).

Clone the repository (replace yourusername and rice-classifier with the actual repository details):
git clone https://github.com/yourusername/rice-classifier.git
cd rice-classifier



If you already have the project files on your machine (e.g., in D:\Desktop\rice classifier), skip this step and navigate to the project directory:
cd D:\Desktop\rice classifier

2. Set Up a Virtual Environment
To avoid dependency conflicts, create a virtual environment:

In the project directory (D:\Desktop\rice classifier), run:
python -m venv venv


Activate the virtual environment:

On Windows:
venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate


You should see (venv) in your terminal prompt, indicating the virtual environment is active.

3. Install Dependencies
Install the required Python packages listed in requirements.txt:

With the virtual environment activated, run:
pip install -r requirements.txt


This installs Flask, TensorFlow, NumPy, and other necessary libraries. Ensure you have an active internet connection, as this step downloads packages.


4. Verify Project Files
Ensure the following files are present in D:\Desktop\rice classifier:

rice_classification_system.py
rice_classifier.h5
templates/index.html
static/style.css
requirements.txt

The uploads/ folder will be created automatically when you run the application.
Running the Application
Once the environment is set up, follow these steps to run the application locally:

Activate the Virtual Environment (if not already active):

On Windows:
D:\Desktop\rice classifier\venv\Scripts\activate


On macOS/Linux:
source D:\Desktop\rice classifier\venv\bin\activate


Navigate to the Project Directory:
cd D:\Desktop\rice classifier


Run the Flask Application:

Start the Flask server by running:
python rice_classification_system.py


You should see output indicating the server is running, such as:
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


Access the Application:

Open a web browser and navigate to:
http://localhost:5000


The Rice Classification System webpage should load, displaying the description, how-to steps, and upload section.


Test the Application:

Upload an Image:
Drag and drop a rice image (JPG/PNG) into the upload area, or click to select an image from your device.
A preview of the image will appear, and the "Upload Image" button will enable.


Upload Image:
Click the "Upload Image" button to send the image to the backend for processing.
The "Result" button will enable after the upload completes.


View Result:
Click the "Result" button to display the predicted rice type (e.g., "Basmati").


Test with images from the Rice Image Dataset (e.g., located at kagglehub\datasets\muratkokludataset\rice-image-dataset\versions\1\Rice_Image_Dataset\).


Debugging:

Check the terminal for Flask logs (e.g., "Predicted class: Basmati").
Open the browser’s developer tools (F12) and go to the Console tab to view JavaScript logs (e.g., "Parsed prediction: Basmati").
If errors occur, ensure all files are in place and dependencies are installed correctly.



Dependencies
The project relies on the following Python libraries, listed in requirements.txt:

Flask (v2.0.1): For building the web application and handling routing.
TensorFlow (v2.11.0): For loading and running the pre-trained MobileNetV2 model.
NumPy (v1.23.5): For numerical operations during image preprocessing.

These versions are specified to ensure compatibility with the model and Flask setup. If you need to upgrade or change versions, test thoroughly to avoid breaking the application.
Additional Notes

Model Performance: The pre-trained model (rice_classifier.h5) achieved 100% accuracy on the test set, as shown in the confusion matrix. However, real-world images may vary in quality, so monitor predictions for consistency.
Dataset: The model was trained on the Rice Image Dataset, containing 15,000 images per class (75,000 total). Test images are available at kagglehub\datasets\muratkokludataset\rice-image-dataset\versions\1\Rice_Image_Dataset\.
Troubleshooting:
If the webpage doesn’t load, ensure the Flask server is running and you’re accessing http://localhost:5000.
If predictions fail, check terminal logs for errors (e.g., "Prediction error") and verify rice_classifier.h5 is in the project directory.
If the CSS doesn’t load, confirm style.css is in static/ and the browser cache is cleared (Ctrl+F5).


Future Improvements:
Add confidence scores to predictions for better user feedback.
Deploy the app to a platform like Vercel or Railway for public access.
Enhance the UI with additional features like a reset button or image zoom.
