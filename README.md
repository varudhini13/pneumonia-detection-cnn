                                                       Pneumonia Detection from Chest X-rays using CNN




This project applies deep learning to classify chest X-ray images as Normal or Pneumonia using a custom-built Convolutional Neural Network (CNN) in TensorFlow/Keras. It demonstrates core concepts in image preprocessing, model design, training, and evaluation—perfect for showcasing medical imaging expertise.

------------------------------------------------

Business problem:

Pneumonia is a serious respiratory condition that can be life-threatening if not diagnosed and treated promptly. In many under-resourced areas, access to radiologists is limited, leading to delayed diagnoses and poor health outcomes.

Goal:
To assist healthcare professionals by building an AI-powered tool that automatically detects signs of pneumonia from chest X-ray images. This system can serve as a decision support tool, reducing diagnostic time and improving accuracy, especially in remote or overloaded healthcare settings.


------------------------------------------------

Overview

- Goal              : Binary image classification (Normal vs. Pneumonia)
- Dataset           : Chest X-ray dataset with grayscale images
- Input Size        : 64x64 pixels
- Frameworks        : TensorFlow, Keras
- Output            : Trained model (`pnemonia.keras`) capable of real-time predictions

------------------------------------------------

Model Architecture

| Layer Type      | Parameters                        |
|-----------------|------------------------------------|
| Conv2D          | 32 filters, 3x3 kernel, ReLU       |
| MaxPooling2D    | 2x2 pool size                      |
| Flatten         | —                                  |
| Dense           | 128 units, ReLU                    |
| Output (Dense)  | 1 unit, Sigmoid                    |

- Optimizer           : Adam  
- Metrics             : Accuracy  

---------------------------------------------------------

Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
├── val/
```

- Images rescaled using `ImageDataGenerator(rescale=1./255)`
- Validation split: 20% of training data

------------------------------------------------

Training

- Epochs                      : 25  
- Batch Size                  : 32  
- Color Mode                  : Grayscale  
- Validation                  : Handled via split in `image_dataset_from_directory()`

------------------------------------------------

Tech Stack

- Languages               : Python  
- Libraries               : TensorFlow, Keras, PIL, NumPy, Matplotlib  
- Environment             : Jupyter Notebook  

------------------------------------------------

Highlights

- End-to-end image classification pipeline
- Clean, minimal CNN architecture with solid results
- Real-world medical application
- Ideal as a resume/portfolio project
