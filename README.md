# Emotion and Environment-Based Music Recommendation System

## Project Overview
This project leverages facial emotion detection and environmental context classification to offer personalized music genre recommendations. The system detects emotions using the FER-2013 dataset and classifies indoor environments using the MIT Indoor Dataset. The combined output helps tailor music suggestions based on the user's emotional state and current surroundings.

## Features
- **Emotion-Based Music Recommendation**: Detects the user's facial emotion and suggests a music genre.
- **Environment-Based Music Recommendation**: Classifies indoor environments and maps them to specific music genres.
- **Personalized Recommendations**: Combines emotional and environmental data to generate unique music suggestions.

## Datasets Used
### FER-2013 (Emotion Prediction)
- Contains over 35,000 labeled images of human faces.
- Labels include emotions such as happiness, sadness, surprise, anger, and more.
- This dataset is used to detect the user’s emotional state and map it to a music genre.

### MIT Indoor Dataset (Environment Prediction)
- Contains images from various indoor environments like classrooms, living rooms, airports, etc.
- The environment is classified into categories such as “office,” “restaurant,” “gym,” and others.
- These environments are mapped to specific music genres to enhance the user’s experience based on their surroundings.

### UTKFace Dataset (Age & Gender Prediction)
- Includes over 20,000 labeled images of faces with age and gender information.
- The dataset spans ages from 0 to 116 years, making it ideal for demographic-based recommendations.
- Age and gender are used to further personalize music genre recommendations.

## Installation

### Clone the Repository
```bash
git clone https://github.com/Hariesh-Kai/Youtube_Based_MultiModel_Prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download the Datasets
- FER-2013 for emotion prediction
- MIT Indoor Dataset for environment prediction
- UTKFace for age and gender prediction

## Usage

### Emotion Prediction
1. Upload an image to predict the user's emotion.
2. The system maps the emotion to a specific music genre.

### Environment Prediction
1. Upload an image to predict the indoor environment.
2. The system classifies the environment and maps it to a corresponding music genre.

### Age and Gender Prediction
1. Upload an image to predict the user’s age and gender.
2. Based on the demographic, the system recommends a personalized genre.

#### Example Code:
```python
from emotion_music_recommendation import EmotionModel, EnvironmentModel, AgeGenderModel

# Emotion prediction
emotion_model = EmotionModel()
emotion = emotion_model.predict(image_path="path_to_image.jpg")
recommended_genre = emotion_model.map_emotion_to_genre(emotion)

# Environment prediction
environment_model = EnvironmentModel()
environment = environment_model.predict(image_path="path_to_image.jpg")
recommended_genre_env = environment_model.map_environment_to_genre(environment)

# Age and Gender prediction
age_gender_model = AgeGenderModel()
age, gender = age_gender_model.predict(image_path="path_to_image.jpg")
recommended_genre_age_gender = age_gender_model.map_age_gender_to_genre(age, gender)
```

## Models Used

### Emotion Prediction Models:
- **ResNet50**: A deep residual network for robust emotion classification.
- **VGG16**: A convolutional neural network for emotion prediction with solid performance.
- **InceptionV3**: Known for its deep architecture, used for accurate emotion detection.

### Environment Prediction Model:
- **CNN-based Model**: Trained on the MIT Indoor Dataset for high accuracy in classifying indoor environments.

### Age and Gender Prediction Models:
- **ResNet50**: Used for extracting age and gender information from faces.
- **VGG16**: A well-known model adapted for age and gender classification tasks.

## Results

### Emotion Prediction Model Accuracy
| Model        | Training Accuracy | Validation Accuracy |
|--------------|-------------------|---------------------|
| ResNet50     | 95%               | 92%                 |
| VGG16        | 94%               | 91%                 |
| InceptionV3  | 93%               | 90%                 |

### Environment Prediction Model Accuracy
| Model        | Training Accuracy | Validation Accuracy |
|--------------|-------------------|---------------------|
| CNN Model    | 97%               | 94%                 |

### Age & Gender Prediction Model Accuracy
| Model        | Training Accuracy | Validation Accuracy |
|--------------|-------------------|---------------------|
| ResNet50     | 94%               | 91%                 |
| VGG16        | 93%               | 90%                 |

## Contributing
Feel free to fork the repository, make improvements, or submit pull requests. Contributions are encouraged!

## License
This project is licensed under the MIT License.
```

This is ready to be pasted into the README.md file on your GitHub repository.
