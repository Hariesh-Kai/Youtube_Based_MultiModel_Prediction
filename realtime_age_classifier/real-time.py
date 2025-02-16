import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = load_model("D:/Music_v3/Emotion_Based_Music_Player_with_Age_Preference/age_classifier_model.h5")

# Define image size
IMG_SIZE = 128  # Ensure it matches your training input size

# Label encoder (Ensure you have the same encoding used during training)
age_categories = ["Child", "Teenager", "Adult", "Older Adult"]
label_encoder = LabelEncoder()
label_encoder.fit(age_categories)

def predict_age_category(frame):
    """ Preprocess frame and predict age category """
    IMG_SIZE = 64  # Match model's expected input size
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Resize to 64x64
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)

    prediction = model.predict(img)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]


# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display instructions
    cv2.putText(frame, "Press 'C' to Capture", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show live webcam feed
    cv2.imshow("Real-Time Age Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture and predict
    if key == ord('c'):
        age_category = predict_age_category(frame)
        print(f"Predicted Age Category: {age_category}")  # Print result in terminal

        # Display result on screen
        cv2.putText(frame, f"Age Category: {age_category}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Age Prediction", frame)

        cv2.waitKey(2000)  # Pause for 2 seconds to show result

    # Press 'q' to exit
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
