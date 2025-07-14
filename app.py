import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('equation_solver_model.keras')

label_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '÷'
}

def preprocess_char(img):
    # Resize with aspect ratio and center on 28x28 canvas
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = max(1, int((w / h) * 20))  # Ensure at least 1 pixel
    else:
        new_w = 20
        new_h = max(1, int((h / w) * 20))  # Ensure at least 1 pixel

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    canvas = 255 - canvas
    canvas = canvas / 255.0
    canvas = canvas.reshape(1, 28, 28, 1).astype('float32')

    return canvas

def segment_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(thresh, axis=0)
    threshold = 5  # Adjust based on noise level
    empty_columns = projection < threshold

    char_indices = []
    start_idx = None
    for i, is_empty in enumerate(empty_columns):
        if not is_empty and start_idx is None:
            start_idx = i
        elif is_empty and start_idx is not None:
            if i - start_idx > 3:  # Ignore very narrow segments
                char_indices.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        char_indices.append((start_idx, len(empty_columns)))

    char_images = []
    for (start_x, end_x) in char_indices:
        char = thresh[:, start_x:end_x]

        # Crop top and bottom white spaces (tight vertical crop)
        vertical_projection = np.sum(char, axis=1)
        top_indices = np.where(vertical_projection > 0)[0]

        if len(top_indices) == 0:
            continue  # Skip empty regions

        top = top_indices[0]
        bottom = top_indices[-1]
        char = char[top:bottom+1, :]

        if char.shape[1] > 3 and char.shape[0] > 3:
            char_images.append(char)

    return char_images

def predict_equation(model, image):
    chars = segment_characters(image)
    equation = ""
    st.write("### Predictions for each character:")
    for i, char_img in enumerate(chars):
        processed = preprocess_char(char_img)
        prediction = model.predict(processed, verbose=0)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        predicted_label = label_map.get(pred_class, '')
        equation += predicted_label

        st.write(f"Character {i+1}: **{predicted_label}** with confidence: **{confidence * 100:.2f}%**")

    return equation

def main():
    st.title("📝 Handwritten Equation Calculator with Confidence")
    st.image("333.gif", use_container_width=True)
    st.write("Upload a handwritten equation image (digits and operators).")

    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Equation"):
            equation = predict_equation(model, img)
            st.markdown(f"### Full Predicted Equation: `{equation}`")

            try:
                result = eval(equation.replace('÷', '/'))
                st.success(f"Result: {result}")
            except Exception as e:
                st.error(f"Could not evaluate equation: {e}")

if __name__ == "__main__":
    main()
