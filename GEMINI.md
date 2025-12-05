# GEMINI Project Analysis: Face Recognition Attendance System

## Project Overview

This project is a comprehensive student attendance system based on real-time, multi-algorithmic face recognition. It is built as a Python Flask web application with a dynamic, JavaScript-driven frontend.

The system is designed to be robust, employing a sophisticated, tiered approach to face recognition. It prioritizes modern deep learning models and falls back to traditional computer vision techniques, ensuring both accuracy and performance. The backend manages student data, face encodings, and attendance records in a SQLite database, while the frontend provides a rich user interface for real-time monitoring, student management, and history review.

**Key Technologies:**

*   **Backend:** Python, Flask
*   **Database:** SQLite
*   **Face Recognition (Backend):** A custom multi-tier system using `numpy` and `scikit-learn` for calculations. It appears to be designed to work with embeddings from models like ArcFace, and also includes traditional methods like LBPH (`opencv-python`).
*   **Face Detection (Backend):** Multi-method detection using OpenCV's Haar Cascades, Dlib's HOG detector, and MTCNN.
*   **Frontend:** HTML, CSS, JavaScript
*   **Face Recognition (Frontend):** `face-api.js` is used for real-time face detection and descriptor extraction directly in the browser.
*   **Key Python Libraries (Inferred):**
    *   `Flask`
    *   `numpy`
    *   `Pillow`
    *   `opencv-python`
    *   `scikit-learn`
    *   `torch` & `torchvision` (for the custom ArcFace model)
    *   `facenet-pytorch` (optional, for MTCNN)
    *   `dlib` (optional, for HOG detector)
    *   `tensorflow` (usage unclear, but imported)

## Building and Running

The project requires a Python environment with several key dependencies.

1.  **Install Dependencies:**
    *   The `requirements.txt` file is currently empty. Based on the project's source code, the following dependencies are required. Install them using pip:
    ```bash
    # It is highly recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install core libraries
    pip install Flask numpy Pillow scikit-learn opencv-python torch torchvision

    # Install optional libraries for advanced detection/recognition
    pip install facenet-pytorch dlib tensorflow
    ```
    *   **Note:** Installing `dlib` and `tensorflow` can have specific system requirements. Please refer to their official documentation.

2.  **Run the Application:**
    *   Once the dependencies are installed, you can start the Flask server with the following command:
    ```bash
    python main.py
    ```

3.  **Access the Application:**
    *   Open a web browser and navigate to `http://localhost:5000/login`.
    *   Default login credentials are:
        *   **Username:** `admin`
        *   **Password:** `admin123`

## Development Conventions

*   **Backend Architecture:** The backend is a monolithic Flask application. Business logic, database interactions, and API endpoints are all contained within `main.py`.
*   **Frontend Architecture:** The frontend consists of HTML templates rendered by Flask. Significant client-side logic, especially for real-time video processing and face detection, is handled by JavaScript written directly inside the `<script>` tags of the HTML files. `face-api.js` is the core library for in-browser computer vision.
*   **Database:** A single SQLite database file (`attendance.db`) is used for all data storage. The schema is created and managed directly within `main.py`.
*   **Face Encodings:** The system uses a hybrid approach. The frontend (`student_management.html`) generates face descriptors using `face-api.js` and sends them to the backend. The backend stores these descriptors as individual `.npy` files in the `uploads/encodings/` directory, named after the student's unique code (`ma_sv`).
*   **API:** The application exposes a RESTful API for all client-server communication, handling tasks like login, student management, face training, and real-time identification. All API routes are defined in `main.py`.
*   **Authentication:** The application uses a session-based authentication system. Most routes and API endpoints are protected and require a user to be logged in. Some routes are further restricted by user role (e.g., 'Admin').
