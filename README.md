# Smart Trial Room

## Overview
The **Smart Trial Room** is a virtual try-on system that overlays a garment (T-shirt) on a user's body in real-time using computer vision techniques. This project utilizes **MediaPipe Pose**, **Haar Cascade Face Detection**, and **OpenCV** to detect body landmarks and accurately position the garment. The system also includes an edge detection mode for enhanced visualization.

## Features
- **Real-time Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces.
- **Pose Estimation**: Uses MediaPipe Pose to identify key body landmarks.
- **Garment Overlay**: Dynamically resizes and positions a T-shirt image on the user's body.
- **Edge Detection Mode**: Provides an optional visualization of edge-detected features.
- **User Interaction**:
  - Press 'e' to toggle edge detection mode.
  - Press 'q' to quit the application.

## Dependencies
Make sure you have the following libraries installed:
```sh
pip install opencv-python mediapipe pillow numpy absl-py
```

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/smart-trial-room.git
   cd smart-trial-room
   ```
2. Place the garment image (T-shirt) in the project directory (default: `rb_80999.png`).
3. Run the script:
   ```sh
   python smart_trial_room.py
   ```
4. Allow camera access and observe the garment overlay on your body in real time.
5. Press 'e' to enable edge detection mode.
6. Press 'q' to exit.

## How It Works
1. Captures real-time video from the webcam.
2. Detects the user's face using Haar Cascade.
3. Identifies body landmarks using MediaPipe Pose.
4. Calculates body dimensions and resizes the garment accordingly.
5. Overlays the garment using **alpha blending** to ensure a realistic fit.
6. Displays the final augmented frame with an option to enable edge detection.

## File Structure
```
smart-trial-room/
├── smart_trial_room.py   # Main script
├── rb_80999.png         # Garment image (T-shirt)
└── README.md            # Project documentation
```

## Troubleshooting
- **No garment overlay?** Ensure `rb_80999.png` exists in the project folder.
- **Camera not working?** Check your webcam permissions and ensure OpenCV can access the camera.
- **Garment misaligned?** Adjust the scaling factors in `overlay_garment()` if necessary.

## Future Enhancements
- Support for multiple garment types.
- Improved alignment using deep learning-based pose estimation.
- Real-time background removal for a cleaner trial experience.

## License
This project is open-source and available under the MIT License.

## Author
[Priyanshu Rai] - [priyanshurai439@gmail.com]

