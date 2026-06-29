## <samp>MODEL REQUIREMENTS (SAPIENS-2B)</samp>

This project utilizes Meta's **Sapiens-2B** foundation model for high-resolution Human Pose Estimation and the extraction of the gymnast's spatial coordinates.

Due to their massive size, the model weights are not included in this repository. To process raw videos and create your own evaluations, you must download the model manually:

#### <samp> 1  *DOWNLOAD THE WEIGHTS*</samp>
Click the following link https://huggingface.co/facebook/sapiens and scroll down to the pose estimation weights downloads, from Sapiens-0.3B to Sapiens-2B.

#### <samp> 2  *PLACE THE MODEL IN YOUR WORKSPACE*</samp>
Once downloaded, move the `.pth` (or `.safetensors`) file to your local project folder, specifically inside the designated checkpoints directory (`backend/hpe/pose/checkpoints/`).

> _<samp>**READY TO GO:**</samp> Once the model is in place, the system will automatically detect the keypoints, allowing you to generate the evaluated videos seamlessly._