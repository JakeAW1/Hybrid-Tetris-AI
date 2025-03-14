# **TetrisAI - Hybrid model**

This project implements a hybrid model involving two AI models, DQN (Deep Q-Network) and CNNs (Convolutional Neural Networks) to attempt playing Tetris efficiently. 
- The DQN model makes gameplay decisions 
- The CNN model processes visual inputs by analyzing game states.
While the model did not achieve optimal gameplay performance, it provided valuable insights into the challenges of integrating reinforcement learning with deep learning techniques.

## Further Insights & Analysis  
For a deeper breakdown of the AI model's performance, challenges, and potential improvements, please refer to the dissertation document included in this repository.

## **Installation Guide**
**Requirements.txt will contain all required packages to run the environment.**

**Step-by-Step Guide to install dependencies for TetrisAI**

**1. Clone the Repository**
First, open a terminal (Command Prompt, PowerShell, or Terminal) and run:
git clone https://github.com/YOUR_GITHUB_USERNAME/TetrisAI.git
cd TetrisAI

**2. Create a Virtual Environment (Recommended)**
A virtual environment helps keep dependencies isolated:

**Windows:**
python -m venv venv
venv\Scripts\activate

**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate

**3. Install Dependencies**
Once inside the project folder, install the required Python libraries using:
pip install -r requirements.txt
If you don’t have pip installed, you can install it first with:
python -m ensurepip --default-pip

**4. Verify Installation**
To check if dependencies were installed correctly, run:
pip list
Make sure essential libraries like PyTorch, NumPy, OpenCV, and Matplotlib are listed.

**5. Running the AI Model**
Now, you can execute the AI model with:
python main.py

**6. (Optional) Troubleshooting Installation Issues**
If an error occurs, try upgrading pip and reinstalling:
python -m pip install --upgrade pip
pip install -r requirements.txt

If a specific library fails, install it manually:
pip install torch numpy opencv-python matplotlib


