🩺 AI Diabetic Retinopathy Screening System
A modular, team-based deep learning system for automated diabetic retinopathy detection from retinal fundus images.
👥 Team Structure
Table
Member	Role	Module
1	🧠 ML Lead	src/model/
2	🔍 Explainable AI	src/explainability/
3	🖼️ Image Processing	src/preprocessing/
4	⚙️ Backend Developer	src/api/
5	💻 Frontend Developer	frontend/
6	📊 Testing & Integration	tests/, docs/
🏗️ Architecture
plain
Frontend (HTML/JS)  →  Flask API  →  Preprocessing  →  AI Model  →  Grad-CAM  →  Result
🚀 Quick Start
bash
 1. Clone and setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Place model
# Download or train retinopathy_model.keras and place in models/

# 3. Run backend
cd src/api
python app.py

# 4. Open frontend
# Open frontend/index.html in browser or use Live Server
📁 Project Structure
See api_contracts.md for full directory tree and interface specifications.
🔗 API Documentation
POST /api/v1/predict — Upload image, get prediction + explanation
GET /api/v1/health — Health check
GET /api/v1/model/info — Model metadata
🧪 Testing
bash
pytest tests/ -v --cov=src --cov-report=html
📄 License
Academic / Educational Use
