

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project-name
```

### 2. Installing Dependencies

```bash


# Install dependencies
pip install -r requirements.txt


```

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

```

## 🚀 Running the Application



#### Start the Backend Server

```bash
# From backend directory
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The FastAPI backend will be available at:
- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

#### Start the Frontend Server

```bash
# From frontend directory (in a new terminal)
cd frontend
npm start
```

The React frontend will be available at: http://localhost:3000

