# Simplex Playground

This is a monorepo containing the frontend and backend for the Simplex Playground application.

## Project Structure

- `frontend/`: React application (Vite)
- `backend/`: FastAPI application (Python)

## Local Development

### Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will run at `http://127.0.0.1:8000`.

### Frontend

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will run at `http://localhost:5173`.

## Deployment

### Backend (Railway)

1. Connect your GitHub repository to Railway.
2. In the service settings, set the **Root Directory** to `backend`.
3. Railway should automatically detect the `requirements.txt` and `Procfile`.
4. The `Procfile` command is: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`.

### Frontend (Vercel)

1. Connect your GitHub repository to Vercel.
2. In the project settings, set the **Root Directory** to `frontend`.
3. Vercel should automatically detect it as a Vite project.
4. **Environment Variables**:
   - Add `VITE_API_URL` and set it to your deployed Railway backend URL (e.g., `https://your-project.up.railway.app`).
