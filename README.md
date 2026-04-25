# Nyay AI - Legal Tech Backend

**Description**
- **Purpose**: Backend for Nyay AI — connects petitioners with lawyers and manages case lifecycle (create cases, search, messaging, reviews, availability).
- **AI-based recommendation**: The service includes a recommendation endpoint (`/recommend-lawyers`) that computes a match score using a heuristic scorer (`calculate_score`) based on case description, budget, location and lawyer profile. This is the current lightweight "AI" layer and can be extended to a true ML/NLP recommender (embeddings, ranking model, or a separate recommender microservice) for improved personalized matches.

**How To Setup**
- **Prerequisites**: Python 3.11+, virtual environment, and a database (Postgres recommended; SQLite supported for quick tests).
- **Install dependencies**: `pip install -r requirements.txt`
- **Environment**: set `DATABASE_URL` (example values):

```env

# Postgres example
# DATABASE_URL=postgresql://username:password@localhost:5432/nyay_db
```

- **Run locally**:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **Notes**: The app uses `python-dotenv` to load a `.env` file if present. On first run the app creates tables and inserts sample data.

**Architecture (Backend)**
- **API**: `main.py` hosts a FastAPI application exposing endpoints for signup/login, case management, lawyer profile management, search, messaging, availability, reviews, bulk insert, health, and a recommendation endpoint.
- **Database & ORM**: SQLAlchemy models in `main.py` represent `User`, `LawyerProfile`, `Case`, `Review`, `Availability`, `Message`, and `DemoRequest`. The DB connection is configured via `DATABASE_URL`.
- **Recommendation (AI-based)**: A simple scorer model evaluates matches using specialization overlap, fees vs. budget, and location — producing a `match_score` and `reason` returned by `/recommend-lawyers`. This is implemented as an:
	- embeddings-based semantic search (SentenceTransformers / OpenAI embeddings) for better understanding of free-text `description`;
	- a learning-to-rank model that trains on past assignments, reviews, and engagement signals;
	- a dedicated recommender microservice to scale model inference and retraining.
- **Lifespan / Initialization**: App `lifespan` creates DB tables and seeds sample users, lawyers, availabilities and reviews on first run.
- **Search & Filters**: `/search-lawyers` supports filters (specialization, location, experience, fees, rating) and returns structured recommendation-like responses.
- **Extensibility**: The recommender can be migrated to a separate service, integrate cached embeddings, or use online learning from `Review` and assignment outcomes.

**Quick Endpoints Overview**
- **POST** `/signup` — create user
- **POST** `/login` — auth (simple password check)
- **POST** `/add-case` — petitioner adds case
- **POST** `/recommend-lawyers` — AI-style recommendations (top 5 by match score)
- **GET** `/search-lawyers` — filter/search lawyers
- **POST** `/send-message` — messaging between users

For implementation details, see `main.py`.

