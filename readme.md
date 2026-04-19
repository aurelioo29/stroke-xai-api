# Installation Guide

1. Clone the repository

```bash
git clone https://github.com/aurelioo29/stroke-xai-api
cd stroke-xai-api
```

2. Create virtual environment

mac / linux

```bash
python3 -m venv venv
source venv/bin/activate
```

windows

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create environment file

Create a `.env` file in the root project directory.

```bash
DATABASE_URL=postgresql://username:password@localhost:5432/stroke_xai_db
```

5. Create the database

```bash
CREATE DATABASE stroke_xai_db;
```

6. Run the development server

```bash
uvicorn app.main:app --reload
```
