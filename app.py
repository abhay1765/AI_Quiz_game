
"""
AI Quiz Master - Main Flask Application
========================================
Full-stack AI-powered adaptive quiz system with:
- User authentication (register/login/logout)
- Adaptive difficulty based on performance
- ML-powered skill level prediction
- AI recommendation engine
- Leaderboard and dashboard
"""

import os
import json
import random
import secrets
from datetime import datetime, date

import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from model import (
    predictor, recommendation_engine,
    get_difficulty_for_accuracy, initialize_models, CATEGORIES
)

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "aiquizmaster_secret_key_2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Enable optional passwordless login for easier dev/testing.
# Set environment variable PASSWORDLESS_LOGIN=false to disable in production.
app.config["PASSWORDLESS_LOGIN"] = os.environ.get("PASSWORDLESS_LOGIN", "true").lower() == "true"

db = SQLAlchemy(app)

# ─────────────────────────────────────────────
#  DATABASE MODELS
# ─────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    quizzes = db.relationship("QuizHistory", backref="user", lazy=True)
    recommendations = db.relationship("Recommendation", backref="user", lazy=True)


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    option1 = db.Column(db.String(300), nullable=False)
    option2 = db.Column(db.String(300), nullable=False)
    option3 = db.Column(db.String(300), nullable=False)
    option4 = db.Column(db.String(300), nullable=False)
    answer = db.Column(db.String(300), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    difficulty = db.Column(db.String(20), nullable=False)


class QuizHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    score = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100))
    difficulty = db.Column(db.String(20))
    correct_answers = db.Column(db.Integer, default=0)
    wrong_answers = db.Column(db.Integer, default=0)
    total_questions = db.Column(db.Integer, default=10)
    time_taken = db.Column(db.Integer, default=0)  # seconds
    date = db.Column(db.DateTime, default=datetime.utcnow)


class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    suggested_topic = db.Column(db.String(100), nullable=False)
    suggestion_text = db.Column(db.Text)
    priority = db.Column(db.String(20), default="medium")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_user_stats(user_id):
    """Aggregate quiz statistics for a user."""
    history = QuizHistory.query.filter_by(user_id=user_id).all()
    if not history:
        return {
            "total_quizzes": 0, "avg_accuracy": 0, "highest_score": 0,
            "total_correct": 0, "total_wrong": 0, "avg_score": 0,
            "level": "Beginner", "level_confidence": 0,
        }

    total_quizzes = len(history)
    avg_accuracy = sum(h.accuracy for h in history) / total_quizzes
    highest_score = max(h.score for h in history)
    avg_score = sum(h.score for h in history) / total_quizzes
    total_correct = sum(h.correct_answers for h in history)
    total_wrong = sum(h.wrong_answers for h in history)
    avg_time = sum(h.time_taken for h in history) / (total_quizzes * max(h.total_questions for h in history))

    hard_count = sum(1 for h in history if h.difficulty == "hard")
    hard_ratio = hard_count / total_quizzes

    # ML prediction
    ml_result = predictor.predict(
        avg_score=avg_score,
        accuracy=avg_accuracy,
        avg_time=avg_time,
        quizzes_taken=total_quizzes,
        hard_ratio=hard_ratio,
    )

    return {
        "total_quizzes": total_quizzes,
        "avg_accuracy": round(avg_accuracy * 100, 1),
        "highest_score": round(highest_score, 1),
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "avg_score": round(avg_score, 1),
        "level": ml_result["level"],
        "level_confidence": round(ml_result["confidence"] * 100, 1),
    }


def update_recommendations(user_id):
    """Refresh AI recommendations for a user."""
    history = QuizHistory.query.filter_by(user_id=user_id).all()
    stats = get_user_stats(user_id)

    if history:
        records = [{"category": h.category, "correct": h.correct_answers, "total": h.total_questions} for h in history]
        df = pd.DataFrame(records)
        weak_topics = recommendation_engine.analyze_weak_topics(df)
    else:
        weak_topics = CATEGORIES[:3]

    recs = recommendation_engine.generate_recommendations(weak_topics, stats["level"])

    # Clear old recommendations
    Recommendation.query.filter_by(user_id=user_id).delete()
    for r in recs:
        rec = Recommendation(
            user_id=user_id,
            suggested_topic=r["topic"],
            suggestion_text=r["suggestion"],
            priority=r["priority"],
        )
        db.session.add(rec)
    db.session.commit()


def load_questions_from_csv():
    """Seed the database from dataset/questions.csv if empty."""
    if Question.query.count() > 0:
        return

    csv_path = os.path.join("dataset", "questions.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        q = Question(
            question=str(row["question"]),
            option1=str(row["option1"]),
            option2=str(row["option2"]),
            option3=str(row["option3"]),
            option4=str(row["option4"]),
            answer=str(row["answer"]),
            category=str(row["category"]),
            difficulty=str(row["difficulty"]),
        )
        db.session.add(q)
    db.session.commit()
    print(f"[DB] Loaded {len(df)} questions from CSV.")


def scramble_answer(text):
    """Scramble each word in the answer to create a puzzle clue."""
    scrambled_words = []
    for word in str(text).split():
        letters = list(word)
        if len(letters) > 1:
            attempt = ''.join(letters)
            while attempt.lower() == word.lower():
                random.shuffle(letters)
                attempt = ''.join(letters)
            scrambled_words.append(attempt)
        else:
            scrambled_words.append(word)
    return ' '.join(scrambled_words)


def create_hint(text):
    """Create a mystery hint showing the first letter of each word."""
    hint_parts = []
    for word in str(text).split():
        if len(word) > 1:
            hint_parts.append(word[0] + '_' * (len(word) - 1))
        else:
            hint_parts.append(word)
    return ' '.join(hint_parts)


# ─────────────────────────────────────────────
#  ROUTES — AUTH
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template("register.html")

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        passwordless_allowed = app.config.get("PASSWORDLESS_LOGIN", False)

        # If user doesn't exist and passwordless is allowed, auto-create a user.
        if not user and passwordless_allowed:
            # create with a placeholder email and random password hash
            placeholder_email = f"{username}@local.invalid"
            user = User(
                username=username,
                email=placeholder_email,
                password_hash=generate_password_hash(secrets.token_urlsafe(16)),
            )
            db.session.add(user)
            db.session.commit()
            flash("Account auto-created (passwordless mode).", "info")

        # Allow normal password login, or optional passwordless login when enabled.
        if user and ((password and check_password_hash(user.password_hash, password))
                     or (not password and passwordless_allowed)):
            session["user_id"] = user.id
            session["username"] = user.username
            if not password and passwordless_allowed:
                flash("Logged in without password (passwordless mode).", "info")
            return redirect(url_for("dashboard"))
        else:
            if not user:
                flash("User not found.", "error")
            elif not password and not passwordless_allowed:
                flash("Password required.", "error")
            else:
                flash("Invalid credentials.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ─────────────────────────────────────────────
#  ROUTES — DASHBOARD
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    stats = get_user_stats(user_id)

    # Recent history
    history = QuizHistory.query.filter_by(user_id=user_id).order_by(QuizHistory.date.desc()).limit(10).all()

    # Recommendations
    recs = Recommendation.query.filter_by(user_id=user_id).all()

    # Chart data — accuracy per quiz (last 10)
    chart_labels = [h.date.strftime("%b %d") for h in reversed(history)]
    chart_data = [round(h.accuracy * 100, 1) for h in reversed(history)]

    # Category breakdown
    all_history = QuizHistory.query.filter_by(user_id=user_id).all()
    cat_data = {}
    for h in all_history:
        cat = h.category or "Unknown"
        if cat not in cat_data:
            cat_data[cat] = {"correct": 0, "total": 0}
        cat_data[cat]["correct"] += h.correct_answers
        cat_data[cat]["total"] += h.total_questions

    insights = recommendation_engine.get_learning_insights(
        [{"category": h.category, "accuracy": h.accuracy} for h in all_history]
    )

    return render_template(
        "dashboard.html",
        stats=stats,
        history=history,
        recommendations=recs,
        chart_labels=json.dumps(chart_labels),
        chart_data=json.dumps(chart_data),
        cat_data=json.dumps(cat_data),
        insights=insights,
        categories=CATEGORIES,
    )


# ─────────────────────────────────────────────
#  ROUTES — QUIZ
# ─────────────────────────────────────────────
@app.route("/quiz/start", methods=["POST"])
@login_required
def start_quiz():
    category = request.form.get("category", "")
    user_id = session["user_id"]

    # Determine adaptive difficulty
    history = QuizHistory.query.filter_by(user_id=user_id).order_by(QuizHistory.date.desc()).limit(5).all()
    if history:
        recent_accuracy = sum(h.accuracy for h in history) / len(history) * 100
        difficulty = get_difficulty_for_accuracy(recent_accuracy)
    else:
        difficulty = "easy"  # Start easy for new users

    # Fetch questions
    query = Question.query
    if category:
        query = query.filter_by(category=category)
    query = query.filter_by(difficulty=difficulty)
    questions = query.all()

    # Fallback: if not enough questions, relax difficulty filter
    if len(questions) < 10:
        query2 = Question.query
        if category:
            query2 = query2.filter_by(category=category)
        questions = query2.all()

    if len(questions) < 5:
        flash("Not enough questions available. Please try another category.", "warning")
        return redirect(url_for("dashboard"))

    selected = random.sample(questions, min(10, len(questions)))

    # Store quiz in session
    session["quiz_questions"] = [
        {
            "id": q.id,
            "question": q.question,
            "options": random.sample([
                q.option1,
                q.option2,
                q.option3,
                q.option4,
            ], 4),
            "answer": q.answer,
            "category": q.category,
            "difficulty": q.difficulty,
        }
        for q in selected
    ]
    session["quiz_category"] = category or "Mixed"
    session["quiz_difficulty"] = difficulty
    session["quiz_start_time"] = datetime.utcnow().isoformat()

    return redirect(url_for("quiz"))


@app.route("/quiz")
@login_required
def quiz():
    if "quiz_questions" not in session:
        return redirect(url_for("dashboard"))

    return render_template(
        "quiz.html",
        questions=session["quiz_questions"],
        category=session.get("quiz_category", "Mixed"),
        difficulty=session.get("quiz_difficulty", "easy"),
        total=len(session["quiz_questions"]),
    )


@app.route("/quiz/submit", methods=["POST"])
@login_required
def submit_quiz():
    user_id = session["user_id"]
    data = request.get_json(silent=True)

    if data is not None:
        answers = data.get("answers", {})
        time_taken = data.get("time_taken", 0)
    else:
        answers = request.form.to_dict(flat=True)
        time_taken = request.form.get("time_taken", 0)

    questions = session.get("quiz_questions", [])

    correct = 0
    wrong = 0
    for q in questions:
        user_ans = answers.get(f"answer_{q['id']}", "")
        if user_ans.strip().lower() == q["answer"].strip().lower():
            correct += 1
        else:
            wrong += 1

    total = len(questions)
    score = (correct / total) * 100 if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    # Save to DB
    quiz_record = QuizHistory(
        user_id=user_id,
        score=round(score, 2),
        accuracy=round(accuracy, 4),
        category=session.get("quiz_category", "Mixed"),
        difficulty=session.get("quiz_difficulty", "easy"),
        correct_answers=correct,
        wrong_answers=wrong,
        total_questions=total,
        time_taken=time_taken,
    )
    db.session.add(quiz_record)
    db.session.commit()

    # Update recommendations
    update_recommendations(user_id)

    # Store result in session for result page
    session["last_result"] = {
        "score": round(score, 1),
        "accuracy": round(accuracy * 100, 1),
        "correct": correct,
        "wrong": wrong,
        "total": total,
        "time_taken": time_taken,
        "category": session.get("quiz_category", "Mixed"),
        "difficulty": session.get("quiz_difficulty", "easy"),
    }

    if request.is_json:
        return jsonify({"status": "ok", "redirect": url_for("result")})
    return redirect(url_for("result"))


@app.route("/result")
@login_required
def result():
    result_data = session.get("last_result")
    if not result_data:
        return redirect(url_for("dashboard"))

    stats = get_user_stats(session["user_id"])
    recs = Recommendation.query.filter_by(user_id=session["user_id"]).limit(3).all()

    return render_template("result.html", result=result_data, stats=stats, recommendations=recs)


# ─────────────────────────────────────────────
#  ROUTES — LEADERBOARD
# ─────────────────────────────────────────────
@app.route("/leaderboard")
@login_required
def leaderboard():
    # Aggregate max score per user
    users = User.query.all()
    board = []
    for u in users:
        history = QuizHistory.query.filter_by(user_id=u.id).all()
        if not history:
            continue
        best_score = max(h.score for h in history)
        avg_acc = sum(h.accuracy for h in history) / len(history)
        board.append({
            "username": u.username,
            "best_score": round(best_score, 1),
            "avg_accuracy": round(avg_acc * 100, 1),
            "total_quizzes": len(history),
            "is_current": u.id == session["user_id"],
        })

    board.sort(key=lambda x: x["best_score"], reverse=True)
    for i, entry in enumerate(board):
        entry["rank"] = i + 1

    return render_template("leaderboard.html", board=board)


# ─────────────────────────────────────────────
#  ROUTES — PROFILE
# ─────────────────────────────────────────────
@app.route("/profile")
@login_required
def profile():
    user = User.query.get(session["user_id"])
    stats = get_user_stats(user.id)
    history = QuizHistory.query.filter_by(user_id=user.id).order_by(QuizHistory.date.desc()).limit(20).all()
    recs = Recommendation.query.filter_by(user_id=user.id).all()

    # Category performance
    cat_perf = {}
    for h in history:
        cat = h.category or "Unknown"
        if cat not in cat_perf:
            cat_perf[cat] = {"scores": [], "correct": 0, "total": 0}
        cat_perf[cat]["scores"].append(h.score)
        cat_perf[cat]["correct"] += h.correct_answers
        cat_perf[cat]["total"] += h.total_questions

    cat_summary = []
    for cat, vals in cat_perf.items():
        acc = vals["correct"] / vals["total"] * 100 if vals["total"] > 0 else 0
        cat_summary.append({
            "category": cat,
            "avg_score": round(sum(vals["scores"]) / len(vals["scores"]), 1),
            "accuracy": round(acc, 1),
            "quizzes": len(vals["scores"]),
        })

    return render_template(
        "profile.html",
        user=user,
        stats=stats,
        history=history,
        recommendations=recs,
        cat_summary=cat_summary,
        categories=CATEGORIES,
    )


# ─────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(get_user_stats(session["user_id"]))


@app.route("/api/questions/<category>/<difficulty>")
@login_required
def api_questions(category, difficulty):
    questions = Question.query.filter_by(category=category, difficulty=difficulty).all()
    return jsonify([{
        "id": q.id, "question": q.question, "category": q.category, "difficulty": q.difficulty
    } for q in questions])


# ─────────────────────────────────────────────
#  APP INITIALIZATION
# ─────────────────────────────────────────────
def create_app():
    with app.app_context():
        db.create_all()
        load_questions_from_csv()
        initialize_models()
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)


