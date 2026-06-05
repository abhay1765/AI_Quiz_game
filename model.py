"""
AI Quiz Master - Machine Learning Models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_PATH = os.path.join("models", "performance_model.pkl")

CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Python Programming",
    "Cloud Computing",
    "Computer Science",
    "General Knowledge",
]

TOPIC_RESOURCES = {
    "Artificial Intelligence": [
        "Study AI fundamentals: search algorithms, knowledge representation",
        "Practice with AI problem-solving exercises and case studies",
        "Master expert systems, inference engines and planning algorithms",
    ],
    "Machine Learning": [
        "Review supervised vs unsupervised learning concepts",
        "Practice with sklearn datasets and build your own models",
        "Study advanced evaluation metrics: precision, recall, ROC, AUC",
    ],
    "Python Programming": [
        "Practice Python data structures: list, dict, set, tuple",
        "Learn Python OOP: classes, inheritance, decorators, magic methods",
        "Explore advanced Python: generators, context managers, metaclasses",
    ],
    "Cloud Computing": [
        "Review cloud service models: IaaS, PaaS, SaaS fundamentals",
        "Study AWS/Azure/GCP core services and their use cases",
        "Learn container orchestration with Docker and Kubernetes",
    ],
    "Computer Science": [
        "Review data structures: trees, graphs, hash tables, heaps",
        "Practice algorithm analysis and Big O complexity notation",
        "Study OS concepts: processes, threads, memory management",
    ],
    "General Knowledge": [
        "Read broadly across science, history, and geography topics",
        "Practice with trivia quizzes to reinforce factual knowledge",
        "Focus on foundational STEM and humanities concepts",
    ],
}


class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def _generate_synthetic_data(self, n_samples=600):
        np.random.seed(42)
        data = []
        labels = []

        for _ in range(n_samples // 3):
            data.append([
                np.random.uniform(20, 45),
                np.random.uniform(0.20, 0.48),
                np.random.uniform(20, 40),
                np.random.randint(1, 5),
                np.random.uniform(0.0, 0.2),
            ])
            labels.append("Beginner")

        for _ in range(n_samples // 3):
            data.append([
                np.random.uniform(45, 75),
                np.random.uniform(0.48, 0.78),
                np.random.uniform(12, 22),
                np.random.randint(3, 15),
                np.random.uniform(0.2, 0.5),
            ])
            labels.append("Intermediate")

        for _ in range(n_samples // 3):
            data.append([
                np.random.uniform(75, 100),
                np.random.uniform(0.78, 1.0),
                np.random.uniform(5, 14),
                np.random.randint(10, 50),
                np.random.uniform(0.5, 1.0),
            ])
            labels.append("Advanced")

        return np.array(data), np.array(labels)

    def train(self):
        X, y = self._generate_synthetic_data()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder}, MODEL_PATH)

        return {
            "train_accuracy": round(self.model.score(X_train, y_train), 3),
            "test_accuracy": round(self.model.score(X_test, y_test), 3),
        }

    def load(self):
        if os.path.exists(MODEL_PATH):
            saved = joblib.load(MODEL_PATH)
            self.model = saved["model"]
            self.label_encoder = saved["label_encoder"]
            self.is_trained = True
            return True
        return False

    def predict(self, avg_score, accuracy, avg_time, quizzes_taken, hard_ratio=0.3):
        if not self.is_trained:
            if not self.load():
                self.train()

        features = np.array([[avg_score, accuracy, avg_time, quizzes_taken, hard_ratio]])
        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        level = self.label_encoder.inverse_transform([prediction_idx])[0]

        class_probs = {
            cls: round(float(probabilities[i]), 3)
            for i, cls in enumerate(self.label_encoder.classes_)
        }

        return {
            "level": level,
            "confidence": round(float(max(probabilities)), 3),
            "probabilities": class_probs,
        }


def get_difficulty_for_accuracy(accuracy_percent):
    if accuracy_percent > 80:
        return "hard"
    elif accuracy_percent >= 50:
        return "medium"
    else:
        return "easy"


class RecommendationEngine:
    def analyze_weak_topics(self, quiz_history_df):
        if quiz_history_df is None or len(quiz_history_df) == 0:
            return CATEGORIES[:3]

        category_stats = (
            quiz_history_df.groupby("category")
            .agg(correct=("correct", "sum"), total=("total", "sum"))
            .reset_index()
        )
        category_stats["accuracy"] = category_stats["correct"] / category_stats["total"]
        overall_mean = category_stats["accuracy"].mean()
        weak = category_stats[category_stats["accuracy"] < overall_mean]["category"].tolist()
        return weak if weak else [category_stats.sort_values("accuracy").iloc[0]["category"]]

    def generate_recommendations(self, weak_topics, user_level="Beginner"):
        recommendations = []
        for topic in weak_topics[:3]:
            resources = TOPIC_RESOURCES.get(topic, [])
            idx = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}.get(user_level, 0)
            idx = min(idx, len(resources) - 1) if resources else 0
            tip = resources[idx] if resources else f"Study {topic} concepts"
            recommendations.append({
                "topic": topic,
                "suggestion": tip,
                "priority": "high" if topic == weak_topics[0] else "medium",
            })
        return recommendations

    def get_learning_insights(self, history_records):
        if not history_records:
            return {"trend": "no_data", "best_category": "N/A", "worst_category": "N/A", "improvement_rate": 0}

        df = pd.DataFrame(history_records)
        best_cat = worst_cat = "N/A"

        if "category" in df.columns and "accuracy" in df.columns:
            cat_acc = df.groupby("category")["accuracy"].mean()
            if len(cat_acc) > 0:
                best_cat = cat_acc.idxmax()
                worst_cat = cat_acc.idxmin()

        improvement = 0
        if "accuracy" in df.columns and len(df) >= 2:
            recent = df["accuracy"].tail(5).tolist()
            improvement = recent[-1] - recent[0] if len(recent) >= 2 else 0

        trend = "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable"

        return {
            "trend": trend,
            "best_category": best_cat,
            "worst_category": worst_cat,
            "improvement_rate": round(improvement * 100, 1),
            "total_quizzes": len(history_records),
        }


predictor = PerformancePredictor()
recommendation_engine = RecommendationEngine()


def initialize_models():
    if not predictor.load():
        result = predictor.train()
        print(f"[ML] Model trained — train acc: {result['train_accuracy']}, test acc: {result['test_accuracy']}")
    else:
        print("[ML] Model loaded from disk.")