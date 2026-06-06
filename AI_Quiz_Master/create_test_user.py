from app import create_app, db, User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    existing = User.query.filter_by(username='testuser').first()
    if existing:
        print('testuser already exists:', existing.username, existing.email)
    else:
        user = User(
            username='testuser',
            email='testuser@example.com',
            password_hash=generate_password_hash('Test@1234'),
        )
        db.session.add(user)
        db.session.commit()
        print('Created test user: username=testuser email=testuser@example.com password=Test@1234')
