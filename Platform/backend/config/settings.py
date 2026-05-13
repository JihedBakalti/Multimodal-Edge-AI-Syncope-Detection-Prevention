from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in some environments
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    # Load both local backend env and repo-root env for shared service vars.
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent.parent / ".env")
SHARED_SQLITE_PATH = os.getenv("PLATFORM_SQLITE_PATH", "").strip()
SECRET_KEY = os.getenv("PLATFORM_SECRET_KEY", "dev-only-secret-key")
DEBUG = os.getenv("PLATFORM_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.getenv("PLATFORM_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "corsheaders",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "channels",
    "apps.accounts",
    "apps.clinical",
    "apps.ingestion",
    "apps.messaging",
    "apps.reports",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": (
        {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("PLATFORM_DB_NAME", "platform_db"),
            "USER": os.getenv("PLATFORM_DB_USER", "platform_user"),
            "PASSWORD": os.getenv("PLATFORM_DB_PASSWORD", ""),
            "HOST": os.getenv("PLATFORM_DB_HOST", "127.0.0.1"),
            "PORT": os.getenv("PLATFORM_DB_PORT", "5432"),
            "OPTIONS": {"sslmode": os.getenv("PLATFORM_DB_SSLMODE", "prefer")},
        }
        if os.getenv("PLATFORM_DB_ENGINE", "sqlite").lower() == "postgres"
        else {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": SHARED_SQLITE_PATH or (BASE_DIR / "db.sqlite3"),
        }
    )
}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
AUTH_USER_MODEL = "accounts.User"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
    ),
}

CHANNEL_LAYERS = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
}

PLATFORM_INGEST_TOKEN = os.getenv("PLATFORM_INGEST_TOKEN", "platform-dev-token")
PLATFORM_INGEST_REPLAY_WINDOW_SEC = int(os.getenv("PLATFORM_INGEST_REPLAY_WINDOW_SEC", "300"))

# CORS: comma-separated list, e.g. "https://your-app.vercel.app,https://preview.vercel.app"
# If unset, defaults to local Vite ports only.
_local_cors_origins = [
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
_platform_cors = os.getenv("PLATFORM_CORS_ALLOWED_ORIGINS", "").strip()
if _platform_cors:
    CORS_ALLOWED_ORIGINS = [x.strip() for x in _platform_cors.split(",") if x.strip()]
else:
    CORS_ALLOWED_ORIGINS = list(_local_cors_origins)

# If you use session cookies cross-site (less common with this API’s Basic header flow), set explicitly.
_csrf = os.getenv("PLATFORM_CSRF_TRUSTED_ORIGINS", "").strip()
if _csrf:
    CSRF_TRUSTED_ORIGINS = [x.strip() for x in _csrf.split(",") if x.strip()]
