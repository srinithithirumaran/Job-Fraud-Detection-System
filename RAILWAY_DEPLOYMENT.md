# Railway Deployment Guide

## Prerequisites
- Railway account (https://railway.app)
- Git installed
- GitHub account (for connecting your repo)

## Quick Setup Steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - ready for Railway deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Connect to Railway

**Option A: Using Railway Dashboard**
1. Go to https://railway.app/dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository
5. Railway will auto-detect it as a Python Django app

**Option B: Using Railway CLI**
```bash
npm install -g @railway/cli
railway login
railway link (select your project)
railway up
```

### 3. Configure Environment Variables in Railway

Go to your Railway project → Variables tab and add:

```
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=your-app-name.up.railway.app,localhost
CSRF_TRUSTED_ORIGINS=https://your-app-name.up.railway.app
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
DATABASE_URL=your-database-url (if using PostgreSQL)
```

### 4. Set up Database (Optional)
- Railway provides PostgreSQL add-on
- In Railway dashboard → Plugins → PostgreSQL
- Copy the DATABASE_URL to your environment variables

### 5. Run Migrations
After deployment, connect to Railway:
```bash
railway run python manage.py migrate
railway run python manage.py createsuperuser
```

### 6. Access Your App
Your app will be live at: `https://your-app-name.up.railway.app`

## Important Notes

- ✅ Static files are handled by WhiteNoise
- ✅ Gunicorn is configured as the web server
- ✅ DEBUG is set to False in production
- ✅ CSRF protection enabled
- ✅ SSL/HTTPS redirect enabled (set SECURE_SSL_REDIRECT=True)

## Troubleshooting

**Build Fails:**
- Check `requirements.txt` has all dependencies
- Ensure `Procfile` is in root directory
- Check Railway logs for specific errors

**Static Files Not Loading:**
- Run: `railway run python manage.py collectstatic --noinput`
- WhiteNoise should serve them automatically

**Database Issues:**
- Verify DATABASE_URL is set correctly
- Run migrations: `railway run python manage.py migrate`

## Local Testing

Before deploying, test locally:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python manage.py runserver
```

Visit: http://localhost:8000
