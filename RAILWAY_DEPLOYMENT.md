# Railway Deployment Guide

## Prerequisites
- Railway account
- GitHub repository connected to Railway
- DeepSeek API key (optional - fallback mode available)

## Environment Variables

Set these in Railway Dashboard → Variables:

```bash
# Required
PORT=5000

# Optional - DeepSeek API for AI reports
DEEPSEEK_API_KEY=your_api_key_here
```

## Deployment Steps

1. **Connect Repository to Railway**
   - Go to Railway Dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure Environment Variables**
   - Go to project → Variables
   - Add `DEEPSEEK_API_KEY` (if you have one)
   - Railway automatically sets `PORT`

3. **Deploy**
   - Railway will automatically detect Python and use:
     - `runtime.txt` for Python version (3.11.7)
     - `requirements.txt` for dependencies
     - `Procfile` for startup command

4. **Verify Deployment**
   - Once deployed, visit: `https://your-app.up.railway.app/api/health`
   - Should return JSON with status information

## Health Check

```bash
curl https://your-app.up.railway.app/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-29T...",
  "database": {
    "connected": true,
    "error": null
  },
  "analyzer": true,
  "advisor": {
    "initialized": true,
    "api_available": true
  },
  "components": {
    "database_manager": true,
    "mdq_analyzer": true,
    "deepseek_advisor": true
  },
  "environment": {
    "python_version": "3.11.7...",
    "flask_env": "production"
  }
}
```

## Troubleshooting

### 1. Check Logs
```bash
# In Railway Dashboard
Project → Deployments → View Logs
```

Look for initialization messages:
- ✅ Database manager initialized successfully
- ✅ MDQ analyzer initialized successfully
- ✅ AI advisor initialized successfully
- SUCCESS: DeepSeek API client initialized

### 2. Common Issues

**Issue: 500 Error / HTML instead of JSON**
- Check logs for Python import errors
- Verify all dependencies in `requirements.txt`
- Check Python version compatibility

**Issue: AI advisor not initialized**
- Verify `DEEPSEEK_API_KEY` is set correctly
- Check logs for API initialization errors
- System will use fallback mode if API unavailable

**Issue: Database errors**
- Railway automatically provides persistent storage
- Database will be created on first run
- Check logs for SQLite errors

### 3. Test Endpoints

```bash
# Health check
curl https://your-app.up.railway.app/api/health

# User registration (test)
curl -X POST https://your-app.up.railway.app/api/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123","email":"test@test.com"}'

# Login
curl -X POST https://your-app.up.railway.app/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'
```

## Files Structure

```
.
├── app.py                 # Main Flask application
├── database.py           # Database manager
├── analyse.py            # MDQ analyzer
├── gptadvisor.py         # AI advisor (DeepSeek)
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version (3.11.7)
├── Procfile             # Railway startup command
├── .gitignore           # Git ignore rules
└── *.html               # Frontend files
```

## API Endpoints

### Public Endpoints
- `GET /api/health` - Health check
- `POST /api/register` - User registration
- `POST /api/login` - User login

### Protected Endpoints (require login)
- `GET /api/user/profile` - Get user profile
- `POST /api/test/mdq` - Submit MDQ test
- `GET /api/test/history` - Get test history
- `POST /api/ai/report` - Generate AI report
- `GET /api/ai/report/<id>` - Get specific report

## Notes

1. **Python Version**: Fixed to 3.11.7 in `runtime.txt` for stability
2. **Dependencies**: Locked versions in `requirements.txt` to avoid conflicts
3. **Database**: SQLite file persists in Railway's volume storage
4. **API Keys**: Set via environment variables, never commit to git
5. **Error Handling**: All `/api/*` endpoints return JSON (no HTML errors)

## Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify environment variables
3. Test `/api/health` endpoint
4. Review error messages in browser console
