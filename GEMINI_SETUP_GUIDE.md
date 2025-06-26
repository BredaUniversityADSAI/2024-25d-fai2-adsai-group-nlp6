# 🤖 Gemini AI Setup Guide for Docker Containers

This guide helps you configure Google's Gemini AI for enhanced video summaries in your Docker deployment.

## ✅ What You'll Get

- **AI-Powered Summaries**: Intelligent 50-60 word video summaries
- **Context Awareness**: Analysis of both transcript and emotion data
- **Enhanced UX**: Professional summaries instead of basic fallbacks

## 🔧 Quick Setup (Recommended)

### Step 1: Get Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (starts with `AIza...`)

### Step 2: Configure Environment
Create a `.env` file in your project root:

```bash
# Add this line to your .env file
REACT_APP_GEMINI_API_KEY=AIzaSyC-your-actual-api-key-here
```

### Step 3: Restart Containers
```bash
# Stop containers
docker-compose down

# Start with new environment
docker-compose up -d
```

## 🛠️ Alternative Setup Methods

### Method 1: Direct Environment Variable
```bash
# Set environment variable and start
export REACT_APP_GEMINI_API_KEY="AIzaSyC-your-api-key"
docker-compose up -d
```

### Method 2: Runtime Configuration (Temporary)
If you can't rebuild containers, use the browser console:

```javascript
// Open browser console (F12) and run:
window.geminiSetup.setupApiKey('AIzaSyC-your-api-key');
```

> **Note**: This method is temporary and will reset when container restarts.

## 🔍 Verification

### Check Configuration Status
Open browser console (F12) and run:
```javascript
window.geminiSetup.checkConfiguration().then(console.log);
```

### Expected Responses
- ✅ `configured: true, status: 'working'` - Everything working
- ⚠️ `status: 'missing_api_key'` - Need to set API key
- ❌ `status: 'invalid_api_key'` - Check key format
- 🔄 `status: 'api_error'` - Network/quota issues

## 🚨 Troubleshooting

### API Key Not Working?
1. **Check Format**: Must start with `AIza` and be 39 characters
2. **Verify Permissions**: Enable "Generative Language API" in Google Cloud Console
3. **Check Quota**: Ensure you haven't exceeded free tier limits

### Container Issues?
```bash
# Check if environment variable is passed
docker exec emotion_frontend env | grep GEMINI

# Restart with fresh build
docker-compose down
docker-compose up --build -d
```

### Still Having Issues?
1. Check browser console for detailed error messages
2. Verify API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Test API key with a simple curl request:

```bash
curl -H 'Content-Type: application/json' \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     -X POST 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=YOUR_API_KEY'
```

## 📊 Impact on Summaries

### Without Gemini API
```
"This 5-minute video presents visual content with advanced emotion detection analysis, 
revealing expressions and patterns through AI-powered technology."
```

### With Gemini AI
```
"This engineering tutorial demonstrates building autonomous robots using Raspberry Pi 
and machine learning algorithms. The presenter maintains an enthusiastic and informative 
tone throughout, making complex robotics concepts accessible to viewers."
```

## 💰 Cost Information

- **Free Tier**: 15 requests per minute, 1,500 requests per day
- **Typical Usage**: 1 request per video analysis
- **Cost**: Free for most users, pay-per-use beyond limits

## 🔒 Security Notes

- **Never commit API keys** to version control
- **Use environment variables** for production
- **Monitor usage** in Google Cloud Console
- **Rotate keys regularly** for security

## 🚀 Production Deployment

For production environments:

```yaml
# docker-compose.prod.yml
services:
  frontend:
    environment:
      - REACT_APP_GEMINI_API_KEY=${GEMINI_API_KEY}
    secrets:
      - gemini_api_key

secrets:
  gemini_api_key:
    external: true
```

## 📞 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs frontend`
2. **Verify network**: Ensure container can reach Google APIs
3. **Test locally**: Try the same API key in a local development environment
4. **Fallback mode**: System will work without Gemini, just with basic summaries

---

🎯 **Quick Test**: After setup, analyze any video and check if the summary is detailed and contextual rather than generic. 