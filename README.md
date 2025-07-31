# Instagram Bio Classifier API

A FastAPI service that classifies Instagram bios to identify Christian profiles using AI-powered analysis.

## 🚀 Features

- **Bio Classification**: Analyze Instagram bios to identify Christian profiles
- **Configurable Criteria**: Modify the classification prompt through API endpoints
- **AI-Powered**: Uses OpenAI GPT models for intelligent analysis
- **Fast Processing**: Quick keyword matching with fallback to AI analysis
- **Persistence**: Prompts saved to file and survive API restarts
- **Default Fallback**: Always provides a working default prompt

## 📋 API Endpoints

### POST `/classify`
Classify a list of Instagram bios.

**Request:**
```json
{
  "bios": [
    "Jesus is my savior ✝️",
    "Coffee lover and travel enthusiast",
    "Christian, wife, mom"
  ]
}
```

**Response:**
```json
{
  "results": ["yes", "no", "yes"]
}
```

### GET `/prompt`
Get the current classification prompt.

**Response:**
```json
{
  "prompt": "For each numbered Instagram bio below, reply **yes** or **no**..."
}
```

### PUT `/prompt`
Update the classification prompt.

**Request:**
```json
{
  "prompt": "Your custom classification prompt here..."
}
```

**Response:**
```json
{
  "prompt": "Your custom classification prompt here..."
}
```

### POST `/prompt/reset`
Reset the classification prompt to default.

**Response:**
```json
{
  "prompt": "Default classification prompt..."
}
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd classify_api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. **Run the server**
   ```bash
   python run_local.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📁 Project Structure

```
classify_api/
├── app/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application and endpoints
│   └── model_classification.py   # Classification logic and persistence
├── data/                         # Auto-created directory
│   └── classification_prompt.json # Persistent prompt storage
├── requirements.txt              # Python dependencies
├── run_local.py                  # Local development server
├── test_api.py                   # API testing script
├── test_persistence.py           # Persistence testing script
├── setup.sh                      # Setup automation script
├── Dockerfile                    # Docker configuration
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI-powered classification
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Classification Logic

The service uses a two-stage approach:

1. **Quick Keyword Check**: Fast matching against Christian keywords and Bible references
2. **AI Analysis**: For uncertain cases, sends bios to OpenAI GPT for intelligent classification

### Prompt Persistence

- Prompts are automatically saved to `data/classification_prompt.json`
- Changes persist across API restarts
- Default prompt is used if no saved prompt exists
- File is auto-created on first save

## 🧪 Testing

### Test the API
```bash
python test_api.py
```

### Test Persistence
```bash
python test_persistence.py
```

### Manual Testing
```bash
# Get current prompt
curl http://localhost:8000/prompt

# Update prompt
curl -X PUT http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your custom prompt here"}'

# Reset to default
curl -X POST http://localhost:8000/prompt/reset

# Classify bios
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"bios": ["Jesus is my savior", "Coffee lover"]}'
```

## 🐳 Docker

### Build and Run
```bash
docker build -t classify-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key classify-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  classify-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
```

## 🔒 Security Considerations

- **API Key Management**: OpenAI API key stored in environment variables
- **Input Validation**: All user input validated on both frontend and backend
- **Error Messages**: Generic error messages to avoid information leakage
- **File Permissions**: Prompt file stored with appropriate read/write permissions
- **Data Integrity**: JSON validation ensures prompt file integrity

## 📊 Performance

- **API Timeouts**: 30-second timeout for classification API calls
- **Caching**: No caching implemented; each request fetches fresh data
- **Concurrent Requests**: Limited by FastAPI's async handling
- **Memory Usage**: Minimal; only stores current prompt in memory
- **File I/O**: Prompt persistence uses efficient JSON file storage
- **Startup Time**: Fast prompt loading from file on API startup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: OpenAI API key not configured
   ```
   - Solution: Set `export OPENAI_API_KEY="your-key"`

2. **Port Already in Use**
   ```
   Error: Address already in use
   ```
   - Solution: Change port in `run_local.py` or kill existing process

3. **Permission Denied for Data Directory**
   ```
   Error: Permission denied
   ```
   - Solution: Check file permissions or run with appropriate user

4. **Prompt File Corruption**
   ```
   Error: Invalid JSON in prompt file
   ```
   - Solution: Delete `data/classification_prompt.json` to reset to default

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=debug
python run_local.py
```

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `http://localhost:8000/docs`
- Review the troubleshooting section above 