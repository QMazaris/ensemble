# Docker Deployment Guide

This guide explains how to containerize and run the Ensemble Pipeline full-stack Python application using Docker and Docker Compose.

## 🚀 Quick Start

**For users on the same Wi-Fi network, run this single command:**

```bash
docker-compose up --build
```

The application will be available at:
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend (FastAPI)**: http://localhost:8000

## 📋 Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 2.0 or later)
- At least 4GB of available RAM
- At least 2GB of free disk space

## 🏗️ Architecture

The application consists of two main services:

### Backend Service
- **Technology**: FastAPI with Uvicorn
- **Port**: 8000
- **Base Image**: Python 3.11-slim
- **Health Check**: `/health` endpoint

### Frontend Service  
- **Technology**: Streamlit
- **Port**: 8501
- **Base Image**: Python 3.11-slim (shared with backend)
- **Health Check**: `/_stcore/health` endpoint

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root to customize the configuration:

```bash
# Application Ports
FRONTEND_PORT=8501
BACKEND_PORT=8000

# Host Configuration
BACKEND_HOST=0.0.0.0
FRONTEND_HOST=0.0.0.0

# Debug Mode (set to true for development)
DEBUG=false

# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Backend API Configuration
BACKEND_URL=http://backend:8000
```

### Custom Ports

To run on different ports, modify your `.env` file:

```bash
FRONTEND_PORT=3000
BACKEND_PORT=5000
```

Then run:
```bash
docker-compose up --build
```

## 🚀 Deployment Commands

### Development Mode
```bash
# Build and run with live reload
docker-compose up --build

# Run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Mode
```bash
# Build optimized images
docker-compose build --no-cache

# Run in production mode
docker-compose up -d

# Check service health
docker-compose ps
```

## 🌐 Network Access

### Local Network Access

To allow other devices on your Wi-Fi network to access the application:

1. **Find your local IP address:**
   ```bash
   # Windows
   ipconfig | findstr IPv4
   
   # macOS/Linux
   ifconfig | grep inet
   ```

2. **Access from other devices:**
   - Frontend: `http://YOUR_LOCAL_IP:8501`
   - Backend: `http://YOUR_LOCAL_IP:8000`

### Firewall Configuration

Ensure your firewall allows incoming connections on ports 8501 and 8000:

```bash
# Windows (run as administrator)
netsh advfirewall firewall add rule name="Ensemble Frontend" dir=in action=allow protocol=TCP localport=8501
netsh advfirewall firewall add rule name="Ensemble Backend" dir=in action=allow protocol=TCP localport=8000

# macOS
sudo pfctl -f /etc/pf.conf

# Linux (Ubuntu/Debian)
sudo ufw allow 8501
sudo ufw allow 8000
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8501
   
   # Kill the process or change ports in .env
   ```

2. **Build Failures**
   ```bash
   # Clean build cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

3. **Memory Issues**
   ```bash
   # Check Docker memory usage
   docker stats
   
   # Increase Docker memory limit in Docker Desktop
   ```

4. **Network Connectivity**
   ```bash
   # Test backend connectivity
   curl http://localhost:8000/health
   
   # Test frontend connectivity
   curl http://localhost:8501/_stcore/health
   ```

### Health Checks

Both services include health checks that run every 30 seconds:

```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose logs backend | grep health
docker-compose logs frontend | grep health
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f

# Debug container issues
docker-compose exec backend bash
docker-compose exec frontend bash
```

## 📁 Volume Mounts

The following directories are mounted as volumes:

- `./data` → `/app/data` (read-only)
- `./output` → `/app/output` (read-write)
- `./models` → `/app/models` (read-write)

This allows data persistence and sharing between the host and containers.

## 🔒 Security Considerations

### Production Deployment

1. **Use environment-specific .env files**
2. **Enable HTTPS with a reverse proxy (nginx/traefik)**
3. **Implement proper authentication**
4. **Use Docker secrets for sensitive data**
5. **Regular security updates**

### Network Security

```bash
# Restrict network access (production)
docker-compose -f docker-compose.prod.yml up -d
```

## 🛠️ Advanced Usage

### Custom Docker Images

Build individual services:

```bash
# Build backend only
docker build --target backend -t ensemble-backend .

# Build frontend only  
docker build --target frontend -t ensemble-frontend .
```

### Scaling Services

```bash
# Scale frontend instances
docker-compose up --scale frontend=3

# Use load balancer for multiple instances
```

### Development with Hot Reload

For development with code changes:

```bash
# Mount source code as volume
docker-compose -f docker-compose.dev.yml up
```

## 📊 Monitoring

### Resource Usage

```bash
# Monitor resource usage
docker stats

# View container information
docker-compose ps -a
```

### Performance Optimization

1. **Multi-stage builds** reduce image size
2. **Layer caching** speeds up rebuilds  
3. **Health checks** ensure service reliability
4. **Volume mounts** persist data efficiently

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Docker and Docker Compose logs
3. Ensure all prerequisites are met
4. Verify network connectivity and firewall settings

---

**Happy containerizing! 🐳** 