#!/bin/bash

# Advanced LLM Flow Financial Cleaner v5.0 - Cloud Run Deployment Script
# Usage: ./deploy_v5.sh [project-id] [region] [api-key]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="ai-financial-cleaner-v5"
DEFAULT_REGION="us-central1"
DEFAULT_PROJECT=""

# Parse arguments
PROJECT_ID=${1:-$DEFAULT_PROJECT}
REGION=${2:-$DEFAULT_REGION}
ANTHROPIC_API_KEY=${3:-${CLAUDE_API_KEY:-$ANTHROPIC_API_KEY}}

echo -e "${PURPLE}🚀 Advanced LLM Flow Financial Cleaner v5.0 Deployment${NC}"
echo "================================================================"

# Validate inputs
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: Project ID required${NC}"
    echo "Usage: ./deploy_v5.sh [project-id] [region] [api-key]"
    echo "Example: ./deploy_v5.sh my-project us-central1 sk-ant-..."
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: No Anthropic API key provided. Service will run in mock mode.${NC}"
    echo "Set ANTHROPIC_API_KEY environment variable or pass as third argument for live AI."
fi

echo -e "${BLUE}📋 Deployment Configuration${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Version: v5.0 - Advanced LLM Flow"
echo "API Key: ${ANTHROPIC_API_KEY:+Provided}${ANTHROPIC_API_KEY:-Not provided (mock mode)}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker not found${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Set project
echo -e "${BLUE}🔧 Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${BLUE}🔌 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create temporary app.py for deployment
echo -e "${BLUE}📝 Preparing advanced cleaner for deployment...${NC}"
cp app_v5.py app.py

# Build and deploy
echo -e "${BLUE}🏗️  Building and deploying Advanced LLM Flow to Cloud Run...${NC}"

if [ -n "$ANTHROPIC_API_KEY" ]; then
    # Deploy with API key
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY,FLASK_ENV=production,ENABLE_AI=true,AI_CONFIDENCE_THRESHOLD=0.7,ENABLE_PARALLEL=true,MAX_WORKERS=4"
else
    # Deploy in mock mode
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production,ENABLE_AI=false,AI_CONFIDENCE_THRESHOLD=0.7,ENABLE_PARALLEL=true,MAX_WORKERS=4"
fi

# Get the service URL
echo -e "${BLUE}🔍 Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

# Restore original app.py
mv app.py app_v5.py

echo ""
echo -e "${GREEN}✅ Advanced LLM Flow Financial Cleaner v5.0 deployed successfully!${NC}"
echo "================================================================"
echo -e "${BLUE}🌐 Service URL:${NC} $SERVICE_URL"
echo -e "${BLUE}📊 Health Check:${NC} $SERVICE_URL/health"
echo -e "${BLUE}📈 Statistics:${NC} $SERVICE_URL/stats"
echo -e "${BLUE}⚙️  Configuration:${NC} $SERVICE_URL/config"
echo ""
echo -e "${YELLOW}🔧 Advanced Features Enabled:${NC}"
echo "• Intelligent Rule > Cache > LLM Processing Flow"
echo "• Source Tracking & Confidence Scoring"
echo "• Transaction Intelligence (Tags, Insights, Risk)"
echo "• Comprehensive Cost & Performance Tracking"
echo "• Enhanced DataFrame with Attribution"
echo "• Separate Intelligence Section (not in CSV)"
echo ""
echo -e "${PURPLE}🎯 Next Steps:${NC}"
echo "1. Test the service: curl $SERVICE_URL/health"
echo "2. Try the demo: curl -X POST $SERVICE_URL/demo"
echo "3. Use the advanced UI: interface_v5.html"
echo "4. Monitor performance: $SERVICE_URL/stats"
echo ""
echo -e "${GREEN}🚀 Your advanced financial data cleaner is ready!${NC}" 