#!/bin/bash
# Deploy to Google Cloud Run - Production Ready Script

set -e  # Exit on any error

# Configuration
PROJECT_ID="${1:-your-project-id}"
SERVICE_NAME="financial-cleaner-v2"
REGION="us-central1"
IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/noahs-project-cb100/$SERVICE_NAME"

echo "🚀 Deploying Financial Cleaner API to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Check if user is logged in to gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not logged in to gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo "🔧 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push the image
echo "🏗️  Building and pushing Docker image..."
gcloud builds submit --tag $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "FLASK_ENV=production,ENABLE_AI=true" \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: $SERVICE_URL"
echo "🔍 Health check: $SERVICE_URL/health"
echo ""
echo "📋 Next steps:"
echo "1. Set your ANTHROPIC_API_KEY in Cloud Run environment variables:"
echo "   gcloud run services update $SERVICE_NAME --region=$REGION --set-env-vars=ANTHROPIC_API_KEY=your_key_here"
echo ""
echo "2. Test the API:"
echo "   curl $SERVICE_URL/health"
echo ""
echo "3. For frontend connection, use: $SERVICE_URL"