# Advanced LLM Flow Financial Data Cleaner v5.0

## 🚀 Overview

This is an advanced implementation of the financial data cleaner with a sophisticated LLM flow that provides intelligent processing, comprehensive tracking, and enhanced data insights.

## 🎯 Key Features

### **Advanced LLM Flow**
- **Intelligent Processing Strategy**: Rule > Cache > LLM
- **Source Tracking**: Every decision is tracked with confidence scores
- **Cost Optimization**: Intelligent caching reduces redundant LLM calls
- **Performance Monitoring**: Comprehensive metrics and audit logs

### **Enhanced Data Processing**
- **Vendor Standardization**: Intelligent vendor name cleaning with source attribution
- **Category Classification**: AI-powered transaction categorization
- **Transaction Intelligence**: Tags, insights, risk assessment, and anomaly detection
- **Flexible Input**: Supports JSON, CSV, and various data formats

### **Production-Ready Features**
- **Comprehensive Tracking**: Cost, time, cache performance, and quality metrics
- **Audit Logs**: Complete processing trail for compliance
- **Error Handling**: Graceful degradation and detailed error reporting
- **Scalable Architecture**: Cloud Run deployment with auto-scaling

## 🏗️ Architecture

### **Processing Flow**
```
Input → Schema Validation → Column Analysis
Row Loop:
  ├─ Vendor Standardization
  │    └─ Rule > Cache > LLM → Add source + confidence
  ├─ Category Classification  
  │    └─ Rule > Cache > LLM → Add source + confidence
  ├─ Transaction Intelligence
  │    └─ Tags, Insights, Explainability (separate section)
Cache All LLM Results (Keyed)
Track LLM Cost / Time / Cache Hit Rate
Output Enhanced DataFrame + Summary Report + Audit Logs
```

### **Components**

#### **1. IntelligentCache**
- Keyed storage with MD5 hashing
- Separate caches for vendor and category operations
- Hit rate tracking and statistics
- Automatic cache key generation

#### **2. LLMTracker**
- Comprehensive cost tracking per operation
- Performance metrics and response times
- Success/failure rate monitoring
- Average response time calculation

#### **3. AdvancedLLMProcessor**
- Intelligent processing with source attribution
- Rule-based fallbacks for efficiency
- Confidence scoring for all decisions
- Transaction intelligence generation

#### **4. ProcessingResult**
- Structured results with source tracking
- Confidence scores and explanations
- Processing time and cost tracking
- Enum-based source identification

## 📁 File Structure

```
finances-cleaned/
├── production_cleaner_ai_v5.py    # Advanced cleaner implementation
├── app_v5.py                      # Flask API with advanced flow
├── interface_v5.html              # Enhanced UI for advanced features
├── deploy_v5.sh                   # Deployment script for v5
├── advanced_llm_components.py     # Core LLM flow components
├── llm_client.py                  # LLM client with caching
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container configuration
└── README_ADVANCED.md            # This file
```

## 🚀 Quick Start

### **1. Deploy the Advanced Cleaner**
```bash
# Deploy with API key
./deploy_v5.sh your-project-id us-central1 your-api-key

# Deploy in mock mode
./deploy_v5.sh your-project-id us-central1
```

### **2. Test the Service**
```bash
# Health check
curl https://your-service-url/health

# Demo with sample data
curl -X POST https://your-service-url/demo

# Process your data
curl -X POST https://your-service-url/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "merchant": ["Google Cloud", "Amazon AWS"],
      "amount": [150.00, 89.99]
    },
    "config": {
      "enable_ai": true,
      "ai_vendor_enabled": true,
      "ai_category_enabled": true,
      "enable_transaction_intelligence": true,
      "enable_source_tracking": true
    }
  }'
```

### **3. Use the Advanced UI**
Open `interface_v5.html` in your browser to access the enhanced interface with:
- Advanced configuration options
- Real-time processing feedback
- Comprehensive results display
- Transaction intelligence visualization
- Audit logs and performance metrics

## 🔧 Configuration Options

### **Processing Configuration**
```json
{
  "enable_ai": true,                    // Enable AI processing
  "ai_vendor_enabled": true,            // Vendor standardization
  "ai_category_enabled": true,          // Category classification
  "enable_transaction_intelligence": true, // Transaction insights
  "enable_source_tracking": true,       // Source attribution
  "ai_confidence_threshold": 0.7        // Minimum confidence for AI decisions
}
```

### **Environment Variables**
```bash
ANTHROPIC_API_KEY=sk-ant-...           # Anthropic API key
ENABLE_AI=true                         # Enable AI processing
AI_CONFIDENCE_THRESHOLD=0.7           # Confidence threshold
FLASK_ENV=production                   # Production environment
```

## 📊 Response Format

### **Enhanced Response Structure**
```json
{
  "cleaned_data": [...],               // Enhanced DataFrame with source tracking
  "summary_report": {
    "processing_summary": {
      "total_transactions": 100,
      "vendor_standardizations": 85,
      "category_classifications": 90,
      "llm_calls": 15,
      "cache_hit_rate": 0.85
    },
    "cost_analysis": {
      "total_cost": 0.15,
      "vendor_standardization_cost": 0.08,
      "category_classification_cost": 0.07
    },
    "performance_metrics": {
      "total_time": 2.5,
      "vendor_standardization_time": 1.2,
      "category_classification_time": 1.3
    },
    "cache_performance": {
      "vendor_hit_rate": 0.85,
      "category_hit_rate": 0.80,
      "total_vendor_requests": 100,
      "total_category_requests": 90,
      "cache_size": 150
    },
    "quality_metrics": {
      "average_vendor_confidence": 0.88,
      "average_category_confidence": 0.92
    }
  },
  "audit_logs": {
    "llm_requests": {...},
    "cache_operations": {...},
    "error_logs": [...]
  },
  "transaction_intelligence": {
    "tags_summary": ["high_value", "subscription"],
    "insights_summary": [...],
    "average_risk_score": 0.25,
    "anomaly_report": [...]
  },
  "schema_analysis": {
    "schema_valid": true,
    "column_mapping": {...},
    "processing_capabilities": {...}
  }
}
```

## 🎯 Advanced Features

### **1. Source Tracking**
Every processing decision includes:
- **Source**: `rule_based`, `cache`, or `llm`
- **Confidence**: Score from 0.0 to 1.0
- **Explanation**: Human-readable reasoning
- **Processing Time**: Performance metrics

### **2. Transaction Intelligence**
Separate from CSV data, includes:
- **Tags**: Automatic transaction categorization
- **Insights**: Business intelligence observations
- **Risk Assessment**: Transaction risk scoring
- **Anomaly Detection**: Unusual transaction identification

### **3. Intelligent Caching**
- **Keyed Storage**: MD5-hashed cache keys
- **Hit Rate Tracking**: Performance monitoring
- **Cost Optimization**: Reduces redundant LLM calls
- **Automatic Management**: Self-maintaining cache

### **4. Comprehensive Tracking**
- **Cost Analysis**: Per-operation cost breakdown
- **Performance Metrics**: Response times and throughput
- **Quality Metrics**: Confidence scores and accuracy
- **Cache Statistics**: Hit rates and efficiency

## 🔍 Monitoring & Analytics

### **Health Check**
```bash
curl https://your-service-url/health
```

### **Statistics**
```bash
curl https://your-service-url/stats
```

### **Configuration**
```bash
curl https://your-service-url/config
```

## 🛠️ Development

### **Local Testing**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the advanced cleaner
python app_v5.py

# Test with sample data
curl -X POST http://localhost:8080/demo
```

### **Customization**
1. **Add New Rules**: Extend `_apply_vendor_rules()` and `_apply_category_rules()`
2. **Enhance Intelligence**: Modify transaction intelligence methods
3. **Optimize Caching**: Adjust cache strategies in `IntelligentCache`
4. **Add Metrics**: Extend tracking in `LLMTracker`

## 📈 Performance Optimization

### **Caching Strategy**
- **Vendor Cache**: Keyed by vendor name
- **Category Cache**: Keyed by vendor + amount
- **Hit Rate Target**: >80% for optimal performance
- **Cache Size**: Automatic management

### **Cost Optimization**
- **Rule-Based First**: Use rules before LLM calls
- **Cache Lookup**: Check cache before LLM
- **Confidence Thresholds**: Skip low-confidence AI calls
- **Batch Processing**: Process similar items together

### **Performance Monitoring**
- **Response Times**: Track per-operation timing
- **Success Rates**: Monitor LLM call success
- **Cost Tracking**: Real-time cost monitoring
- **Quality Metrics**: Confidence score tracking

## 🔒 Security & Compliance

### **Data Privacy**
- **No Data Storage**: Processed data not persisted
- **Secure API Keys**: Environment variable management
- **Audit Logs**: Complete processing trail
- **Error Handling**: Secure error responses

### **Compliance Features**
- **Audit Trail**: Complete processing history
- **Source Attribution**: Transparent decision tracking
- **Confidence Scoring**: Quality assurance metrics
- **Error Logging**: Comprehensive error tracking

## 🚀 Deployment

### **Cloud Run Deployment**
```bash
# Deploy with advanced features
./deploy_v5.sh your-project-id us-central1 your-api-key

# Verify deployment
curl https://your-service-url/health
```

### **Environment Configuration**
```bash
# Required environment variables
ANTHROPIC_API_KEY=sk-ant-...
FLASK_ENV=production
ENABLE_AI=true
AI_CONFIDENCE_THRESHOLD=0.7
```

## 📚 API Reference

### **Endpoints**

#### **POST /process**
Main processing endpoint with advanced LLM flow.

**Request:**
```json
{
  "data": {
    "merchant": ["Google Cloud", "Amazon AWS"],
    "amount": [150.00, 89.99]
  },
  "config": {
    "enable_ai": true,
    "ai_vendor_enabled": true,
    "ai_category_enabled": true,
    "enable_transaction_intelligence": true,
    "enable_source_tracking": true
  }
}
```

**Response:** Enhanced response with comprehensive tracking and intelligence.

#### **POST /demo**
Demo endpoint with sample data processing.

#### **GET /health**
Health check endpoint.

#### **GET /stats**
Application statistics and performance metrics.

#### **GET /config**
Current configuration settings.

### **Error Handling**
- **400**: Invalid request format
- **413**: Dataset too large
- **500**: Internal processing error

## 🎯 Use Cases

### **1. Financial Data Cleaning**
- Vendor name standardization
- Transaction categorization
- Data quality improvement
- Anomaly detection

### **2. Business Intelligence**
- Transaction pattern analysis
- Risk assessment
- Cost optimization insights
- Performance monitoring

### **3. Compliance & Auditing**
- Complete audit trails
- Source attribution
- Quality assurance metrics
- Error tracking

## 🔮 Future Enhancements

### **Planned Features**
- **Machine Learning Integration**: Custom model training
- **Advanced Analytics**: Predictive insights
- **Real-time Processing**: Streaming data support
- **Multi-language Support**: International vendor recognition
- **Custom Rules Engine**: User-defined processing rules

### **Performance Improvements**
- **Parallel Processing**: Multi-threaded operations
- **Advanced Caching**: Redis integration
- **Batch Optimization**: Improved batching strategies
- **Memory Optimization**: Reduced memory footprint

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Review the audit logs
- Monitor performance metrics
- Contact the development team

---

**Advanced LLM Flow Financial Data Cleaner v5.0** - Intelligent processing with comprehensive tracking and enhanced insights. 