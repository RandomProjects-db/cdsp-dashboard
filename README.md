# 🎓 Student Engagement Analytics Dashboard

A comprehensive analytics dashboard for student engagement and performance analysis, featuring multiple UI/UX implementations and machine learning-powered insights.

## 📊 Overview

This project provides three different dashboard implementations with varying UI/UX designs, all analyzing the same student engagement dataset to predict academic performance and provide actionable insights for educational stakeholders.

### 🎯 Key Features

- **Machine Learning Predictions**: 91.4% correlation between engagement metrics and academic performance
- **Interactive Data Visualization**: Plotly-powered charts and graphs
- **Risk Assessment**: Real-time student performance risk evaluation
- **Stakeholder-Focused Insights**: Tailored views for different educational roles
- **Multiple UI Themes**: Three distinct interface designs to choose from

## 🚀 Dashboard Versions

### 1. Fixed Analytics Dashboard (`fixed_analytics_dashboard.py`)

- **Focus**: Milestone 4 Communication Artifact
- **Design**: Clean, professional interface
- **Features**:
  - Stakeholder-specific views
  - Real-time risk assessment
  - Evidence-based intervention recommendations
  - Proper data leakage validation

### 2. Modern Analytics Dashboard (`modern_analytics_dashboard.py`)

- **Focus**: Modern UI/UX with light theme
- **Design**: Glassmorphism effects, minimalist approach
- **Features**:
  - Light theme with vibrant accents
  - Floating navigation panels
  - Animated data visualizations
  - Modern typography and micro-interactions

### 3. New Analytics Dashboard (`new_analytics_dashboard.py`)

- **Focus**: Enhanced features and advanced analytics
- **Design**: Comprehensive interface with advanced controls
- **Features**:
  - Multi-dimensional data exploration
  - Advanced filtering capabilities
  - Uncertainty quantification
  - Automated report generation
  - Statistical hypothesis testing

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning models
- `joblib` - Model persistence
- `numpy` - Numerical computing
- `scipy` - Statistical functions

## 🏃‍♂️ Getting Started

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cdsp-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run any dashboard version**
   ```bash
   # Option 1: Fixed Analytics Dashboard
   streamlit run fixed_analytics_dashboard.py
   
   # Option 2: Modern Analytics Dashboard  
   streamlit run modern_analytics_dashboard.py
   
   # Option 3: New Analytics Dashboard
   streamlit run new_analytics_dashboard.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
cdsp-dashboard/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── cleaned_sed_dataset.csv            # Student engagement dataset
├── best_random_forest_classifier.joblib # Trained ML model
├── fixed_analytics_dashboard.py       # Version 1: Communication-focused
├── modern_analytics_dashboard.py      # Version 2: Modern UI/UX
└── new_analytics_dashboard.py         # Version 3: Enhanced features
```

## 🎯 Target Audiences

### Educational Technology Directors

- Technical implementation insights
- System integration guidance
- Performance metrics and analytics

### Academic Affairs Leadership

- Strategic decision-making support
- Institutional performance overview
- Resource allocation insights

### Student Success Coordinators

- Intervention protocols
- Risk assessment tools
- Student progress monitoring

### Faculty Members

- Student engagement monitoring
- Performance prediction tools
- Individual student insights

## 📊 Dataset Information

The dashboard analyzes student engagement data with the following key metrics:

- **Academic Performance**: Average marks, total marks
- **Login Patterns**: Weekend/weekday usage, time-based access patterns
- **Course Engagement**: Number of courses, attendance, resource views
- **Activity Metrics**: Forum posts, quiz completion, file downloads
- **Behavioral Patterns**: Days active, total events, unique course access

## 🔍 Key Insights

- **91.4% correlation** between engagement metrics and academic performance
- **Time-based patterns** significantly impact student success
- **Resource engagement** is a strong predictor of academic outcomes
- **Early intervention** possible through engagement pattern analysis

## 🛠️ Technical Details

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Performance**: High accuracy with proper validation
- **Features**: Multi-dimensional engagement metrics
- **Output**: Academic performance predictions with confidence scores

### Data Processing

- **Cleaning**: Automated data validation and preprocessing
- **Scaling**: StandardScaler for numerical features
- **Validation**: Proper train-test split to prevent data leakage

## 🎨 UI/UX Design Philosophy

Each dashboard version implements different design approaches:

1. **Fixed Dashboard**: Professional, stakeholder-focused interface
2. **Modern Dashboard**: Glassmorphism, light theme, contemporary design
3. **New Dashboard**: Feature-rich, comprehensive analytics interface

## 🚀 Deployment

The dashboards can be deployed using various platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Web app hosting
- **Docker**: Containerized deployment
- **Local**: Development and testing environment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with all dashboard versions
5. Submit a pull request

## 📄 License

This project is available for educational and research purposes.

## 📞 Support

For questions or support, please contact the development team or create an issue in the repository.

---

Built with ❤️ for educational analytics and student success
