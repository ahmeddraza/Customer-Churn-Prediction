# 🔮 ChurnAI - Intelligent Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-Powered Customer Churn Prediction with Dynamic Thresholds, SHAP Explainability, and Revenue Impact Analysis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Demo](#-demo)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Business Value](#-business-value)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

ChurnAI is an enterprise-grade customer churn prediction system that goes beyond traditional ML models. It combines **Random Forest classification** with **business-aware decision-making**, **SHAP explainability**, and **revenue impact analysis** to provide actionable insights for customer retention.

### Why ChurnAI?

Traditional churn models predict with a fixed 50% threshold. ChurnAI revolutionizes this by:

- 🎯 **Dynamic Thresholds**: Each customer gets a personalized prediction threshold based on their lifetime value
- 💰 **Revenue Impact**: Calculates CLV and revenue at risk for every customer
- 🔍 **SHAP Explainability**: Shows exactly which features are causing churn
- 📊 **ROI-Based Recommendations**: Suggests retention offers with expected return on investment
- 🏆 **Churn Categorization**: Identifies root causes (Competitor, Dissatisfaction, Price, Attitude)

---

## ✨ Features

### 🧠 Intelligent Prediction Engine
- **Random Forest Classifier** trained on customer behavior data
- **Dynamic threshold optimization** based on customer lifetime value (CLV)
- **Business-cost-aware predictions** (FP cost vs FN cost)
- Real-time churn probability calculation with risk levels

### 💎 Advanced Analytics
- **SHAP (SHapley Additive exPlanations)** for feature importance
- Top 10 risk factors driving churn for each customer
- Visual progress bars showing impact percentage
- Human-readable feature names

### 💰 Revenue Intelligence
- **Customer Lifetime Value (CLV)** calculation
- **Revenue at Risk** estimation
- **ROI analysis** for retention campaigns (Basic, Standard, Premium)
- Customer value tiering (High, Medium, Standard, Low)
- Priority level assignment (P1-Critical to P4-Low)

### 🎯 Churn Category Detection
- 5-category classification: Competitor, Dissatisfaction, Price, Attitude, Other
- Category-specific retention recommendations
- Targeted intervention strategies

### 🎨 Modern User Interface
- **Responsive design** - works on desktop, tablet, and mobile
- **Interactive form validation** with real-time feedback
- **Animated visualizations** for SHAP values
- **Professional gradient design** with glass morphism effects
- **Smart form dependencies** (auto-disable irrelevant fields)

---

## 🛠 Technology Stack

### Backend
- **Python 3.8+**
- **Flask 2.0+** - Web framework
- **scikit-learn** - Machine learning models
- **SHAP** - Model explainability
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **pickle** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with variables
- **JavaScript (ES6+)** - Interactive features
- **Font Awesome** - Icons
- **Google Fonts (Inter)** - Typography

### Machine Learning
- **Random Forest Classifier** - Churn prediction
- **Random Forest Classifier** - Category prediction
- **Label Encoding** - Target encoding
- **One-Hot Encoding** - Categorical features
- **Ordinal Encoding** - Ordinal features
- **Standard Scaler** - Feature scaling

---

## 📁 Project Structure

```
CHURN-PREDICTION/
│
├── 📄 app.py                          # Main Flask application
├── 📄 threshold_optimizer.py          # Business-aware threshold optimization
├── 📄 revenue_model.py                # CLV and revenue impact calculations
├── 📄 requirements.txt                # Python dependencies
├── 📄 .gitignore                      # Git ignore rules
├── 📄 README.md                       # Project documentation
│
├── 📁 templates/                      # HTML templates
│   └── index.html                     # Main UI template
│
├── 📁 static/                         # Static assets
│   ├── css/
│   │   └── style.css                  # Main stylesheet
│   └── js/
│       └── script.js                  # Frontend interactions
│
├── 📁 p_models/                       # Churn prediction models
│   ├── churn_model.pkl                # Trained Random Forest (churn)
│   ├── label_encoder.pkl              # Label encoder (churn classes)
│   ├── standard_scaler.pkl            # Feature scaler
│   ├── onehot_encoder.pkl             # One-hot encoder
│   ├── ordinal_encoder.pkl            # Ordinal encoder
│   └── feature_names.pkl              # Feature list
│
├── 📁 c_models/                       # Category prediction models
│   ├── category_model.pkl             # Trained Random Forest (category)
│   ├── label_encoder.pkl              # Label encoder (categories)
│   ├── onehot_encoder.pkl             # One-hot encoder
│   └── ordinal_encoder.pkl            # Ordinal encoder
│
└── 📁 data/                           # Data files (optional)
    └── ...
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ahmeddraza/churn-prediction.git
   cd churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   Ensure the following directories contain trained models:
   - `p_models/` - Churn prediction models
   - `c_models/` - Category prediction models

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

## 📖 Usage

### Making a Prediction

1. **Fill in Customer Information**
   - Personal details (age, gender, marital status)
   - Account information (tenure, contract type)
   - Services (phone, internet, streaming)
   - Financial data (monthly charge, total revenue)

2. **Submit for Analysis**
   Click "Analyze Customer Risk" button

3. **Review Results**
   - **Churn Prediction**: Churned/Stayed with probability
   - **Risk Level**: Critical/High/Medium/Low
   - **Dynamic Threshold**: Personalized threshold used
   - **Revenue Impact**: CLV and revenue at risk
   - **Top Risk Factors**: SHAP analysis showing what's driving churn
   - **Recommendations**: Actionable retention strategies with ROI

### Example Output

```
🎯 Prediction: Customer Churned
   Churn Probability: 72%
   Risk Level: Critical
   Dynamic Threshold: 0.28

💰 Revenue Impact:
   CLV: $2,150.00
   Revenue at Risk: $1,548.00
   Priority: P1 - Critical

🔍 Top Risk Factors:
   1. Month-to-Month Contract (7.55%)
   2. Short Tenure (4.69%)
   3. No Online Security (4.34%)
   
✅ Recommendations:
   • Offer Standard retention package (ROI: 156%)
   • P1 - Critical - URGENT intervention required
   • Category: Competitor - Counter-offer needed
```

---

## 🔬 How It Works

### 1. Data Preprocessing Pipeline

```python
Input Data (29 features)
    ↓
Text Normalization (lowercase, strip)
    ↓
Numerical Conversion
    ↓
Ordinal Encoding (contract, offer)
    ↓
One-Hot Encoding (16 categorical features)
    ↓
Feature Filtering & Reindexing (56 features)
    ↓
Standard Scaling (11 numerical features)
    ↓
Ready for Prediction
```

### 2. Dynamic Threshold Calculation

Unlike traditional models that use a fixed 0.5 threshold, ChurnAI calculates a personalized threshold for each customer:

```python
Formula: threshold = retention_cost / (retention_cost + CLV)

Thresholds by CLV:
├─ CLV ≥ $2000  → threshold ≥ 0.25 (Very Aggressive)
├─ CLV ≥ $1000  → threshold ≥ 0.30 (Aggressive)
├─ CLV ≥ $500   → threshold ≥ 0.35 (Moderate)
├─ CLV ≥ $200   → threshold ≥ 0.40 (Conservative)
└─ CLV < $200   → threshold ≥ 0.45 (Very Conservative)
```

**Business Logic**: High-value customers get lower thresholds (catch them early), while low-value customers need higher confidence before intervention.

### 3. SHAP Explainability

SHAP (SHapley Additive exPlanations) provides transparent insights into model predictions:

```python
For each prediction:
1. Calculate SHAP values for all features
2. Identify top 10 features with positive impact (pushing towards churn)
3. Convert to percentage contribution
4. Display with visual bars
```

### 4. Revenue Impact Model

```python
CLV Calculation:
├─ Simple Method: monthly_charge × remaining_months
└─ Advanced Method: (avg_historical_revenue × 0.6) + (current_charge × 0.4) × remaining_months

Revenue at Risk = CLV × churn_probability

ROI Analysis (for each offer tier):
├─ Expected loss without action = churn_prob × CLV
├─ Expected loss with retention = (churn_prob × 0.5) × CLV
├─ Revenue saved = expected_loss - expected_loss_with_retention
├─ Net benefit = revenue_saved - retention_cost
└─ ROI % = (net_benefit / retention_cost) × 100
```

---

## 💼 Business Value

### For Customer Retention Teams
- **Prioritize interventions** based on revenue impact
- **Personalized outreach** with category-specific strategies
- **ROI-driven campaigns** - only invest when profitable
- **Proactive retention** - catch churners early

### For Executives
- **Revenue protection** - quantify dollars at risk
- **Resource optimization** - focus on high-value customers
- **Cost-benefit analysis** - justify retention spending
- **Strategic insights** - understand why customers leave

### For Data Scientists
- **Explainable AI** - SHAP values provide transparency
- **Business-aligned ML** - thresholds based on real costs
- **Comprehensive evaluation** - beyond accuracy metrics
- **Reproducible pipeline** - documented preprocessing

---

## 📊 Model Performance

### Churn Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 56 (after encoding)
- **Classes**: 3 (Churned, Stayed, Joined)
- **Training Data**: Customer historical data

### Category Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 56 (after encoding)
- **Classes**: 5 (Competitor, Dissatisfaction, Price, Attitude, Other)
- **Training Data**: Churned customers with labeled categories

---

## 🔑 Key Concepts

### Dynamic Thresholds
Traditional ML models use a fixed 0.5 threshold. ChurnAI uses customer-specific thresholds based on business value.

### SHAP Values
SHAP values answer: "How much did each feature contribute to this prediction?" providing transparency and trust.

### ROI-Based Retention
Not all retention efforts are profitable. ChurnAI only recommends offers with positive ROI.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 📞 Contact & Support

- **Email**: ahemdraza810@gmail.com

---

<div align="center">

**Made with ❤️ and ☕ by the ChurnAI Team**

⭐ Star this repo if you find it useful!

[Back to Top](#-churnai---intelligent-customer-churn-prediction-system)

</div>
