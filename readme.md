# 🔮 ChurnAI - Intelligent Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-Powered Customer Churn Prediction with Dynamic Thresholds, SHAP Explainability, and Revenue Impact Analysis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Pipeline](#-pipeline)

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
- [Dual Model Architecture](#-dual-model-architecture)
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
- 🔄 **Dual Model System**: Separate pipelines for churn prediction and category classification

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
- **imblearn** - Handling imbalanced datasets

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
- **Simple Imputer** - Missing value handling
- **Yeo-Johnson Power Transformer** - Outlier handling

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
│   ├── ordinal_encoder.pkl            # Ordinal encoder
│   └── imputer.pkl                    # Missing value imputer
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
   git clone https://github.com/yourusername/churn-prediction-ml-webapp.git
   cd churn-prediction-ml-webapp
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
   - `p_models/` - Churn prediction models (6 files)
   - `c_models/` - Category prediction models (5 files)

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

🔍 Top Risk Factors (SHAP):
   1. Month-to-Month Contract (7.55%)
   2. Short Tenure (4.69%)
   3. No Online Security (4.34%)
   
🏷️ Churn Category: Competitor

✅ Recommendations:
   • Offer Standard retention package (ROI: 156%)
   • P1 - Critical - URGENT intervention required
   • Counter-offer: Match or beat competitor pricing
   • Highlight unique value propositions
```

---

## 🔬 How It Works

### Complete Data Pipeline

ChurnAI uses two separate but coordinated pipelines:

---

## 🔄 Dual Model Architecture

### Model 1: Churn Prediction Pipeline

```
Input Data (29 raw features)
    ↓
┌─────────────────────────────────────┐
│ 1. Missing Value Imputation        │
│    - SimpleImputer                  │
│    - Strategy: mean/mode            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Text Normalization               │
│    - Lowercase conversion           │
│    - Whitespace stripping           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Numerical Conversion             │
│    - Cast to numeric dtypes         │
│    - Handle type mismatches         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Outlier Handling                 │
│    - Yeo-Johnson Power Transform    │
│    - Normalize skewed distributions │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. Ordinal Encoding                 │
│    - contract: [month-to-month,     │
│      one year, two year]            │
│    - offer: [none, A, B, C, D, E]   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. One-Hot Encoding                 │
│    - 16 categorical features        │
│    - handle_unknown='ignore'        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 7. Feature Filtering & Reindexing   │
│    - Keep only training features    │
│    - Reindex to 56 features         │
│    - Fill missing with 0            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 8. Standard Scaling                 │
│    - Scale 11 numerical features    │
│    - Mean=0, Std=1                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 9. Churn Prediction                 │
│    - Random Forest Classifier       │
│    - Output: [Churned, Stayed,      │
│      Joined] probabilities          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 10. Dynamic Threshold Application   │
│     - Calculate customer CLV        │
│     - Compute optimal threshold     │
│     - Apply business logic          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 11. SHAP Explainability             │
│     - TreeExplainer                 │
│     - Feature attributions          │
│     - Top 10 risk factors           │
└─────────────────────────────────────┘
    ↓
Final Prediction + Explanations
```

**Output**: Churn/Stayed prediction with probability, risk level, and SHAP values

---

### Model 2: Category Prediction Pipeline (If Churned)

```
IF Customer Predicted as "Churned":
    ↓
Input: Same 29 raw features
    ↓
┌─────────────────────────────────────┐
│ 1. Missing Value Imputation        │
│    - Same SimpleImputer             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Text Normalization               │
│    - Lowercase + strip              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Numerical Conversion             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Ordinal Encoding                 │
│    - Uses c_models encoder          │
│    - (contract, offer)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. One-Hot Encoding                 │
│    - Uses c_models encoder          │
│    - Same 16 features               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. Feature Alignment                │
│    - Match category model features  │
│    - No scaling (model trained on   │
│      unscaled data)                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 7. Category Classification          │
│    - Random Forest Classifier       │
│    - 5 classes: Competitor,         │
│      Dissatisfaction, Price,        │
│      Attitude, Other                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 8. Category-Specific                │
│    Recommendations                  │
│    - Tailored to root cause         │
└─────────────────────────────────────┘
    ↓
Category Label + Targeted Actions
```

**Output**: Churn category (why they're leaving) with specific recommendations

---

### Key Differences Between Pipelines

| Aspect | Churn Model (p_models) | Category Model (c_models) |
|--------|------------------------|---------------------------|
| **Purpose** | Predict IF customer will churn | Predict WHY customer churned |
| **When Used** | Always (for all customers) | Only if churned prediction |
| **Output Classes** | 3 (Churned, Stayed, Joined) | 5 (Competitor, Dissatisfaction, Price, Attitude, Other) |
| **Scaling** | Yes (StandardScaler) | No (trained on raw features) |
| **Encoders** | From p_models/ | From c_models/ |
| **Training Data** | All customers | Only churned customers |
| **Features** | 56 (after encoding) | 56 (after encoding) |

---

### Dynamic Threshold Calculation

Unlike traditional models that use a fixed 0.5 threshold, ChurnAI calculates a personalized threshold for each customer:

```python
Formula: threshold = retention_cost / (retention_cost + CLV)

Thresholds by CLV Tier:
├─ CLV ≥ $2000  → threshold ≥ 0.25 (Very Aggressive)
│   Example: $2,500 CLV → threshold = 0.25
│   Logic: Catch high-value customers early
│
├─ CLV ≥ $1000  → threshold ≥ 0.30 (Aggressive)
│   Example: $1,200 CLV → threshold = 0.30
│   Logic: Proactive retention for valuable customers
│
├─ CLV ≥ $500   → threshold ≥ 0.35 (Moderate)
│   Example: $650 CLV → threshold = 0.35
│   Logic: Balanced approach
│
├─ CLV ≥ $200   → threshold ≥ 0.40 (Conservative)
│   Example: $350 CLV → threshold = 0.40
│   Logic: Higher confidence needed
│
└─ CLV < $200   → threshold ≥ 0.45 (Very Conservative)
    Example: $150 CLV → threshold = 0.45
    Logic: Only intervene with strong signal
```

**Business Logic**: High-value customers get lower thresholds (catch them early), while low-value customers need higher confidence before expensive intervention.

---

### SHAP Explainability

SHAP (SHapley Additive exPlanations) provides transparent insights into model predictions:

```python
For each churned prediction:
1. Initialize TreeExplainer with trained model
2. Calculate SHAP values for all 56 features
3. Extract values for "Churned" class
4. Sort by absolute impact
5. Select top 10 positive contributors (pushing towards churn)
6. Convert to percentage: (shap_value × 100)
7. Format feature names for readability
8. Display with animated progress bars
```

**Example SHAP Output:**
- `contract` contributes +7.55% towards churn
- `age` contributes +4.69% towards churn  
- `number_of_referrals` contributes +4.34% towards churn

This tells us: "The customer's month-to-month contract is the single biggest factor pushing them towards churn."

---

### Revenue Impact Model

```python
CLV Calculation:
├─ Advanced Method (Used):
│   avg_monthly = (total_revenue / tenure) × 0.6 + monthly_charge × 0.4
│   remaining_months = max(0, avg_lifespan - current_tenure)
│   CLV = avg_monthly × remaining_months
│
└─ Simple Method (Fallback):
    CLV = monthly_charge × (avg_lifespan - tenure)

Revenue at Risk:
    revenue_at_risk = CLV × churn_probability

ROI Analysis (3 retention tiers):
├─ Basic ($25):   Small discount, quick win
├─ Standard ($50): Moderate offer, good ROI
└─ Premium ($100): Aggressive retention, high stakes

For each tier:
1. expected_loss_without = churn_prob × CLV
2. expected_loss_with = (churn_prob × 0.5) × CLV
3. revenue_saved = expected_loss_without - expected_loss_with
4. net_benefit = revenue_saved - retention_cost
5. roi_percentage = (net_benefit / retention_cost) × 100

Recommendation: Offer with highest positive ROI
```

**Example:**
- CLV: $1,500
- Churn Probability: 70%
- Revenue at Risk: $1,050

| Offer | Cost | Revenue Saved | Net Benefit | ROI |
|-------|------|---------------|-------------|-----|
| Basic | $25 | $525 | $500 | 2000% ✅ |
| Standard | $50 | $525 | $475 | 950% ✅ |
| Premium | $100 | $525 | $425 | 425% ✅ |

**Recommended**: Standard (best balance of cost and effectiveness)

---

## 💼 Business Value

### For Customer Retention Teams
- **Prioritize interventions** based on revenue impact (P1-P4)
- **Personalized outreach** with category-specific strategies
- **ROI-driven campaigns** - only invest when profitable
- **Proactive retention** - catch high-value churners early with dynamic thresholds
- **Explainable decisions** - SHAP shows exactly what's wrong

### For Executives
- **Revenue protection** - quantify exact dollars at risk
- **Resource optimization** - focus budget on high-value customers
- **Cost-benefit analysis** - justify every retention dollar spent
- **Strategic insights** - understand root causes (Competitor 30%, Price 25%, etc.)
- **KPI tracking** - measure actual prevented churn revenue

### For Data Scientists
- **Explainable AI** - SHAP values provide full transparency
- **Business-aligned ML** - thresholds based on actual business costs
- **Comprehensive pipeline** - imputation, transformation, encoding, scaling
- **Dual model architecture** - modular design for prediction + diagnosis
- **Reproducible** - documented preprocessing and model configs

---

## 📊 Model Performance

### Churn Prediction Model (p_models)
- **Algorithm**: Random Forest Classifier
- **Input Features**: 29 raw features
- **Processed Features**: 56 (after encoding)
- **Output Classes**: 3 (Churned, Stayed, Joined)
- **Training Data**: All customer records
- **Preprocessing**: Imputation → Normalization → Encoding → Scaling
- **Evaluation**: Dynamic threshold optimization per customer segment

### Category Prediction Model (c_models)
- **Algorithm**: Random Forest Classifier
- **Input Features**: 29 raw features
- **Processed Features**: 56 (after encoding, no scaling)
- **Output Classes**: 5 (Competitor, Dissatisfaction, Price, Attitude, Other)
- **Training Data**: Only churned customers with labeled reasons
- **Preprocessing**: Imputation → Normalization → Encoding (no scaling)
- **Evaluation**: Multi-class F1-score, per-category precision/recall

### Key Advantages
- ✅ **Handles missing data** via SimpleImputer
- ✅ **Robust to outliers** via Yeo-Johnson transformation
- ✅ **Business-aware predictions** via dynamic thresholds
- ✅ **Explainable** via SHAP feature attributions
- ✅ **Actionable** via category classification + ROI recommendations

---

## 🔑 Key Concepts

### 1. Dynamic Thresholds
**Problem**: Traditional ML uses fixed 0.5 threshold for all customers.
```python
if probability >= 0.5: predict "Churned"  # Same for everyone
```

**ChurnAI Solution**: Customer-specific thresholds based on business value.
```python
threshold = f(customer_CLV, retention_cost)
if probability >= threshold: predict "Churned"  # Personalized per customer
```

**Impact**: High-value customers (CLV $2000+) get threshold 0.25, while low-value customers (CLV $150) get threshold 0.45. This maximizes revenue protection.

---

### 2. SHAP Explainability
**Question**: "Why is this customer predicted to churn?"

**Answer**: SHAP values quantify each feature's contribution.

**Example**:
```
Customer A (72% churn probability):
- Month-to-Month Contract: +7.55% (biggest driver)
- Age (68): +4.69%
- Zero Referrals: +4.34%
- High Monthly Charge: +2.37%
```

**Insight**: "This customer is churning primarily because of their month-to-month contract. Offering a contract upgrade could be the most effective retention strategy."

---

### 3. Dual Model System
**Why Two Models?**
1. **Model 1 (Churn)**: Predicts IF they'll leave → Guides WHO to contact
2. **Model 2 (Category)**: Predicts WHY they're leaving → Guides WHAT to say

**Example Workflow**:
```
Customer X → Churn Model → 75% Churned
                ↓
         [CLV: $2,100]
                ↓
    Dynamic Threshold: 0.25
                ↓
     Prediction: CHURNED ✓
                ↓
         Category Model
                ↓
    Category: "Competitor"
                ↓
   Recommendation: "Counter-offer with 15% discount + highlight unique features"
```

---

### 4. ROI-Based Retention
**Principle**: Not all retention efforts are profitable.

**ChurnAI Calculation**:
```
Customer CLV: $800
Churn Probability: 60%
Revenue at Risk: $480

Standard Offer ($50):
- Expected revenue saved: $240 (assuming 50% retention success)
- Net benefit: $240 - $50 = $190
- ROI: 380% ✅ RECOMMEND

Premium Offer ($100):
- Expected revenue saved: $240
- Net benefit: $240 - $100 = $140
- ROI: 140% ✅ RECOMMEND

Conclusion: Offer Standard (better ROI)
```

**Result**: Only spend retention budget when expected return is positive.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Write clean, documented code
- Add tests for new features
- Update README if needed
- Follow existing code style

---

## 📝 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 ChurnAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 📞 Contact & Support

- **Developer**: Ahem Draza
- **Email**: ahemdraza810@gmail.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Repository**: [churn-prediction-ml-webapp](https://github.com/yourusername/churn-prediction-ml-webapp)
- **Issues**: [GitHub Issues](https://github.com/yourusername/churn-prediction-ml-webapp/issues)

---

## 🗺 Roadmap

### Version 1.0 (Current) ✅
- ✅ Dual model architecture (churn + category)
- ✅ Dynamic threshold optimization
- ✅ SHAP explainability
- ✅ Revenue impact analysis with ROI
- ✅ Complete preprocessing pipeline with imputation & outlier handling
- ✅ Modern responsive UI

### Version 1.1 (Planned)
- [ ] Batch prediction API endpoint
- [ ] Export detailed reports to PDF
- [ ] Historical tracking dashboard
- [ ] A/B testing framework for retention strategies
- [ ] Customer segmentation clustering

### Version 2.0 (Future)
- [ ] Deep learning models (LSTM for sequential behavior)
- [ ] Real-time prediction streaming
- [ ] CRM system integrations (Salesforce, HubSpot)
- [ ] Advanced visualization dashboards (Plotly/Dash)
- [ ] Mobile application (iOS/Android)

---

## 📚 Additional Resources

### Research Papers
- [SHAP: SHapley Additive exPlanations](https://arxiv.org/abs/1705.07874)
- [Cost-Sensitive Learning](https://link.springer.com/article/10.1023/A:1007614023302)
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
- [Customer Churn Prediction: A Survey](https://ieeexplore.ieee.org/document/8360437)

### Tutorials
- [SHAP for ML Explainability](https://shap.readthedocs.io/)
- [Flask Web Development](https://flask.palletsprojects.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

<div align="center">

**Made with ❤️ and ☕ by Ahem Draza**

⭐ Star this repo if you find it useful!

[Back to Top](#-churnai---intelligent-customer-churn-prediction-system)

</div>
