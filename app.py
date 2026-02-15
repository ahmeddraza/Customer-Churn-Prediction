from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import shap
from threshold_optimizer import ThresholdOptimizer
from revenue_model import RevenueImpactModel

app = Flask(__name__)

# Load churn prediction models (p_models)
print("Loading churn prediction models...")
le = pickle.load(open('./p_models/label_encoder.pkl', 'rb'))
sc = pickle.load(open('./p_models/standard_scaler.pkl', 'rb'))
ohe = pickle.load(open('./p_models/onehot_encoder.pkl', 'rb'))
oe = pickle.load(open('./p_models/ordinal_encoder.pkl', 'rb'))
rf = pickle.load(open('./p_models/churn_model.pkl', 'rb'))
feature_names = pickle.load(open('./p_models/feature_names.pkl', 'rb'))

# Load churn category models (c_models)
try:
    rf_category = pickle.load(open('./c_models/category_model.pkl', 'rb'))
    le_category = pickle.load(open('./c_models/label_encoder.pkl', 'rb'))
    ohe_category = pickle.load(open('./c_models/onehot_encoder.pkl', 'rb'))
    oe_category = pickle.load(open('./c_models/ordinal_encoder.pkl', 'rb'))
    HAS_CATEGORY_MODEL = True
    print("✅ Category model loaded")
except Exception as e:
    HAS_CATEGORY_MODEL = False
    print(f"⚠️  Category model not found: {e}")

print("=" * 60)
print("MODELS LOADED SUCCESSFULLY")
print(f"Churn prediction: Enabled")
print(f"Category prediction: {'Enabled' if HAS_CATEGORY_MODEL else 'Disabled'}")
print("=" * 60)

# Initialize business modules
threshold_optimizer = ThresholdOptimizer(
    cost_fp=10,   # Cost of offering retention to a loyal customer
    cost_fn=200   # Cost of losing a customer (base cost, will be adjusted per customer)
)

revenue_model = RevenueImpactModel(
    avg_customer_lifespan_months=24,  # Average customer stays 2 years
    discount_rate=0.1
)

print("✅ Threshold Optimizer initialized")
print("✅ Revenue Impact Model initialized")
print("✅ Using DYNAMIC thresholds based on customer CLV")
print("=" * 60)

def format_feature_name(feature_name):
    """
    Convert technical feature names to human-readable format
    
    Examples:
        'contract' -> 'Contract Type'
        'payment_method_credit card' -> 'Payment: Credit Card'
        'online_security_no' -> 'No Online Security'
        'internet_type_fiber optic' -> 'Internet: Fiber Optic'
    """
    # Handle one-hot encoded features (with values appended)
    if '_' in feature_name:
        parts = feature_name.rsplit('_', 1)
        
        # Check if last part is a value (yes/no or specific value)
        if len(parts) == 2:
            base_feature = parts[0]
            value = parts[1]
            
            # Common yes/no patterns
            if value in ['yes', 'no']:
                base_formatted = base_feature.replace('_', ' ').title()
                if value == 'no':
                    return f"No {base_formatted}"
                else:
                    return f"Has {base_formatted}"
            
            # Payment method, internet type, etc.
            elif base_feature in ['payment_method', 'internet_type', 'contract']:
                base_formatted = base_feature.replace('_', ' ').title()
                value_formatted = value.replace('_', ' ').title()
                return f"{value_formatted}"
            
            # Other encoded features
            else:
                return feature_name.replace('_', ' ').title()
    
    # Simple features without encoding
    return feature_name.replace('_', ' ').title()


def preprocess_for_churn(data_dict):
    """Preprocess data for churn prediction (with scaling)"""
    input_df = pd.DataFrame([data_dict])
    
    # Text normalization
    categorical_cols = ['gender', 'married', 'offer', 'phone_service', 
                       'multiple_lines', 'internet_service', 'internet_type', 
                       'online_security', 'online_backup', 'device_protection_plan', 
                       'premium_tech_support', 'streaming_tv', 'streaming_movies', 
                       'streaming_music', 'unlimited_data', 'contract', 
                       'paperless_billing', 'payment_method']
    
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).str.strip().str.lower()
    
    # Numerical conversion
    numerical_cols = ['age', 'number_of_dependents', 'number_of_referrals', 
                     'tenure_in_months', 'avg_monthly_long_distance_charges',
                     'avg_monthly_gb_download', 'monthly_charge',
                     'total_refunds', 'total_extra_data_charges', 
                     'total_long_distance_charges', 'total_revenue']
    
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Ordinal encoding
    ordinal_cols = ['contract', 'offer']
    input_df[ordinal_cols] = oe.transform(input_df[ordinal_cols])
    
    # OneHot encoding
    onehot_cols = ['gender', 'married', 'phone_service', 'multiple_lines',
                  'internet_service', 'internet_type', 'online_security', 'online_backup',
                  'device_protection_plan', 'premium_tech_support',
                  'streaming_tv', 'streaming_movies', 'streaming_music',
                  'unlimited_data', 'paperless_billing', 'payment_method']
    
    input_ohe = ohe.transform(input_df[onehot_cols])
    ohe_feature_names = ohe.get_feature_names_out(onehot_cols)
    input_ohe_df = pd.DataFrame(input_ohe, columns=ohe_feature_names, index=input_df.index)
    input_df = pd.concat([input_df.drop(columns=onehot_cols), input_ohe_df], axis=1)
    
    # Filter and reindex
    columns_to_keep = [col for col in input_df.columns if col in feature_names]
    input_df = input_df[columns_to_keep]
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Standard scaling
    numerical_features_to_scale = [col for col in feature_names if col in numerical_cols]
    numerical_data = input_df[numerical_features_to_scale].values
    scaled_numerical_data = sc.transform(numerical_data)
    
    for i, col in enumerate(numerical_features_to_scale):
        input_df[col] = scaled_numerical_data[:, i]
    
    return input_df

def preprocess_for_category(data_dict):
    """Preprocess data for category prediction (without scaling)"""
    input_df = pd.DataFrame([data_dict])
    
    # Text normalization
    categorical_cols = ['gender', 'married', 'offer', 'phone_service', 
                       'multiple_lines', 'internet_service', 'internet_type', 
                       'online_security', 'online_backup', 'device_protection_plan', 
                       'premium_tech_support', 'streaming_tv', 'streaming_movies', 
                       'streaming_music', 'unlimited_data', 'contract', 
                       'paperless_billing', 'payment_method']
    
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).str.strip().str.lower()
    
    # Numerical conversion
    numerical_cols = ['age', 'number_of_dependents', 'number_of_referrals', 
                     'tenure_in_months', 'avg_monthly_long_distance_charges',
                     'avg_monthly_gb_download', 'monthly_charge',
                     'total_refunds', 'total_extra_data_charges', 
                     'total_long_distance_charges', 'total_revenue']
    
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Ordinal encoding (using category model's encoder)
    ordinal_cols = ['contract', 'offer']
    input_df[ordinal_cols] = oe_category.transform(input_df[ordinal_cols])
    
    # OneHot encoding (using category model's encoder)
    onehot_cols = ['gender', 'married', 'phone_service', 'multiple_lines',
                  'internet_service', 'internet_type', 'online_security', 'online_backup',
                  'device_protection_plan', 'premium_tech_support',
                  'streaming_tv', 'streaming_movies', 'streaming_music',
                  'unlimited_data', 'paperless_billing', 'payment_method']
    
    input_ohe = ohe_category.transform(input_df[onehot_cols])
    ohe_feature_names = ohe_category.get_feature_names_out(onehot_cols)
    input_ohe_df = pd.DataFrame(input_ohe, columns=ohe_feature_names, index=input_df.index)
    input_df = pd.concat([input_df.drop(columns=onehot_cols), input_ohe_df], axis=1)
    
    return input_df

def calculate_dynamic_threshold(clv, retention_cost=50):
    """
    Calculate optimal threshold for this specific customer based on their CLV
    
    Logic:
    - High CLV customers: Lower threshold (catch them early)
    - Low CLV customers: Higher threshold (only act on high confidence)
    
    Formula: threshold = FP_cost / (FP_cost + FN_cost)
    where FN_cost = CLV (revenue we'd lose)
    
    Args:
        clv: Customer Lifetime Value
        retention_cost: Cost of retention campaign (FP cost)
    
    Returns:
        Optimal threshold for this customer
    """
    # Calculate threshold using cost-sensitive formula
    # threshold = FP_cost / (FP_cost + FN_cost)
    fn_cost = clv  # Cost of missing this churner = their CLV
    optimal_threshold = retention_cost / (retention_cost + fn_cost)
    
    # Apply reasonable bounds based on customer value tiers
    # Don't let it go too extreme - we still need reasonable predictions
    if clv >= 2000:  # Very high value
        optimal_threshold = max(optimal_threshold, 0.25)  # At least 25%
    elif clv >= 1000:  # High value
        optimal_threshold = max(optimal_threshold, 0.30)  # At least 30%
    elif clv >= 500:  # Medium value
        optimal_threshold = max(optimal_threshold, 0.35)  # At least 35%
    elif clv >= 200:  # Standard value
        optimal_threshold = max(optimal_threshold, 0.40)  # At least 40%
    else:  # Low value
        optimal_threshold = max(optimal_threshold, 0.45)  # At least 45%
    
    # Cap at reasonable maximum
    optimal_threshold = min(optimal_threshold, 0.65)
    
    return round(optimal_threshold, 3)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Step 1: Predict churn status
        input_df_churn = preprocess_for_churn(data)
        churn_proba = rf.predict_proba(input_df_churn)[0]
        
        # Get churn probability (probability of "Churned" class)
        churned_class_index = list(le.classes_).index("Churned")
        churn_probability = churn_proba[churned_class_index]
        
        # Calculate revenue impact FIRST (to determine threshold)
        customer_data = {
            'monthly_charge': float(data.get('monthly_charge', 0)),
            'tenure_in_months': int(data.get('tenure_in_months', 0)),
            'total_revenue': float(data.get('total_revenue', 0))
        }
        
        # Calculate CLV first
        clv = revenue_model.calculate_clv_advanced(
            customer_data['monthly_charge'],
            customer_data['tenure_in_months'],
            customer_data['total_revenue']
        )
        
        # Calculate DYNAMIC threshold based on this customer's value
        retention_cost = 50  # Base retention campaign cost
        dynamic_threshold = calculate_dynamic_threshold(clv, retention_cost)
        
        # Update optimizer with customer-specific threshold
        threshold_optimizer.optimal_threshold = dynamic_threshold
        
        print(f"💰 Customer CLV: ${clv:,.2f}")
        print(f"🎯 Dynamic Threshold for this customer: {dynamic_threshold:.3f}")
        
        # Use threshold optimizer for business-aware prediction
        threshold_result = threshold_optimizer.predict_single(churn_probability)
        prediction_label = "Churned" if threshold_result['prediction'] == 1 else "Stayed"
        
        print(f"✅ Churn Probability: {churn_probability:.2%}")
        print(f"✅ Prediction: {prediction_label} (Risk: {threshold_result['risk_level']})")
        
        # Calculate full revenue impact
        revenue_impact = revenue_model.get_customer_revenue_impact(
            customer_data, 
            churn_probability
        )
        
        print(f"💰 CLV: ${revenue_impact['customer_lifetime_value']:,.2f}")
        print(f"💰 Revenue at Risk: ${revenue_impact['revenue_at_risk']:,.2f}")
        
        # Initialize response variables
        category_label = None
        top_features = None
        insights = []
        recommendations = []
        
        # Generate insights for ALL predictions (not just churned)
        if data.get('contract') == 'Month-to-Month':
            insights.append("Month-to-Month contract - no commitment")
        if int(data.get('tenure_in_months', 0)) < 6:
            insights.append(f"Very short tenure ({data.get('tenure_in_months')} months)")
        if float(data.get('total_refunds', 0)) > 0:
            insights.append(f"Has refunds (${data.get('total_refunds')}) - dissatisfaction indicator")
        if float(data.get('total_extra_data_charges', 0)) > 0:
            insights.append(f"Extra data charges (${data.get('total_extra_data_charges')}) - unexpected costs")
        if int(data.get('number_of_referrals', 0)) == 0:
            insights.append("Zero referrals - not engaged")
        if float(data.get('monthly_charge', 0)) > 80:
            insights.append(f"High monthly charge (${data.get('monthly_charge')})")
        
        # Generate basic recommendations for CHURNED customers
        if prediction_label == "Churned":
            # Add revenue-based recommendations
            if revenue_impact['recommended_offer'] != 'Monitor Only':
                offer = revenue_impact['recommended_offer']
                roi_data = revenue_impact['roi_analysis'][offer]
                recommendations.append(f"💰 Offer {offer.title()} retention package (Expected ROI: {roi_data['roi_percentage']:.0f}%)")
                recommendations.append(f"   Investment: ${roi_data['retention_cost']:.2f} | Potential Benefit: ${roi_data['net_benefit']:.2f}")
            
            # Add priority-based recommendation
            recommendations.append(f"🚨 {revenue_impact['priority']} - {threshold_result['recommendation']}")
            
            # Add general retention strategies
            recommendations.append("📞 Immediate outreach within 24 hours")
            recommendations.append("🎁 Personalized retention offer based on customer profile")
            recommendations.append("📊 Schedule account review meeting")
            recommendations.append("💡 Highlight unused services or features")
        
        # Step 2: Calculate SHAP values for ALL churned customers
        if prediction_label == "Churned":
            try:
                # Compute SHAP values for churn model
                print("🔍 Computing SHAP values...")
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(input_df_churn)
                
                # Get SHAP values for "Churned" class
                shap_values_churned = shap_values[churned_class_index][0]
                
                # Get top features
                feature_impact = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values_churned
                })
                feature_impact = feature_impact.sort_values('shap_value', ascending=False)
                top_hurting = feature_impact[feature_impact['shap_value'] > 0].head(10)
                
                # Format for template (convert to percentage and round)
                top_features = [
                    {
                        'name': format_feature_name(row['feature']),  # Use formatted name
                        'value': round(row['shap_value'] * 100, 2)
                    }
                    for idx, row in top_hurting.iterrows()
                ]
                
                print(f"✅ SHAP analysis complete - {len(top_features)} risk factors identified")
                
            except Exception as e:
                print(f"⚠️  Error calculating SHAP values: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: If churned AND category model exists, predict category
        if prediction_label == "Churned" and HAS_CATEGORY_MODEL:
            try:
                # Preprocess for category prediction
                input_df_category = preprocess_for_category(data)
                
                # Predict category
                category_prediction = rf_category.predict(input_df_category)
                category_label = le_category.inverse_transform(category_prediction)[0]
                
                print(f"✅ Category: {category_label}")
                
                # Add category-specific recommendations
                if 'competitor' in category_label.lower():
                    recommendations.insert(0, f"🏆 Category: {category_label} - Competitive threat detected")
                    recommendations.append("🔍 Conduct competitive analysis - identify what competitor is offering")
                    recommendations.append("💳 Counter-offer: Match or beat competitor pricing")
                    recommendations.append("⭐ Highlight unique value propositions and differentiators")
                elif 'dissatisfaction' in category_label.lower():
                    recommendations.insert(0, f"😞 Category: {category_label} - Service quality issues")
                    recommendations.append("🆘 Immediate customer service escalation")
                    recommendations.append("🔧 Address specific pain points and service issues")
                    recommendations.append("🎁 Offer premium service upgrade at no cost for 3 months")
                elif 'price' in category_label.lower():
                    recommendations.insert(0, f"💸 Category: {category_label} - Price sensitivity")
                    recommendations.append("💰 Review pricing tier - consider loyalty discount (10-15%)")
                    recommendations.append("📦 Bundle services for better perceived value")
                    recommendations.append("📈 Show cost-benefit analysis vs competitors")
                elif 'attitude' in category_label.lower():
                    recommendations.insert(0, f"😠 Category: {category_label} - Service attitude concerns")
                    recommendations.append("🙏 Formal apology from management")
                    recommendations.append("👤 Assign dedicated account manager")
                    recommendations.append("🎓 Internal customer service training review")
                else:
                    recommendations.insert(0, f"❓ Category: {category_label}")
                    recommendations.append("📋 Conduct detailed exit interview")
                    recommendations.append("🔬 Deep-dive analysis of customer journey")
                
            except Exception as e:
                print(f"⚠️  Error in category/SHAP: {e}")
                import traceback
                traceback.print_exc()
        
        return render_template('index.html', 
                             prediction=prediction_label,
                             churn_probability=round(churn_probability * 100, 1),
                             risk_level=threshold_result['risk_level'],
                             risk_color=threshold_result['color'],
                             threshold_used=round(threshold_result['threshold_used'], 3),
                             clv=revenue_impact['customer_lifetime_value'],
                             revenue_at_risk=revenue_impact['revenue_at_risk'],
                             revenue_tier=revenue_impact['revenue_tier'],
                             priority=revenue_impact['priority'],
                             recommended_offer=revenue_impact['recommended_offer'],
                             category=category_label,
                             top_features=top_features,
                             insights=insights,
                             recommendations=recommendations)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("❌ ERROR:", error_msg)
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)