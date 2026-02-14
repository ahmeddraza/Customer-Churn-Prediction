from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import shap

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        input_df_churn = preprocess_for_churn(data)
        prediction = rf.predict(input_df_churn)
        prediction_label = le.inverse_transform(prediction)[0]
        
        print(f"✅ Churn Prediction: {prediction_label}")
        
        category_label = None
        top_features = None
        insights = None
        recommendations = None
        
        if prediction_label == "Churned" and HAS_CATEGORY_MODEL:
            try:
                input_df_category = preprocess_for_category(data)
                
                category_prediction = rf_category.predict(input_df_category)
                category_label = le_category.inverse_transform(category_prediction)[0]
                
                print(f"✅ Category: {category_label}")
                
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(input_df_churn)
                
                churned_class_index = list(le.classes_).index("Churned")
                shap_values_churned = shap_values[churned_class_index][0]
                
                feature_impact = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values_churned
                })
                feature_impact = feature_impact.sort_values('shap_value', ascending=False)
                top_hurting = feature_impact[feature_impact['shap_value'] > 0].head(10)
                
                top_features = [
                    {'name': row['feature'], 'value': row['shap_value']}
                    for idx, row in top_hurting.iterrows()
                ]
                
                insights = []
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
                
                recommendations = []
                if 'competitor' in category_label.lower():
                    recommendations.append("Conduct competitive analysis")
                    recommendations.append("Offer retention discount or loyalty program")
                    recommendations.append("Highlight unique value propositions")
                elif 'dissatisfaction' in category_label.lower():
                    recommendations.append("Immediate customer service outreach")
                    recommendations.append("Address specific service quality issues")
                    recommendations.append("Offer complementary premium services")
                elif 'price' in category_label.lower():
                    recommendations.append("Review pricing for this customer segment")
                    recommendations.append("Consider personalized discount")
                    recommendations.append("Bundle services for better value")
                elif 'attitude' in category_label.lower():
                    recommendations.append("Customer service training for staff")
                    recommendations.append("Assign dedicated account manager")
                    recommendations.append("Apologize and offer goodwill gesture")
                else:
                    recommendations.append("Conduct exit interview to identify specific reasons")
                    recommendations.append("Personalized retention approach")
                
                recommendations.append(f"Address {category_label.lower()} concerns specifically")
                recommendations.append("Proactive follow-up within 24 hours")
                
            except Exception as e:
                print(f"⚠️  Error in category/SHAP: {e}")
                import traceback
                traceback.print_exc()
        
        return render_template('index.html', 
                             prediction=prediction_label,
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