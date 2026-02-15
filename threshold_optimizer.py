"""
Threshold Optimizer - Business-Aware ML for Churn Prediction

This module optimizes the prediction threshold based on business costs
instead of using the default 0.5 threshold.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


class ThresholdOptimizer:
    """
    Optimizes prediction threshold based on business costs
    """
    
    def __init__(self, cost_fp=10, cost_fn=200):
        """
        Initialize with business costs
        
        Args:
            cost_fp: Cost of False Positive (wrongly targeting loyal customer)
            cost_fn: Cost of False Negative (missing a churner)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.optimal_threshold = 0.5  # default
        self.threshold_analysis = None
        
    def calculate_cost(self, y_true, y_pred):
        """
        Calculate total business cost based on confusion matrix
        
        Args:
            y_true: Actual labels (0=not churned, 1=churned)
            y_pred: Predicted labels
            
        Returns:
            Total cost
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Cost calculation
        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)
        
        return total_cost, fp, fn, tn, tp
    
    def find_optimal_threshold(self, y_true, y_proba, thresholds=None):
        """
        Find optimal threshold by minimizing business cost
        
        Args:
            y_true: Actual labels
            y_proba: Predicted probabilities for churn class
            thresholds: Array of thresholds to test (default: 0.1 to 0.9)
            
        Returns:
            Optimal threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.01)
        
        results = []
        
        for threshold in thresholds:
            # Make predictions at this threshold
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate cost
            total_cost, fp, fn, tn, tp = self.calculate_cost(y_true, y_pred)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'total_cost': total_cost,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'tp': tp,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Convert to DataFrame
        self.threshold_analysis = pd.DataFrame(results)
        
        # Find threshold with minimum cost
        min_cost_idx = self.threshold_analysis['total_cost'].idxmin()
        self.optimal_threshold = self.threshold_analysis.loc[min_cost_idx, 'threshold']
        
        return self.optimal_threshold
    
    def predict_with_optimal_threshold(self, y_proba):
        """
        Make predictions using the optimal threshold
        
        Args:
            y_proba: Predicted probabilities for churn class
            
        Returns:
            Binary predictions (0 or 1)
        """
        return (y_proba >= self.optimal_threshold).astype(int)
    
    def get_threshold_summary(self):
        """
        Get summary of optimal threshold analysis
        
        Returns:
            Dictionary with key metrics
        """
        if self.threshold_analysis is None:
            return None
        
        min_cost_idx = self.threshold_analysis['total_cost'].idxmin()
        optimal_row = self.threshold_analysis.loc[min_cost_idx]
        
        return {
            'optimal_threshold': self.optimal_threshold,
            'min_total_cost': optimal_row['total_cost'],
            'false_positives': int(optimal_row['fp']),
            'false_negatives': int(optimal_row['fn']),
            'precision': optimal_row['precision'],
            'recall': optimal_row['recall'],
            'f1_score': optimal_row['f1'],
            'cost_fp': self.cost_fp,
            'cost_fn': self.cost_fn
        }
    
    def predict_single(self, churn_probability):
        """
        Predict for a single customer using optimal threshold
        
        Args:
            churn_probability: Probability of churn (0-1)
            
        Returns:
            Prediction (0 or 1), risk level, recommendation
        """
        prediction = 1 if churn_probability >= self.optimal_threshold else 0
        
        # Determine risk level
        if churn_probability >= 0.7:
            risk_level = "Critical"
            color = "red"
        elif churn_probability >= self.optimal_threshold:
            risk_level = "High"
            color = "orange"
        elif churn_probability >= 0.3:
            risk_level = "Medium"
            color = "yellow"
        else:
            risk_level = "Low"
            color = "green"
        
        # Generate recommendation
        if prediction == 1:
            if risk_level == "Critical":
                recommendation = "URGENT: Immediate retention intervention required"
            else:
                recommendation = "Proactive retention campaign recommended"
        else:
            recommendation = "Monitor customer satisfaction"
        
        return {
            'prediction': prediction,
            'probability': churn_probability,
            'risk_level': risk_level,
            'color': color,
            'recommendation': recommendation,
            'threshold_used': self.optimal_threshold
        }


# Example usage function for testing
def optimize_threshold_example(model, X_test, y_test):
    """
    Example function showing how to use ThresholdOptimizer
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
    """
    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn class
    
    # Initialize optimizer with business costs
    optimizer = ThresholdOptimizer(
        cost_fp=10,   # Cost of offering retention to non-churner
        cost_fn=200   # Cost of missing a churner
    )
    
    # Find optimal threshold
    optimal_threshold = optimizer.find_optimal_threshold(y_test, y_proba)
    
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"\nThreshold Summary:")
    summary = optimizer.get_threshold_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return optimizer