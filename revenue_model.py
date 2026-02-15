"""
Revenue Impact Model - Calculate Business Value of Churn Prediction

This module estimates revenue loss and customer lifetime value (CLV)
to provide executive-level business insights.
"""

import numpy as np


class RevenueImpactModel:
    """
    Calculate revenue impact and customer lifetime value
    """
    
    def __init__(self, avg_customer_lifespan_months=24, discount_rate=0.1):
        """
        Initialize revenue model
        
        Args:
            avg_customer_lifespan_months: Average customer lifespan in months
            discount_rate: Monthly discount rate for CLV calculation
        """
        self.avg_customer_lifespan_months = avg_customer_lifespan_months
        self.discount_rate = discount_rate
        
    def calculate_clv_simple(self, monthly_charge, tenure_months=None):
        """
        Calculate Customer Lifetime Value (Simple Method)
        
        CLV = Monthly Charge × Expected Remaining Months
        
        Args:
            monthly_charge: Monthly recurring charge
            tenure_months: Current tenure (optional)
            
        Returns:
            Estimated CLV
        """
        if tenure_months is not None:
            # Estimate remaining months
            remaining_months = max(0, self.avg_customer_lifespan_months - tenure_months)
        else:
            remaining_months = self.avg_customer_lifespan_months
        
        clv = monthly_charge * remaining_months
        return clv
    
    def calculate_clv_advanced(self, monthly_charge, tenure_months, total_revenue):
        """
        Calculate CLV (Advanced Method with historical data)
        
        Args:
            monthly_charge: Current monthly charge
            tenure_months: Current tenure
            total_revenue: Total historical revenue from customer
            
        Returns:
            Estimated CLV
        """
        # Calculate average monthly revenue (might differ from current charge)
        avg_monthly_revenue = total_revenue / tenure_months if tenure_months > 0 else monthly_charge
        
        # Estimate remaining months
        remaining_months = max(0, self.avg_customer_lifespan_months - tenure_months)
        
        # Project future value with weighted average
        projected_monthly = (avg_monthly_revenue * 0.6) + (monthly_charge * 0.4)
        
        clv = projected_monthly * remaining_months
        return clv
    
    def calculate_revenue_at_risk(self, churn_probability, clv):
        """
        Calculate revenue at risk based on churn probability
        
        Revenue at Risk = Churn Probability × CLV
        
        Args:
            churn_probability: Probability of churn (0-1)
            clv: Customer Lifetime Value
            
        Returns:
            Estimated revenue at risk
        """
        return churn_probability * clv
    
    def calculate_retention_roi(self, retention_cost, churn_probability, clv):
        """
        Calculate ROI of retention campaign
        
        Args:
            retention_cost: Cost of retention offer/campaign
            churn_probability: Probability of churn
            clv: Customer Lifetime Value
            
        Returns:
            Dictionary with ROI metrics
        """
        # Expected value without intervention
        expected_loss = churn_probability * clv
        
        # Assume retention reduces churn probability by 50%
        retention_success_rate = 0.5
        reduced_churn_prob = churn_probability * (1 - retention_success_rate)
        
        # Expected value with intervention
        expected_loss_with_retention = reduced_churn_prob * clv
        
        # Revenue saved
        revenue_saved = expected_loss - expected_loss_with_retention
        
        # Net benefit
        net_benefit = revenue_saved - retention_cost
        
        # ROI
        roi = (net_benefit / retention_cost * 100) if retention_cost > 0 else 0
        
        return {
            'retention_cost': retention_cost,
            'expected_loss_without_action': expected_loss,
            'expected_loss_with_retention': expected_loss_with_retention,
            'revenue_saved': revenue_saved,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'recommendation': 'Proceed' if net_benefit > 0 else 'Not Recommended'
        }
    
    def get_customer_revenue_impact(self, customer_data, churn_probability):
        """
        Calculate complete revenue impact for a single customer
        
        Args:
            customer_data: Dictionary with customer info
                - monthly_charge
                - tenure_in_months
                - total_revenue
            churn_probability: Probability of churn (0-1)
            
        Returns:
            Dictionary with revenue impact metrics
        """
        monthly_charge = customer_data.get('monthly_charge', 0)
        tenure_months = customer_data.get('tenure_in_months', 0)
        total_revenue = customer_data.get('total_revenue', 0)
        
        # Calculate CLV
        clv_simple = self.calculate_clv_simple(monthly_charge, tenure_months)
        clv_advanced = self.calculate_clv_advanced(monthly_charge, tenure_months, total_revenue)
        
        # Use advanced CLV
        clv = clv_advanced
        
        # Calculate revenue at risk
        revenue_at_risk = self.calculate_revenue_at_risk(churn_probability, clv)
        
        # Determine revenue tier
        if revenue_at_risk >= 1000:
            revenue_tier = "High Value"
            priority = "P1 - Critical"
        elif revenue_at_risk >= 500:
            revenue_tier = "Medium Value"
            priority = "P2 - High"
        elif revenue_at_risk >= 200:
            revenue_tier = "Standard Value"
            priority = "P3 - Medium"
        else:
            revenue_tier = "Low Value"
            priority = "P4 - Low"
        
        # Calculate retention ROI with different offer costs
        retention_offers = {
            'basic': 25,      # Basic discount/offer
            'standard': 50,   # Standard retention package
            'premium': 100    # Premium retention package
        }
        
        roi_analysis = {}
        for offer_name, offer_cost in retention_offers.items():
            roi_analysis[offer_name] = self.calculate_retention_roi(
                offer_cost, 
                churn_probability, 
                clv
            )
        
        # Recommend best offer
        best_offer = None
        best_roi = -float('inf')
        for offer_name, roi_data in roi_analysis.items():
            if roi_data['net_benefit'] > 0 and roi_data['roi_percentage'] > best_roi:
                best_roi = roi_data['roi_percentage']
                best_offer = offer_name
        
        return {
            'customer_lifetime_value': round(clv, 2),
            'revenue_at_risk': round(revenue_at_risk, 2),
            'revenue_tier': revenue_tier,
            'priority': priority,
            'churn_probability': churn_probability,
            'monthly_charge': monthly_charge,
            'tenure_months': tenure_months,
            'total_historical_revenue': total_revenue,
            'recommended_offer': best_offer if best_offer else 'Monitor Only',
            'roi_analysis': roi_analysis
        }
    
    def format_currency(self, amount):
        """Format amount as currency"""
        return f"${amount:,.2f}"
    
    def get_executive_summary(self, customer_data, churn_probability):
        """
        Generate executive summary for business stakeholders
        
        Args:
            customer_data: Customer information
            churn_probability: Churn probability
            
        Returns:
            Formatted executive summary
        """
        impact = self.get_customer_revenue_impact(customer_data, churn_probability)
        
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║          REVENUE IMPACT ANALYSIS                         ║
╚══════════════════════════════════════════════════════════╝

Customer Value Metrics:
  • Customer Lifetime Value: {self.format_currency(impact['customer_lifetime_value'])}
  • Revenue at Risk: {self.format_currency(impact['revenue_at_risk'])}
  • Revenue Tier: {impact['revenue_tier']}
  • Priority Level: {impact['priority']}

Churn Assessment:
  • Churn Probability: {impact['churn_probability']*100:.1f}%
  • Current Monthly Charge: {self.format_currency(impact['monthly_charge'])}
  • Customer Tenure: {impact['tenure_months']} months
  • Total Historical Revenue: {self.format_currency(impact['total_historical_revenue'])}

Retention Recommendation:
  • Suggested Offer: {impact['recommended_offer'].upper()}
"""
        
        if impact['recommended_offer'] != 'Monitor Only':
            best_offer = impact['roi_analysis'][impact['recommended_offer']]
            summary += f"""  • Expected ROI: {best_offer['roi_percentage']:.1f}%
  • Investment Required: {self.format_currency(best_offer['retention_cost'])}
  • Expected Net Benefit: {self.format_currency(best_offer['net_benefit'])}
"""
        
        return summary


# Example usage
if __name__ == "__main__":
    # Example customer
    customer = {
        'monthly_charge': 85.50,
        'tenure_in_months': 12,
        'total_revenue': 1026.00
    }
    
    churn_prob = 0.75
    
    # Initialize model
    revenue_model = RevenueImpactModel(
        avg_customer_lifespan_months=24,
        discount_rate=0.1
    )
    
    # Get impact analysis
    impact = revenue_model.get_customer_revenue_impact(customer, churn_prob)
    
    print("Revenue Impact Analysis:")
    print(f"CLV: ${impact['customer_lifetime_value']:,.2f}")
    print(f"Revenue at Risk: ${impact['revenue_at_risk']:,.2f}")
    print(f"Priority: {impact['priority']}")
    print(f"Recommended Offer: {impact['recommended_offer']}")
    
    # Get executive summary
    print("\n" + revenue_model.get_executive_summary(customer, churn_prob))