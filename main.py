import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

file_path = 'MH_home_task_dataset_(3).csv'
df = pd.read_csv(file_path)

### Task 1: Setting the Decline Threshold for 90% Approval Rate
# We sort and use quantile directly for simplicity and efficiency.
decline_threshold = df['classification_score'].quantile(0.1)

# Confirm the threshold results in 90% approval.
approved_orders = df[df['classification_score'] >= decline_threshold]
approval_rate = len(approved_orders) / len(df)
print(f"Decline threshold: {decline_threshold}, Approval rate: {approval_rate:.2%}")

### Task 2: Plotting the Model Score Distribution
plt.hist(df['classification_score'], bins=30, color='skyblue', edgecolor='black')
plt.axvline(decline_threshold, color='red', linestyle='--', label=f'Threshold ({decline_threshold:.2f})')
plt.title('Classification Score Distribution with Decline Threshold')
plt.xlabel('Classification Score')
plt.ylabel('Order Count')
plt.legend()
plt.show()

### Task 3: Calculating the Business Model Fee
# Calculate total CHB amount and required revenue based on a 50% CHB/revenue ratio.
total_chb = df[df['order_status'] == 'chargeback']['price'].sum()
required_revenue = 2 * total_chb  # Revenue needed for a 50% CHB cost-to-revenue ratio
total_price = df['price'].sum()

# Calculate the business model fee as a percentage.
business_model_fee = required_revenue / total_price
print(f"Business model fee required: {business_model_fee:.2%}")

### Task 4: Risk Analysis of Digital vs. Tangible Products
# Calculate the mean classification score and chargeback rate for digital and non-digital products.
digital_mean_score = df[df['digital_product'] == True]['classification_score'].mean()
non_digital_mean_score = df[df['digital_product'] == False]['classification_score'].mean()

# T-test for significant difference in classification scores
t_stat, p_val = stats.ttest_ind(df[df['digital_product'] == True]['classification_score'],
                                 df[df['digital_product'] == False]['classification_score'],
                                 equal_var=False)

# Chargeback counts
digital_chargebacks = df[(df['digital_product'] == True) & (df['order_status'] == 'chargeback')].shape[0]
non_digital_chargebacks = df[(df['digital_product'] == False) & (df['order_status'] == 'chargeback')].shape[0]

# Chargeback rate comparison
digital_cb_rate = digital_chargebacks / df[df['digital_product'] == True].shape[0]
non_digital_cb_rate = non_digital_chargebacks / df[df['digital_product'] == False].shape[0]

print(f"Digital Products - Mean Score: {digital_mean_score:.2f}, Chargeback Rate: {digital_cb_rate:.2%}")
print(f"Non-Digital Products - Mean Score: {non_digital_mean_score:.2f}, Chargeback Rate: {non_digital_cb_rate:.2%}")
print(f"T-test p-value: {p_val:.3f} (indicates {'significant' if p_val < 0.05 else 'no significant'} difference)")

### Task 5: Additional Insights
# Insight 1: Customer Account Age Analysis
plt.hist(df['customer_account_age'], bins=50, color='lightcoral', edgecolor='black')
plt.title('Customer Account Age Distribution')
plt.xlabel('Customer Account Age (days)')
plt.ylabel('Order Count')
plt.show()

# Average price by account age to find profitable age groups
age_price_means = df.groupby(pd.cut(df['customer_account_age'], bins=10))['price'].mean()
age_price_means.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Average Price by Customer Account Age Group')
plt.xlabel('Customer Account Age Group (days)')
plt.ylabel('Average Order Price')
plt.show()

# Insight 2: Order Source Analysis
source_counts = df['order_source'].value_counts(normalize=True) * 100
source_counts.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Order Source Distribution')
plt.xlabel('Order Source')
plt.ylabel('Percentage of Orders')
plt.show()

# Analysis of mean order value per source
source_price_means = df.groupby('order_source')['price'].mean()
source_price_means.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Average Order Value by Order Source')
plt.xlabel('Order Source')
plt.ylabel('Average Order Value')
plt.show()

# Additional Insight: Web vs Mobile Risk Analysis
# Here, we compare chargeback rates between web and mobile orders
web_cb_rate = df[(df['order_source'] == 'web') & (df['order_status'] == 'chargeback')].shape[0] / df[df['order_source'] == 'web'].shape[0]
mobile_cb_rate = df[(df['order_source'] == 'mobile_app') & (df['order_status'] == 'chargeback')].shape[0] / df[df['order_source'] == 'mobile_app'].shape[0]
print(f"Web Chargeback Rate: {web_cb_rate:.2%}, Mobile App Chargeback Rate: {mobile_cb_rate:.2%}")

# Final Print Statements for Clear Results Summary
print(f"\nSummary of Results:\n- Decline Threshold: {decline_threshold}\n- Business Model Fee Required: {business_model_fee:.2%}")
print(f"- Digital Product Risk: Digital products show {'higher' if digital_cb_rate > non_digital_cb_rate else 'lower'} risk compared to tangible products.")
print(f"- Web orders have {'higher' if web_cb_rate > mobile_cb_rate else 'lower'} risk than mobile app orders.")
