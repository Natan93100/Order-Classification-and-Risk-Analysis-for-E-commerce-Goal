import pandas as pd
import matplotlib.pyplot as plt
import math

file_path = 'MH_home_task_dataset_(3).csv'
df = pd.read_csv(file_path)

'''
1.
We are asked to set the decline bar so that 90% of the total orders will be approved.
Translating that number, we want 36,742 orders to be approved given they are sorted by classification rates .
'''

# let's sort the orders by the classification_score column:
sorted_df = df.sort_values(by='classification_score', ascending=False)

# let's get the 36,742nd row's classification_score:
quantile_int = math.ceil(0.9 * df.index[-1])
decline_threshold = sorted_df.iloc[quantile_int]['classification_score']


# let's verify the threshold is unique and if not fix the row number and its value accordingly by using recursion:
def after_search(quantile_int, decline_threshold):
    for i in range(quantile_int + 1, df.index[-1]):
        # non_unique_scores:
        if decline_threshold == sorted_df.iloc[i]['classification_score']:
            pass
        # in case the first condition doesn't apply, meaning the threshold is unique:
        elif i == quantile_int + 1:
            break
        # finding the new size and row number of the df and threshold; calling the function again:
        else:
            new_df_size = df.index[-1] - (i - quantile_int)
            new_quantile_int = round(new_df_size * 0.9)
            after_search(new_quantile_int, decline_threshold)
            return


def before_search(quantile_int, decline_threshold):
    for i in range(quantile_int + 1, df.index[0], -1):
        # non_unique_scores:
        if decline_threshold == sorted_df.iloc[i]['classification_score']:
            pass
        # in case the first condition doesn't apply, meaning the threshold is unique:
        elif i == quantile_int + 1:
            new_quantile_int = i
            break
        # finding the new size and row number of the df and threshold; calling the function again:
        else:
            new_df_size = df.index[-1] - (i - quantile_int)
            new_quantile_int = round(new_df_size * 0.9)
            before_search(new_quantile_int, decline_threshold)
            return


# an integrative function
def unique_search(quantile_int, decline_threshold):
    before_search(quantile_int, decline_threshold)
    after_search(quantile_int, decline_threshold)


unique_search(quantile_int, decline_threshold)

'''
logic explained : if row is not unique we'll find the amount
of rows containing the same classification_score and minimize the df size
resulting in a new row number needed to be calculated.
we'll recur the process till the decline threshold is unique in its area
'''

'''
2.
We are asked to plot the model distribution.
It seems a histogram will work best since we model a 2D table data.
'''

plt.hist(df['classification_score'], bins=30)
plt.title('classification ccore distribution')
plt.xlabel('classification score')
plt.ylabel('orders count')
plt.show()

'''
3.
We are asked the calculate the fee based on the model which 
ensures that the sum of CHBs incurred won't exceed 50% of the revenue.
'''
# let's firstly calculate the CHB sum:

CHBs_sum = df[df['order_status'] == 'chargeback']['price'].sum()
# meaning that the revenue needs to be 2*CHBs_sum
Revenue = 2 * CHBs_sum
# Now let's find the fee, based on the sum of prices:

business_model_fee = Revenue / df['price'].sum()
# basically we looked for the multiplier of the sum of the prices which will result in Revenue.

'''
4.
We are asked to analyze the risk of digital vs tangible products.
we'll check the mean as well the amount of chargebacks for each segment to asses the risk.

'''
# Mean Difference of the score based on digital products
score_mean_dig = df[df['digital_product'] == True]['classification_score'].mean()
score_mean_non_dig = df[df['digital_product'] == False]['classification_score'].mean()
score_mean = df['classification_score'].mean()
difference = score_mean_non_dig - score_mean_dig
difference_to_mean = score_mean - score_mean_dig

# the difference is > 5% making digital products riskier than non ones.
# overall the difference in mean in digital products is about 1.7% which isn't as risky though it deviates.

# number of chargebacks for each category:
digital_chargebacks = df[(df['digital_product'] == True) & (df['order_status'] == 'chargeback')].shape[0]
non_digital_chargebacks = df[(df['digital_product'] == False) & (df['order_status'] == 'chargeback')].shape[0]

# let's calculate the standard error (SE) to assess the risk's
# reliability of each since the dataset is very small:

SE = math.sqrt((1 / (score_mean_dig * digital_chargebacks)) + (1 / (score_mean_non_dig * non_digital_chargebacks)))
# a SE of 0.17 means that although the dataset being checked is small, the results are quite reliable

# let's look at how riskier digital products are compared with non ones.
dig_to_nodig_chargeback_ratio = digital_chargebacks / non_digital_chargebacks

# taking into account our SE of 0.17: it seems that
# digital products are 6.5 to 8 times more likely to  have a chargeback

'''
5.
We are asked to find additional interesting insights.

'''

# 1. customer_account_age_insights:

# let's firstly check the distribution:
plt.hist(df['customer_account_age'], bins=100)
plt.title('customer_account_age distribution')
plt.xlabel('customer_account_age')
plt.ylabel('orders count')

# we notice most of the orders are of customers with an account age between 0 and 1500.

# let's check if the mean prices of these orders is higher than the mean of price per order:
price_mean = df['price'].mean()  # 396$
common_orders_mean = df[df['customer_account_age'] <= 1500]['price'].mean()
# 408$
# it seems the mean price of orders with account_age of
# 0 to 1500 is a lil higher than the regular price mean, making this range more profitable than average.

# let's also see how price is affected by account_age to target the most profitable accounts:
mean_price_by_age = df.groupby('customer_account_age')['price'].mean()
plt.plot(mean_price_by_age.index, mean_price_by_age.values)
plt.xlabel('Customer Account Age')
plt.ylabel('Average Price')
plt.title('Average Price by Customer Account Age')

# it seems that the area of 2000 is the most profitable.
price_mean_discard_2000 = df[(df['customer_account_age'] >= 1950) & (df['customer_account_age'] <= 2050)][
    'price'].mean()
# = 645$

# 2. order_source insights:

# let's look at the distribution:

order_source_dist = df['order_source'].value_counts()
web_ratio = df[df['order_source'] == 'web'].shape[0] / df.shape[0]
mobile_app_ratio = df[df['order_source'] == 'mobile_app'].shape[0] / df.shape[0]

# 98% of orders were placed through the web.

# we expect to get similar ratio when comparing price mean based on order source.
# let's check:

web_ratio_price_mean = df[df['order_source'] == 'web']['price'].sum() / df['price'].sum()
app_ratio_price_mean = df[df['order_source'] == 'mobile_app']['price'].sum() / df['price'].sum()
app_dif_ratio = mobile_app_ratio / app_ratio_price_mean
# we notice that the orders price placed on the app are also approximately 5 times lower.

# let's also look at dist of each by time:

order_source_df = df[df['order_source'].isin(['mobile_app', 'web'])]
order_counts = order_source_df.groupby(['order_date', 'order_source']).size().unstack(fill_value=0)

# Plot
order_counts.plot(kind='area', stacked=True, alpha=0.6, figsize=(12, 6))
plt.xlabel('order_date')
plt.ylabel('number_of_orders')
plt.title('Distribution of Mobile & Web')
plt.legend(title='type of source')
# we can see that app is quite low stable, and web is volatile, especially high around the 15th.

# Results:
# 1
print(decline_threshold)
# 3
print(business_model_fee)
