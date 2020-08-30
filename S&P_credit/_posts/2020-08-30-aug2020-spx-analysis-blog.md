# Python exercise: investigating corporate credit ratings and stock performance during the COVID-19 crash



```python
import pandas as pd
import numpy as np

files = ['spx_daily_2.24.20_7.30.20.csv', 'spx_credit_rating_and_mkt_cap.csv']

#import the csv files using pandas, assigning index names, column names, and data types; leave out 'End_Rating'
spx_daily_prices = pd.read_csv(files[0], sep=',', index_col = 'DATES')
spx_credit_marketcap = pd.read_csv(files[1], sep=',', index_col = 'Security', header = 0,
                                  names = ['Security','Start_Rating', 'End_Rating', 'Market_Cap'],
                                   usecols = ['Security', 'Start_Rating', 'Market_Cap'],
                                  dtype = {'Start_Rating': 'str', 'Market_Cap':np.float64})

#display the info() for each df
print('The daily prices df:')
print(spx_daily_prices.info(), '\n')

print('The credit rating and market cap df:')
print(spx_credit_marketcap.info())
```

    The daily prices df:
    <class 'pandas.core.frame.DataFrame'>
    Index: 159 entries, 2/24/2020 to 7/31/2020
    Columns: 505 entries, LYB UN Equity to DISCK UW Equity
    dtypes: float64(505)
    memory usage: 628.5+ KB
    None 
    
    The credit rating and market cap df:
    <class 'pandas.core.frame.DataFrame'>
    Index: 505 entries, LYB UN Equity to DISCK UW Equity
    Data columns (total 2 columns):
    Start_Rating    446 non-null object
    Market_Cap      505 non-null float64
    dtypes: float64(1), object(1)
    memory usage: 11.8+ KB
    None


Right off the bat we can see that the indices are swapped. In the first dataframe, each row represents a date, and each column is a company. In the second dataframe, each row is a company, and the columns hold the credit ratings and market capitalization information. We will have to keep this in mind when we combine the dataframes.

We can also note the presence of around 19 `NaN` values in the `'Start_Rating'` column of the second dataset. This means we will have to address these `NaN` values in some way during our data preparation step.  Options like entering an average are not available with discrete data, so let's just remove these companies.

To get a better idea of how the data actually looks, let's use the `head()` function to view the first 10 rows. Let's also check what kind of unique values we get in the `'Start_Rating'` column of the credit rating and market cap dataframe.

```python
# display the first ten rows of each df
print('The daily prices df:')
display(spx_daily_prices.head(10))

print('The credit rating and market cap df:')
display(spx_credit_marketcap.head(10))

print('Unique values in "Start_Rating" column:')
display(spx_credit_marketcap['Start_Rating'].unique())
```

    The daily prices df:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LYB UN Equity</th>
      <th>AXP UN Equity</th>
      <th>VZ UN Equity</th>
      <th>AVGO UW Equity</th>
      <th>BA UN Equity</th>
      <th>CAT UN Equity</th>
      <th>JPM UN Equity</th>
      <th>CVX UN Equity</th>
      <th>KO UN Equity</th>
      <th>ABBV UN Equity</th>
      <th>...</th>
      <th>ALGN UW Equity</th>
      <th>ILMN UW Equity</th>
      <th>LKQ UW Equity</th>
      <th>NLSN UN Equity</th>
      <th>GRMN UW Equity</th>
      <th>ZTS UN Equity</th>
      <th>EQIX UW Equity</th>
      <th>DLR UN Equity</th>
      <th>LVS UN Equity</th>
      <th>DISCK UW Equity</th>
    </tr>
    <tr>
      <th>DATES</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2/24/2020</td>
      <td>79.65</td>
      <td>128.19</td>
      <td>57.99</td>
      <td>291.60</td>
      <td>317.90</td>
      <td>132.17</td>
      <td>132.16</td>
      <td>104.71</td>
      <td>58.65</td>
      <td>93.14</td>
      <td>...</td>
      <td>240.11</td>
      <td>281.12</td>
      <td>32.330</td>
      <td>21.38</td>
      <td>95.36</td>
      <td>138.39</td>
      <td>644.90</td>
      <td>135.27</td>
      <td>62.21</td>
      <td>27.13</td>
    </tr>
    <tr>
      <td>2/25/2020</td>
      <td>75.41</td>
      <td>120.90</td>
      <td>57.12</td>
      <td>282.68</td>
      <td>304.14</td>
      <td>129.00</td>
      <td>126.26</td>
      <td>100.71</td>
      <td>57.82</td>
      <td>89.18</td>
      <td>...</td>
      <td>229.61</td>
      <td>269.95</td>
      <td>31.320</td>
      <td>20.42</td>
      <td>92.00</td>
      <td>134.82</td>
      <td>631.98</td>
      <td>133.54</td>
      <td>60.16</td>
      <td>26.30</td>
    </tr>
    <tr>
      <td>2/26/2020</td>
      <td>74.39</td>
      <td>118.50</td>
      <td>57.14</td>
      <td>285.88</td>
      <td>305.59</td>
      <td>128.25</td>
      <td>126.64</td>
      <td>98.04</td>
      <td>57.60</td>
      <td>88.41</td>
      <td>...</td>
      <td>227.54</td>
      <td>274.70</td>
      <td>30.540</td>
      <td>20.29</td>
      <td>90.38</td>
      <td>136.15</td>
      <td>629.84</td>
      <td>130.38</td>
      <td>59.16</td>
      <td>25.42</td>
    </tr>
    <tr>
      <td>2/27/2020</td>
      <td>70.90</td>
      <td>112.81</td>
      <td>55.06</td>
      <td>273.95</td>
      <td>287.76</td>
      <td>123.27</td>
      <td>121.37</td>
      <td>94.13</td>
      <td>54.93</td>
      <td>85.42</td>
      <td>...</td>
      <td>223.97</td>
      <td>259.93</td>
      <td>29.720</td>
      <td>18.58</td>
      <td>89.28</td>
      <td>133.95</td>
      <td>597.49</td>
      <td>123.19</td>
      <td>58.43</td>
      <td>23.79</td>
    </tr>
    <tr>
      <td>2/28/2020</td>
      <td>71.46</td>
      <td>109.93</td>
      <td>54.16</td>
      <td>272.62</td>
      <td>275.11</td>
      <td>124.24</td>
      <td>116.11</td>
      <td>93.34</td>
      <td>53.49</td>
      <td>85.71</td>
      <td>...</td>
      <td>218.35</td>
      <td>265.67</td>
      <td>29.580</td>
      <td>18.21</td>
      <td>88.39</td>
      <td>133.23</td>
      <td>572.80</td>
      <td>120.11</td>
      <td>58.31</td>
      <td>25.10</td>
    </tr>
    <tr>
      <td>2/29/2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3/1/2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3/2/2020</td>
      <td>74.52</td>
      <td>113.87</td>
      <td>57.32</td>
      <td>284.97</td>
      <td>289.27</td>
      <td>127.60</td>
      <td>121.52</td>
      <td>96.59</td>
      <td>55.92</td>
      <td>88.69</td>
      <td>...</td>
      <td>230.22</td>
      <td>278.28</td>
      <td>29.900</td>
      <td>18.12</td>
      <td>93.47</td>
      <td>138.91</td>
      <td>618.88</td>
      <td>130.49</td>
      <td>58.26</td>
      <td>25.17</td>
    </tr>
    <tr>
      <td>3/3/2020</td>
      <td>72.24</td>
      <td>108.01</td>
      <td>55.70</td>
      <td>274.25</td>
      <td>280.62</td>
      <td>124.38</td>
      <td>116.96</td>
      <td>94.39</td>
      <td>56.06</td>
      <td>87.57</td>
      <td>...</td>
      <td>224.56</td>
      <td>273.00</td>
      <td>28.730</td>
      <td>17.58</td>
      <td>90.07</td>
      <td>137.24</td>
      <td>615.16</td>
      <td>129.59</td>
      <td>55.90</td>
      <td>25.08</td>
    </tr>
    <tr>
      <td>3/4/2020</td>
      <td>75.23</td>
      <td>115.70</td>
      <td>58.12</td>
      <td>285.45</td>
      <td>283.12</td>
      <td>127.40</td>
      <td>119.85</td>
      <td>98.53</td>
      <td>58.92</td>
      <td>91.75</td>
      <td>...</td>
      <td>234.70</td>
      <td>282.13</td>
      <td>29.645</td>
      <td>17.79</td>
      <td>93.41</td>
      <td>143.62</td>
      <td>645.49</td>
      <td>134.98</td>
      <td>57.25</td>
      <td>25.11</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 505 columns</p>
</div>


    The credit rating and market cap df:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Rating</th>
      <th>Market_Cap</th>
    </tr>
    <tr>
      <th>Security</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LYB UN Equity</td>
      <td>BBB+</td>
      <td>2.162277e+10</td>
    </tr>
    <tr>
      <td>AXP UN Equity</td>
      <td>BBB+</td>
      <td>7.508127e+10</td>
    </tr>
    <tr>
      <td>VZ UN Equity</td>
      <td>BBB+</td>
      <td>2.396347e+11</td>
    </tr>
    <tr>
      <td>AVGO UW Equity</td>
      <td>BBB-</td>
      <td>1.311084e+11</td>
    </tr>
    <tr>
      <td>BA UN Equity</td>
      <td>A-</td>
      <td>9.293648e+10</td>
    </tr>
    <tr>
      <td>CAT UN Equity</td>
      <td>A</td>
      <td>7.138405e+10</td>
    </tr>
    <tr>
      <td>JPM UN Equity</td>
      <td>A-</td>
      <td>2.908938e+11</td>
    </tr>
    <tr>
      <td>CVX UN Equity</td>
      <td>AA</td>
      <td>1.609709e+11</td>
    </tr>
    <tr>
      <td>KO UN Equity</td>
      <td>A+</td>
      <td>2.003822e+11</td>
    </tr>
    <tr>
      <td>ABBV UN Equity</td>
      <td>A-</td>
      <td>1.676239e+11</td>
    </tr>
  </tbody>
</table>
</div>


    Unique values in "Start_Rating" column:



    array(['BBB+', 'BBB-', 'A-', 'A', 'AA', 'A+', 'BB+', 'BBB', 'AA+', 'AAA',
           'AA-', 'BB', nan, 'BB-', 'NR', 'BBu', 'B+', 'B', 'BBBu'],
          dtype=object)


In addition to the point on swapped axes and `NaN` values listed above, there are a few more things to note:
* To take advantage of pandas time series analysis funcitonality, we should convert the index of the daily_prices dataframe to `datetime` format.
* We see that the companies are referenced by their sticker name, e.g. VZ for Verizon, in both datasets, which will make combining the datasets easier. They appear to be sorted in the same manner, too, but we can make sure by alphabetizing them.
* The daily stock prices for the weekends have been included and will need to be removed in the data preparation stage.
* Besides `NaN`, we also see ratings like 'BBu' and 'BBBu' in the `'Start_Rating'` column. This distinction is unnecessary, so we will merge these ratings with 'BB' and 'BBB,' respectively.  

### Cleaning and preparing the data

In this case we will transpose the daily_prices index to be able to match and sort the indices. We must transpose this one because this is the only information the datasets have in common. Seems counterintuitive. Maybe there is a better way.

Remove the 'u' character by using the `replace()` function with regular expression functionality

Great, let's take those steps now with the code below.

```python
import datetime

#convert prices index to datetime format
spx_daily_prices.index = pd.to_datetime(spx_daily_prices.index, format = '%m/%d/%Y')

#change BBu and BBBu to BB and BBB (remove 'u') :
spx_credit_marketcap.loc[:,'Start_Rating'].replace('u$', '', regex=True, inplace=True)

#transpose prices to be able to match index
spx_daily_prices = spx_daily_prices.transpose()
spx_daily_prices.index.name = 'Security'

#sort both dataframes' indices alphabetically to be able to match them
spx_daily_prices.sort_index(inplace=True)
spx_credit_marketcap.sort_index(inplace=True)

#join credit df with prices df into new dataframe
spx_cred_prices = spx_credit_marketcap.join(spx_daily_prices, how = 'left')

#drop rows that are missing the credit rating
spx_cred_prices = spx_cred_prices.dropna(subset = ['Start_Rating'])

#check that only rows with legitimate credit ratings are left
print(spx_cred_prices.Start_Rating.unique())
print(spx_cred_prices.info())

#remove weekend dates
spx_cred_prices.dropna(axis=1, how='all', inplace=True)
```

    ['BBB+' 'BB-' 'BBB-' 'AA+' 'A-' 'A+' 'A' 'BBB' 'AA' 'BB+' 'NR' 'BB' 'AA-'
     'B+' 'B' 'AAA']
    <class 'pandas.core.frame.DataFrame'>
    Index: 446 entries, A UN Equity to ZTS UN Equity
    Columns: 161 entries, Start_Rating to 2020-07-31 00:00:00
    dtypes: float64(160), object(1)
    memory usage: 564.5+ KB
    None


**Bingo**. We have successfully created a new dataframe with NaN weekend prices and credit rating values removed and BBBu and BBu corrected. Data preprocessing has been a success!

### Processing the data

Next, let's think about the task at hand. We want to group the companies by their credit rating, provide a weight for each company based on its market cap, and then normalize the price data to see a percentage based performance over the chosen time frame.

To provide a weight for each company based on its market cap, we need to divide each company's market cap by its credit rating group's total market cap. 

For example: American Express, under the ticker AXP, has a credit rating of BBB+, and a market cap of USD 75,081,270,000, or around 75 billion dollars. Verizon, under the ticker VZ, also has the credit rating BBB+, and a market cap of USD 239,634,700,000, or around 239 billion dollars. Imagining for a moment that these two companies are the only BBB+ entries in our dataset, the BBB+ group's total market cap would be 75 + 239 = USD 314 billion.

If we wanted to track the stock price movement of an index of this two-member BBB+ group, AXP would account for 23.8% (75 / 314) of the index's movement and VZ for 76.2% of the movement. So if the index was at 100 on day 1, and VZ doubled its share price the next day, the index would rise to 172.

An elegant solution to accomplish this arithmetic is to calculate the total market cap for each group and store this in a dictionary called `group_mktcaps`. Then we can used `pandas'` `map()` function (which matches a key to a value in a dictionary) to divide each company's market cap by the total market cap for each company's group. 

```python
#get each group's total market cap and turn it into a dictionary
#groupby Start_Rating, and sum the Market_Cap column
group_mktcaps = spx_cred_prices.groupby(by = ['Start_Rating'])['Market_Cap'].sum().to_dict()

#create the Weight column by dividing each company's market cap by the group's market cap
spx_cred_prices['Weight'] = spx_cred_prices['Market_Cap'].div(spx_cred_prices['Start_Rating'].map(group_mktcaps), axis = 0)

#display the head of the spx_cred_prices dataframe to see the weight column
display(spx_cred_prices.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start_Rating</th>
      <th>Market_Cap</th>
      <th>2020-02-24 00:00:00</th>
      <th>2020-02-25 00:00:00</th>
      <th>2020-02-26 00:00:00</th>
      <th>2020-02-27 00:00:00</th>
      <th>2020-02-28 00:00:00</th>
      <th>2020-03-02 00:00:00</th>
      <th>2020-03-03 00:00:00</th>
      <th>2020-03-04 00:00:00</th>
      <th>...</th>
      <th>2020-07-21 00:00:00</th>
      <th>2020-07-22 00:00:00</th>
      <th>2020-07-23 00:00:00</th>
      <th>2020-07-24 00:00:00</th>
      <th>2020-07-27 00:00:00</th>
      <th>2020-07-28 00:00:00</th>
      <th>2020-07-29 00:00:00</th>
      <th>2020-07-30 00:00:00</th>
      <th>2020-07-31 00:00:00</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Security</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A UN Equity</td>
      <td>BBB+</td>
      <td>3.020460e+10</td>
      <td>80.50</td>
      <td>77.95</td>
      <td>78.11</td>
      <td>77.43</td>
      <td>77.07</td>
      <td>81.97</td>
      <td>80.32</td>
      <td>84.35</td>
      <td>...</td>
      <td>96.02</td>
      <td>96.35</td>
      <td>96.66</td>
      <td>94.81</td>
      <td>97.27</td>
      <td>95.30</td>
      <td>97.10</td>
      <td>95.93</td>
      <td>96.33</td>
      <td>0.010006</td>
    </tr>
    <tr>
      <td>AAL UW Equity</td>
      <td>BB-</td>
      <td>5.919652e+09</td>
      <td>25.45</td>
      <td>23.12</td>
      <td>22.31</td>
      <td>20.60</td>
      <td>19.05</td>
      <td>18.86</td>
      <td>17.85</td>
      <td>18.53</td>
      <td>...</td>
      <td>11.47</td>
      <td>11.36</td>
      <td>11.77</td>
      <td>11.39</td>
      <td>11.39</td>
      <td>11.77</td>
      <td>11.40</td>
      <td>11.18</td>
      <td>11.12</td>
      <td>0.022503</td>
    </tr>
    <tr>
      <td>AAP UN Equity</td>
      <td>BBB-</td>
      <td>1.050753e+10</td>
      <td>139.91</td>
      <td>137.60</td>
      <td>134.54</td>
      <td>134.13</td>
      <td>132.98</td>
      <td>132.99</td>
      <td>130.33</td>
      <td>132.90</td>
      <td>...</td>
      <td>147.38</td>
      <td>149.40</td>
      <td>148.37</td>
      <td>147.27</td>
      <td>146.83</td>
      <td>146.19</td>
      <td>148.21</td>
      <td>152.87</td>
      <td>150.14</td>
      <td>0.010200</td>
    </tr>
    <tr>
      <td>AAPL UW Equity</td>
      <td>AA+</td>
      <td>1.877303e+12</td>
      <td>298.18</td>
      <td>288.08</td>
      <td>292.65</td>
      <td>273.52</td>
      <td>273.36</td>
      <td>298.81</td>
      <td>289.32</td>
      <td>302.74</td>
      <td>...</td>
      <td>388.00</td>
      <td>389.09</td>
      <td>371.38</td>
      <td>370.46</td>
      <td>379.24</td>
      <td>373.01</td>
      <td>380.16</td>
      <td>384.76</td>
      <td>425.04</td>
      <td>0.462429</td>
    </tr>
    <tr>
      <td>ABBV UN Equity</td>
      <td>A-</td>
      <td>1.676239e+11</td>
      <td>93.14</td>
      <td>89.18</td>
      <td>88.41</td>
      <td>85.42</td>
      <td>85.71</td>
      <td>88.69</td>
      <td>87.57</td>
      <td>91.75</td>
      <td>...</td>
      <td>97.40</td>
      <td>97.70</td>
      <td>98.03</td>
      <td>97.11</td>
      <td>97.16</td>
      <td>96.71</td>
      <td>97.01</td>
      <td>96.04</td>
      <td>94.91</td>
      <td>0.042421</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 115 columns</p>
</div>


Great. From the Weight column, which has been added as the last column, we can see for example that Apple (AAPL) makes up 46% of the AA+ group. Because the Weight columns is last, we will need to leave it out of our operations below by using the `:-1` slicing.

Next we want to normalize the daily price data by dividing each row by its price on the first date, February 24, 2020. This will enable us to see the percentage change each day in the analysis.

After that, we will multiply each row by its weight to receive the weighted daily performance of each company. All that remains then is to group the data again by credit rating, sum up each day's performance, and multiply by 100 to receive a normalized and weighty daily performance by credit rating.

```python
#normalize the daily price data by dividing each row by the price on the first date
norm_spx_cred_prices = spx_cred_prices.iloc[:,2:-1].div(spx_cred_prices.iloc[:,2], axis = 0)

#multiply each row by its weight 
# iloc[-1] because weight column is last. 
wtd_norm_spx_cred_prices = norm_spx_cred_prices.mul(spx_cred_prices.iloc[:,-1], axis = 0)

#add the Start_Rating column back in to be able to sort by Start_Rating
full_wtd_norm_spx_cred_prices = wtd_norm_spx_cred_prices.join(spx_cred_prices['Start_Rating'], how='left')

#groupby the Start_Rating, leave out the Start_Rating column, sum it up, and multiply by 100
wtd_spx_bygroup = full_wtd_norm_spx_cred_prices.groupby('Start_Rating')[full_wtd_norm_spx_cred_prices.columns[:-1]].sum().mul(100, axis = 0)
```

Next, let's plot the data to get an initial idea of how each group's performance looks. First, we will transpose the dataframe so that each row is a credit rating group and column is a date, which makes it  use `Pandas`' built-in `plot()` functionality. We will then create the plot with a legend.

```python
#let's plot this
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})

#transpose to make plotting easier (now each Column is a credit rating group)
wtd_spx_bygroupT = wtd_spx_bygroup.transpose()

#plot 
graph1 = wtd_spx_bygroupT.plot().legend(loc='center left',bbox_to_anchor=(1.0, 0.5)).get_figure()
```


![png](aug2020-spx-analysis-blog_files/output_10_0.png)


### Interpretting the results

Great! At first glance we can see that there is indeed some variation. Some credit rating groups recovered their losses already in mid-April, whereas one is still down over 60% in late July.

This graph is a bit hard to interpret, however. There are lots of lines bunched together and the colors seem to overlap. What if we plot just a couple of them?

For example, one approach would be to chart the credit rating groups that reach the highest highs and the lowest lows.

```python
#find the credit rating group that reaches the absolute highest and the one that reaches the absolute lowest price
max_and_min = [wtd_spx_bygroupT.max().idxmax(), wtd_spx_bygroupT.min().idxmin()]

#select the desired columns
wtd_spx_subset = wtd_spx_bygroupT[max_and_min]

#plot the winning couple
graph2 = wtd_spx_subset.plot().legend(loc='center left',bbox_to_anchor=(1.0, 0.5)).get_figure()
```


![png](aug2020-spx-analysis-blog_files/output_12_0.png)


So BB- reaches the highest high, and B+ reaches the lowest low. Interestingly enough, these two credit ratings are adjacent to each other towards the lower end of the pack. This suggests that the relationship between credit rating and stock price performance is not particularly strong. Otherwise we would expect AAA to be the best performer and B to be the lowest. 

Why did BB- chart so high? And why did B+ chart so high? Let's take a look at the constituent companies and their market caps?

```python
print(spx_cred_prices[spx_cred_prices['Start_Rating'] == 'BB-'].index)
print(spx_cred_prices[spx_cred_prices['Start_Rating'] == 'B+'].index)
```

    Index(['AAL UW Equity', 'IRM UN Equity', 'LB UN Equity', 'LYV UN Equity',
           'MGM UN Equity', 'NFLX UW Equity'],
          dtype='object', name='Security')
    Index(['COTY UN Equity'], dtype='object', name='Security')


The beauty products company Coty Inc is the only stock in the B+ group, so the B+ really just tracks Coty's performance during our time period. To say the least, Coty's stock has seen better days (which makes sense, since all that working from home can't be good for cosmetic sales). 

BB- comprises American Airlines, enterprise information management company Iron Mountain Inc, fashion retail group L Brands (owner of Bath and Body Works and Victoria's Secret), Live Nation Entertainment, MGM Resorts International, and Netflix. What gives? Most of these are in tourism and entertainment and seem like they would have been terrible investments. Is there one outlier in BB- that had an especially strong performance? Let's find out. 

```python
#create a bar chart of BB- market cap
BBm_subset = spx_cred_prices[spx_cred_prices['Start_Rating'] == 'BB-']
BBm_subset.plot.bar(y = 'Market_Cap')

#normalize BB- stocks by dividing by share price on first day
norm_BBm_subset = BBm_subset.iloc[:,2:-1].div(BBm_subset.iloc[:,2], axis = 0)

#transpose and plot BB- stock performances
BBm_graph = norm_BBm_subset.T.plot().legend(loc='center left',bbox_to_anchor=(1.0, 0.5)).get_figure()

```


![png](aug2020-spx-analysis-blog_files/output_16_0.png)



![png](aug2020-spx-analysis-blog_files/output_16_1.png)


Clearly, the strong performance of the BB- line almost entirely reflects the strong performance of Netflix, which makes up 84.8% of the group's market cap. Netflix's strong performance during the COVID quarantine certainly makes sense. More time at home in your PJs translates to more time for bingewatching Netflix.

A few important observations based on our investigation of the BB- and B+ groups. First, with such limited sample size for some of the credit ratings, there can be susbtantial sampling error and biases, making it impossible to generate any valid conclusions. Second, we can already posit that other (potentially related) factors than credit rating, such as the industry the company operates in, are more important in determining the stockprice performance during the COVID crisis.

With these caveats about data quality in mind, to reach an informal conclusion to our main question of how credit rating affects stock price performance, let's take one last look at the relationship between the two. We will adopt a simpler approach that we can visually interpret with ease: we will create a bar chart of the average stock price of each credit rating group during the entire time period, as well as the entire index average (a kind of informal one way analysis of variance approach). 

We will also reorder the columns according to match the credit rating score in descending order..

```python
#generate weighted average performance for entire S&P500 index
#add the Index_Weight column for each company
spx_cred_prices['Index_Weight'] = spx_cred_prices['Market_Cap'].div(spx_cred_prices['Market_Cap'].sum(), axis = 0)

#normalize by dividing by first day
spx_index_mean = spx_cred_prices.iloc[:,2:-2].div(spx_cred_prices.iloc[:,2], axis = 0)

#multiply each row by its weight 
norm_spx_index_mean = spx_index_mean.mul(spx_cred_prices['Index_Weight'], axis=0)
final_spx_index_mean = norm_spx_index_mean.sum().mul(100, axis=0)

order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'NR', 'Full_Index']

#add this to the wtd_spx_bygroup
wtd_spx_bygroupT['Full_Index'] = final_spx_index_mean
reordered_wtd_spx_bygroupT = wtd_spx_bygroupT[order] 
averages_bygroup = reordered_wtd_spx_bygroupT.mean(axis = 0)
averages_bygroup.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e03e0d0>




![png](aug2020-spx-analysis-blog_files/output_18_1.png)


### Conclusion

In general, we can see a slight downward tendency as the credit rating decreases, meaning that our prediction in the beginning was at least partially valid. However, the stock price performance does not strongly follow the creditworthiness, and from our exercise it seems to have more to do with variables not accounted for in the data here, such as industry (Coty and Netflix). 

This exercise was meant to demonstrate the steps involved in any data science project: importing the data, understanding the data, cleaning and preparing the data, processing the data, visualizing the results, investigating the initial findings, adjusting the investigation approach, and drawing conclusions.

One exercise for next time could be to complete a more thorough investigation of the determinants of stock price performance during the COVID crisis, with more data points, such as industry and other financial statistics beyond just market cap. Then, we could run a more proper statistical analysis and find correlation coefficients to determine which factors most strongly predict stock price performance.

Thanks for reading!
