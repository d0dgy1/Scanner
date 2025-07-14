#Once done will add functions to their own page and import them to tidy it up

import streamlit as st
import pandas as pd
import numpy as np

import ccxt
from datetime import datetime
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



#streamlit run xxx.py

st.set_page_config(
    page_title = 'Crypto Screener',
    page_icon = '📈',)

st.write('# Crypto Screener #')



##   Copy data now from Notebook and bring into Streamlit


#FUNCTIONS


#Function to get ticker data
#added cahcing
@st.cache_data
def get_bybit_ohlcv(tf='D', limit=50):

    #bybit = ccxt.bybit()
    bybit = ccxt.bybit({'enableRateLimit': True, 'rateLimit': 500})
    markets = pd.Series(bybit.load_markets())

    tickers = []
    for m in markets:
        #if m['id'].endswith('USDT') & (m['type'] == 'spot'):
        if m['id'].endswith('USDT') & (m['type'] == 'swap'):
            tickers.append(m['id'])
        
    ticker_data = {}

    for t in tickers:
        try:
            ohlcv = pd.DataFrame(bybit.fetch_ohlcv(symbol=t, timeframe=tf, limit=limit))
            ohlcv.rename(columns={0:'Time', 1:'Open', 2:'High', 3:'Low', 4:'Close', 5:'Volume'}, inplace=True)
            #Some columns come back empty, restart IDE
            ohlcv['Time'] = pd.to_datetime(ohlcv['Time'], unit='ms')
            ticker_data[t] = ohlcv
        except:
            pass

    return ticker_data

data = get_bybit_ohlcv()
data_num = len(data)
st.write(f'There are {data_num} tokens on ByBit')
#st.dataframe(data)





#0 is current day, 1 is yesterday
#Gets x days OHLCV RVOL, Min & MAX. All data there for the period to then make stats/conditions from
#Tried to cache here, doesnt speed it up at all
def get_xdays_ohlcv_rvmaxmin(ticker_data, days, period):
    d = -(days+1)
    ticker = pd.DataFrame([])
    for t in ticker_data:
        if len(ticker_data[t]) > abs(d):
            open = ticker_data[t]['Open'].iloc[d]
            high = ticker_data[t]['High'].iloc[d]
            low = ticker_data[t]['Low'].iloc[d]
            close = ticker_data[t]['Close'].iloc[d]
            vol = ticker_data[t]['Volume'].iloc[d]
            avgvol = ticker_data[t]['Volume'].rolling(12).mean().iloc[d]
            rvol = vol / avgvol
            #min and max for previosu day
            min = ticker_data[t]['Low'].rolling(period+1).min().iloc[d-1]
            max = ticker_data[t]['High'].rolling(period+1).max().iloc[d-1]
            #min and max for current day, for range calc
            CDmin = ticker_data[t]['Low'].rolling(period+1).min().iloc[d]
            CDmax = ticker_data[t]['High'].rolling(period+1).max().iloc[d]
            #today['HLrange'] = today.apply(lambda x: (x['close'] - x['min']) / (x['max'] - x['min']) * 100, axis=1)
            #Using CD min and max
            HLrange =  round((close- CDmin) / (CDmax - CDmin) * 100,2)
            #change min & max around, add range
            ticker = pd.concat([ticker, pd.DataFrame([[t, open, high, low, close, vol, rvol, HLrange, min, max]])], ignore_index=True)
            #ticker = pd.concat([ticker, pd.DataFrame([[t, open, high, low, close, vol, rvol, min, max]])], ignore_index=True)

        else:
            pass

    #change min and max around, add HLrange
    ticker.rename(columns = {0:'ticker', 1:'open', 2:'high', 3:'low', 4:'close', 5:'vol', 6:'rvol', 7:'HLrange', 8:'min', 9:'max'}, inplace=True)
    #ticker.rename(columns = {0:'ticker', 1:'open', 2:'high', 3:'low', 4:'close', 5:'vol', 6:'rvol', 7:'min', 8:'max'}, inplace=True)

    ticker['BO'] = np.where(ticker['high'] > ticker['max'], 1, 0)
    ticker['BD'] = np.where(ticker['low'] < ticker['min'], 1, 0)

    return ticker



#Returns BO & BD data for each period for x days
break_period = [3, 5, 7, 11, 19, 31, 43]
days = 7
list1 = []
for d in range(days):
    for p in break_period:
        exec(f'D{d}P{p:02d} = get_xdays_ohlcv_rvmaxmin(data, {d}, {p})')
        list1.append((f'D{d}P{p:02d}'))

#st.write(list1)


#Splits list1 above into BO & BD 
list2 = []
for r in range(len(list1)):
    BOsum = eval(list1[r])[eval(list1[r])['BO'] == 1]
    BDsum = eval(list1[r])[eval(list1[r])['BD'] == 1]
    exec(f'{list1[r]}BO = BOsum')
    exec(f'{list1[r]}BD = BDsum')
    
    list2.append((f'{list1[r]}BO'))
    list2.append((f'{list1[r]}BD'))

#st.write(list2)


#We want the breaks for each day
#Counts BO & BD for each period in each day and adds it to a df to plot for y
#format y(day)BO & BD
for d in range(days):
    #print(d)
    exec(f'BO{d} = []')
    exec(f'BD{d} = []')
    for l in list2:
        if (int(l[1]) == d) & (l[-2:] == 'BO'):
                exec(f'BO{d}.append(len(eval(l)))')
        if (int(l[1]) == d) & (l[-2:] == 'BD'):
                exec(f'BD{d}.append(-len(eval(l)))')


#x = list(map(str, break_period))
#x = ['3', '5', '7', '11', '19', '31', '43']
#Above isnt working so have to use this
x = ['three', 'five', 'seven', 'eleven', 'nineteen', 'thirty one', 'fourty three']
fig = go.Figure()
fig = make_subplots(rows=1, cols=1)
for d in range(days):
    exec(f"fig.add_trace(go.Bar(x={x}, y=BO{d}, text=BO{d}, textposition='outside', offsetgroup='{d}', marker=dict(color='green', opacity=1)), row=1, col=1)")
    exec(f"fig.add_trace(go.Bar(x={x}, y=BD{d}, text=BD{d}, textposition='outside', offsetgroup='{d}', marker=dict(color='red', opacity=1)), row=1, col=1)")
    
fig.update_layout(barmode='group',bargap=0.2, title=f'Breakout/Down Count for Period by Day, {data_num} tokens', xaxis_title='Period', yaxis_title='Count')
fig.update_traces(showlegend = False)
#fig.show()
st.plotly_chart(fig, use_container_width=True)



# #Print a TV list of condition
# select_condition = st.selectbox('Condition', ('BO', 'BD'))
# select_day = int(st.selectbox('Day', range(days)))
# select_break_period = int(st.selectbox('Period', break_period))

# exec(f"scan_num = len(D{select_day}P{select_break_period:02d}{select_condition})")
# scan_perc = round((scan_num / data_num) * 100, 2)
# st.write(f'{scan_num} ({scan_perc}%), tokens make this criteria')
# exec(f"st.code(', '.join('BYBIT:' + D{select_day}P{select_break_period:02d}{select_condition}['ticker']))")



# select_rvol = st.slider('RVOL', min_value=0.0, max_value=5.0, step=0.5)
# #Dont need to change period for this, but can change the day
# exec(f"scan_rvol = D{select_day}P03[D{select_day}P03['rvol'] > {select_rvol}][['ticker', 'rvol']].sort_values(['rvol'], ascending=False)")
# rvol_scan_perc = round((len(scan_rvol) / data_num)*100, 2)
# st.write(f'{len(scan_rvol)} ({rvol_scan_perc}%), tokens make this criteria')
# st.code(', '.join('BYBIT:' + scan_rvol['ticker']))
# st.dataframe(scan_rvol, hide_index=True)





with st.form('conditions'):
    st.write('Select Values')
    select_condition = st.selectbox('Condition', ('BO', 'BD'))
    select_day = int(st.selectbox('Day', range(days)))
    select_break_period = int(st.selectbox('Period', break_period))
    select_rvol = st.slider('RVOL', min_value=0.0, max_value=5.0, step=0.5)

    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write('Condition', select_condition, 'Day', select_day, 'Period', select_break_period, 'RVOL', select_rvol)



    #Sort these by RVOL as well
    exec(f"scan_result = D{select_day}P{select_break_period:02d}{select_condition}")
    break_scan_result = scan_result[scan_result['rvol'] >= select_rvol].sort_values(['rvol'], ascending=False)
    scan_perc = round((len(break_scan_result) / data_num) * 100, 2)
    st.write(f'BREAK: {len(break_scan_result)} ({scan_perc}%), stocks make this criteria')
    st.code(', '.join('BYBIT:' + break_scan_result['ticker']+'.P'))

    #avg vol is 11 days rolling, so use 11P
    exec(f"rvol_scan_result = D{select_day}P03")
    rvol_scan_result = rvol_scan_result[rvol_scan_result['rvol'] >= select_rvol].sort_values(['rvol'], ascending=False)
    rvol_scan_perc = round((len(rvol_scan_result) / data_num) * 100, 2)
    st.write(f'RVOL: {len(rvol_scan_result)} ({rvol_scan_perc}%), stocks make this criteria')
    st.code(', '.join('BYBIT:' + rvol_scan_result['ticker']+'.P'))
    st.write('Data RVOL')
    st.dataframe(rvol_scan_result, hide_index=True)

    select_number = st.number_input('Range Highs/Lows list', value=20)
    #sort by HLrange, then RVOL
    exec(f"hlrange_scan_result = D{select_day}P{select_break_period:02d}")
    #HLrange = rvol_scan_result.sort_values(['HLrange', 'rvol'], ascending=False)
    HLrange = hlrange_scan_result.sort_values(['HLrange', 'rvol'], ascending=False)
    #st.dataframe(HLrange.head(select_number), hide_index=True)
    st.write(f'Top {select_number} Coins at {select_break_period} Period Range Highs')
    st.code(', '.join('BYBIT:' + HLrange['ticker'].head(select_number)+'.P'))
    st.write(f'Bottom {select_number} Coins at {select_break_period} Period Range Lows')
    st.code(', '.join('BYBIT:' + HLrange['ticker'].tail(select_number)+'.P'))
    st.write('Data High/Low Range')
    st.dataframe(HLrange, hide_index=True)
