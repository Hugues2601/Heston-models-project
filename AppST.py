#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st

# Set page configuration for layout and initial wide mode
st.set_page_config(page_title='Heston Project', layout='wide')

# Remove Streamlit's default header by targeting the class and customize the new header
st.markdown("""
<style>
header {visibility: hidden;}
.css-18e3th9 {visibility: hidden;}
.css-1d391kg {padding-top: 3rem;}
</style>
<div style="background-color:#1E2A4A; color:white; padding: 20px 20px; font-weight: bold; font-size: 32px; border-radius: 10px; display: flex; align-items: center;">
    <img src="https://i.postimg.cc/B67xHWh3/Logo-emlyon-2021.png" style="width: 80px; height: 80px; margin-right: 20px; flex-shrink: 0;">
    <div style="flex-grow: 1; text-align: center;">Options market making with Heston models</div>
</div>
""", unsafe_allow_html=True)

# Custom CSS to specifically target sidebar navigation text and icons
st.markdown("""
<style>
/* Target the sidebar to increase text size and change font */
div[data-testid="stSidebar"] .stRadio > label {
    font-size: 24px !important;  /* Increase font size */
    font-weight: bold !important;
    font-family: 'Arial', sans-serif !important;  /* Ensure this matches your desired font */
}
/* Improve hover effect */
div[data-testid="stSidebar"] .stRadio > label:hover {
    background-color: #f0f2f6 !important;
    color: #1E2A4A !important;
}
/* Style for selected item */
div[data-testid="stSidebar"] .stRadio > label[data-baseweb="radio"] > div:first-child > div {
    background-color: #1E2A4A !important;
    color: white !important;
}
/* Sidebar overall styling */
div[data-testid="stSidebar"] > div {
    padding-top: 3rem;
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title('Navigation')
page = st.sidebar.radio(
    "Choose a page:",
    [
        ('🧮 Black-Scholes Model', 'Black-Scholes Model'),
        ('💹 Basic Heston Model', 'Basic Heston Model'),
        ('📉 Bates Model', 'Bates Model'),
        ('📈 Double Heston Model', 'Double Heston Model'),
        ('⚙️ Calibrator with Excel', 'Calibrator with Excel'),
        ('🎚️ Surface Slider', 'Surface Slider'),
        ('⚖️ Market Making', 'Market Making')
    ],
    index=0, 
    format_func=lambda x: x[0]
)


# In[3]:


st.markdown("""
<style>
button {
    display: block;
    margin: 0 auto;
    width: 50%;   /* Adjust width to your preference */
    height: 50px; /* Adjust height to your preference */
    font-size: 20px; /* Adjust font size to your preference */
    font-weight: bold;
}
div.stButton > button:first-child {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)


# In[4]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.graph_objects as go
import math
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import ipywidgets as widgets
import warnings
import time
import yfinance as yf
from datetime import datetime


# In[5]:


def callback_func(xk, progress_bar, max_iter):
    global iteration_count
    iteration_count += 1
    progress = iteration_count / max_iter
    progress_bar.progress(progress)


# ### Black-Scholes

# In[6]:


def CallBS(S, sigma, K, T, r):
    d1 = (math.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * T**.5)
    d2 = d1 - sigma * T**0.5
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    DF = math.exp(-r * T)
    price=S * n1 - K * DF * n2
    return price


# In[7]:


def black_scholes(S, sigma, K, T, r, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    DF = math.exp(-r * T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * DF * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * DF * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


# ### Heston basic model

# In[8]:


def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    a = -0.5 * phi**2 - 0.5j * phi
    b = kappa - rho * sigma * 1j * phi
    g = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) / ((b + np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2)
    C = kappa * (((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * T - 2 / sigma**2 * np.log((1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g)))
    D = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * (1 - np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T))
    
    cf = np.exp(C * theta + D * v0 + 1j * phi * np.log(S0 * np.exp(r * T)))
    return cf


# In[9]:


def heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):
    params = (S0, T, r, kappa, v0, theta, sigma, rho)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2
    for i in range(n):
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi
        cf1 = heston_cf(phi - 1j, *params) / heston_cf(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        P1 += 1 / np.pi * np.real(temp1) * du
        cf2 = heston_cf(phi, *params)
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du
        phi += du
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price


# In[10]:


if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False
if 'bates_calibrated' not in st.session_state:
    st.session_state.bates_calibrated = False
if 'double_calibrated' not in st.session_state:
    st.session_state.double_calibrated = False


# ### Bates Model

# In[11]:


import numpy as np

def bates_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J):
    a = -0.5 * phi**2 - 0.5j * phi
    b = kappa - rho * sigma * 1j * phi
    g = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) / ((b + np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2)
    C = kappa * (((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * T - 2 / sigma**2 * np.log((1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g)))
    D = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * (1 - np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T))
    
    jump_cf = np.exp(lambda_ * T * (np.exp(1j * phi * mu_J - 0.5 * phi**2 * delta_J**2) - 1))

    cf = np.exp(C * theta + D * v0 + 1j * phi * np.log(S0 * np.exp(r * T))) * jump_cf
    return cf


# In[12]:


def bates_price(S0, K, T, r, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J):
    params = (S0, T, r, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2

    for i in range(n):
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi

        cf1 = bates_cf(phi - 1j, *params) / bates_cf(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        P1 += 1 / np.pi * np.real(temp1) * du

        cf2 = bates_cf(phi, *params)
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du

        phi += du

    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price


# ### Double Heston

# In[13]:


def heston_double_cf_v2(u, S0, T, r, kappa1, v01, theta1, sigma1, rho1, kappa2, v02, theta2, sigma2, rho2):

    d1=np.sqrt(((kappa1-rho1*sigma1*u*1j)**2)+(sigma1**2)*u*(u+1j))
    d2=np.sqrt(((kappa2-rho2*sigma2*u*1j)**2)+(sigma2**2)*u*(u+1j))
    g1=(kappa1-rho1*sigma1*u*1j-d1)/(kappa1-rho1*sigma1*u*1j+d1)
    g2=(kappa2-rho2*sigma2*u*1j-d2)/(kappa2-rho2*sigma2*u*1j+d2)
    
    logterm1=(1-g1*np.exp(-d1*T))/(1-g1)
    logterm2=(1-g2*np.exp(-d2*T))/(1-g2)
    
    A1=((kappa1*theta1)/(sigma1**2))*((kappa1-rho1*sigma1*u*1j-d1)*T - 2*np.log(logterm1))
    A2=((kappa2*theta2)/(sigma2**2))*((kappa2-rho2*sigma2*u*1j-d2)*T - 2*np.log(logterm2))
    A=A1+A2
    
    first_B1=(kappa1-rho1*sigma1*u*1j-d1)/(sigma1**2)
    second_B1=(1-np.exp(-d1*T))/(1-g1*np.exp(-d1*T))
    B1=first_B1*second_B1
    
    first_B2=(kappa2-rho2*sigma2*u*1j-d2)/(sigma2**2)
    second_B2=(1-np.exp(-d2*T))/(1-g2*np.exp(-d2*T))
    B2=first_B2*second_B2
    
    
    first_term=np.exp(1j*u*np.log(S0))
    second_term=np.exp(A)
    third_term=np.exp(B1*v01)
    fourth_term=np.exp(B2*v02)
    
    psy=first_term*second_term*third_term*fourth_term
    return psy


# In[14]:


def double_price(S0, K, T, r, kappa1, v01, theta1, sigma1, rho1, kappa2, v02, theta2, sigma2, rho2):
    params = (S0, T, r, kappa1, v01, theta1, sigma1, rho1, kappa2, v02, theta2, sigma2, rho2)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2
    for i in range(n):
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi
        cf1 = heston_double_cf_v2(phi - 1j, *params) / heston_double_cf_v2(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        P1 += 1 / np.pi * np.real(temp1) * du
        cf2 = heston_double_cf_v2(phi, *params)
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du
        phi += du
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price


# ### App

# In[15]:


# Title of the app
st.title("")


# In[16]:


def show_home():
    st.title("💶 Heston Model Option Pricing")
    
    st.latex(r'''
\begin{align*}
\frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)}, \\
dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
\rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
\end{align*}
''')


    S0 = st.number_input("Spot Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    d = st.number_input("Dividend Yield (d)", value=0.02) 
    r_m_d = r - d 
    kappa = st.number_input("Mean Reversion Rate (kappa)", value=2.0)
    v0 = st.number_input("Initial Variance (v0)", value=0.04)
    theta = st.number_input("Long-Term Variance (theta)", value=0.2)
    sigma = st.number_input("Volatility of Volatility (sigma)", value=0.2)
    rho = st.number_input("Correlation between Stock and Variance (rho)", value=-0.6)
    if st.button("Calculate Heston Price"):
        price = heston_price(S0, K, T, r_m_d, kappa, v0, theta, sigma, rho)
        st.write(f"The Heston model price is: ${price:.2f}")


# In[17]:


def show_second_page():
    st.title("🦘 Bates Model Option Pricing")
    
    st.latex(r'''
\begin{align*}
\frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)} + dJ_t, \\
dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
\rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
\end{align*}
''')


    S0 = st.number_input("Spot Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    d = st.number_input("Dividend Yield (d)", value=0.02)  
    r_m_d = r - d 
    kappa = st.number_input("Mean Reversion Rate (kappa)", value=2.0)
    v0 = st.number_input("Initial Variance (v0)", value=0.04)
    theta = st.number_input("Long-Term Variance (theta)", value=0.2)
    sigma = st.number_input("Volatility of Volatility (sigma)", value=0.2)
    rho = st.number_input("Correlation between Stock and Variance (rho)", value=-0.6)
    lambda_ = st.number_input("Jump Intensity (lambda)", value=0.1)
    mu_J = st.number_input("Mean of Jump Size (mu_J)", value=0.05)
    delta_J = st.number_input("Volatility of Jump (delta_J)", value=0.1)
    if st.button("Calculate Bates Price"):
        price = bates_price(S0, K, T, r_m_d, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J)
        st.write(f"The Bates model price is: ${price:.2f}")


# In[18]:


def Double_model_page():
    st.title("💴 Double Heston Model Option Pricing")
    
    st.latex(r'''
\begin{align*}
\frac{dS_t}{S_t} &= (r-d) \, dt + \sqrt{v_{1,t}} \, dW_t^{(1)} + \sqrt{v_{2,t}} \, dW_t^{(2)}, \\
dv_{1,t} &= \kappa_1 (\theta_1 - v_{1,t}) \, dt + \sigma_1 \sqrt{v_{1,t}} \, dZ_t^{(1)}, \\
dv_{2,t} &= \kappa_2 (\theta_2 - v_{2,t}) \, dt + \sigma_2 \sqrt{v_{2,t}} \, dZ_t^{(2)}, \\
\rho_1 \, dt &= dW_t^{(1)} \cdot dZ_t^{(1)}, \\
\rho_2 \, dt &= dW_t^{(2)} \cdot dZ_t^{(2)}, \\
0 &= dW_t^{(1)} \cdot dW_t^{(2)}, \\
0 &= dZ_t^{(1)} \cdot dZ_t^{(2)}
\end{align*}
''')

    S0 = st.number_input("Spot Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    d = st.number_input("Dividend Yield (d)", value=0.02)  
    r_m_d = r - d  
    kappa1 = st.number_input("Mean Reversion Rate (kappa)", value=2.0)
    v01 = st.number_input("Initial Variance (v0)", value=0.04)
    theta1 = st.number_input("Long-Term Variance (theta)", value=0.2)
    sigma1 = st.number_input("Volatility of Volatility (sigma)", value=0.2)
    rho1 = st.number_input("Correlation between Stock and Variance (rho)", value=-0.6)
    kappa2 = st.number_input("Second Mean Reversion Rate (kappa)", value=2.0)
    v02 = st.number_input("Second Initial Variance (v0)", value=0.04)
    theta2 = st.number_input("Second Long-Term Variance (theta)", value=0.2)
    sigma2 = st.number_input("Second Volatility of Volatility (sigma)", value=0.2)
    rho2 = st.number_input("Second Correlation between Stock and Variance (rho)", value=-0.6)
    if st.button("Calculate Double Heston Price"):
        price = double_price(S0, K, T, r_m_d, kappa1, v01, theta1, sigma1, rho1, kappa2, v02, theta2, sigma2, rho2)
        st.write(f"The Double model price is: ${price:.2f}")


# In[19]:


def show_black_scholes():
    st.title("💵 Black-Scholes Model Option Pricing")
    
    st.latex(r'''
\begin{align*}
\frac{dS_t}{S_t} &= (r - d) \, dt + \sigma \, dW_t \\
\end{align*}
''')

    with st.form(key='my_form'):
        S = st.number_input("Spot Price (S)", value=100.0)
        sigma = st.number_input("Volatility (sigma)", value=0.2)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("Time to Maturity (T in years)", value=1.0)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
        d = st.number_input("Dividend Yield (d)", value=0.01)
        r_m_d = r-d
        option_type = st.selectbox("Option Type", ['call', 'put'])
        submit_button = st.form_submit_button(label='Calculate Option Price')

    if submit_button:
        price = black_scholes(S, sigma, K, T, r_m_d, option_type)
        st.write(f"The {option_type} option price is: ${price:.2f}")


# In[20]:


def show_excel_processor():
    st.title("🔧 Calibrator")
    st.write('Please select an Excel file containing the volatility surface')
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

    if uploaded_file:
        xl = pd.ExcelFile(uploaded_file)

        sheet_names = xl.sheet_names

        selected_sheet = st.selectbox("Select a sheet", sheet_names)

        if selected_sheet:
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        
            st.write(f"Displaying data from the sheet: {selected_sheet}")
            st.dataframe(df)
            
            maturity_rows = df.iloc[7:20] 
            strike_prices = df.iloc[6, 3:] 

            T = []
            K = []
            IV = []

            def convert_to_years(maturity):
                if 'M' in maturity:
                    return int(maturity.replace('M', '')) / 12
                elif 'Y' in maturity:
                    return int(maturity.replace('Y', ''))

            for index, row in maturity_rows.iterrows():
                maturity = row['Unnamed: 2'] 
                maturity_in_years = convert_to_years(maturity)  
                ivs = row[3:]  

                for strike, iv in zip(strike_prices, ivs):
                    T.append(maturity_in_years)
                    K.append(strike)
                    IV.append(iv / 100) 
            
            S0 = df.iloc[2, 4]  
            r = df.iloc[0, 10]  
            d = df.iloc[1, 10]
            r_m_d = r - d
            desired_T = [0.25, 0.5, 1, 2, 5, 10]
            desired_K = [S0*0.95, S0, S0*1.05]

            T_h = [t for t, k in zip(T, K) if t in desired_T and k in desired_K]
            K_h = [k for k, t in zip(K, T) if t in desired_T and k in desired_K]
            IV_h = [iv for iv, t, k in zip(IV, T, K) if t in desired_T and k in desired_K]

            
            st.write("Spot Price (S0):", S0)
            st.write("Risk-Free Rate:", r)
            st.write("Dividend Rate:", d)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            T_plot = np.array(T)
            K_plot = np.array(K)
            IV_plot = np.array(IV)

            fig = go.Figure(data=[go.Mesh3d(
                x=K, 
                y=T, 
                z=IV, 
                color='lightblue',
                opacity=0.50,
                intensity=IV,
                colorscale='RdYlBu_r',
                cmin=min(IV),
                cmax=max(IV)
                )])
            
            fig.update_layout(scene=dict(
                xaxis_title='Strike Prices',
                yaxis_title='Time to Maturity (Years)',
                zaxis_title='Implied Volatility'
                ), width=1300,
                height=800)

            st.plotly_chart(fig, use_container_width=True)


            space1, col1, space3, col2, space4, col3, space2 = st.columns([2,2,1,2,1,2,1])

            with space1:
                pass 

            with col1:
                with st.container():
                    button_heston = st.button("Calibrate the Heston Model")
                    st.write("Estimated time: 2 minutes")
                    st.latex(r'''
                    \begin{align*}
                    \frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)}, \\
                    dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
                    \rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
                    \end{align*}
                    ''')
            with space3:
                pass

            with col2:
                with st.container():
                    button_bates = st.button("Calibrate the Bates Model")
                    st.write("Estimated time: 4 minutes")
                    st.latex(r'''
                    \begin{align*}
                    \frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)} + dJ_t, \\
                    dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
                    \rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
                    \end{align*}
                    ''')
            with space4:
                pass

            with col3:
                with st.container():
                    button_double = st.button("Calibrate the Double Model")
                    st.write("Estimated time: 6 minutes")
                    st.latex(r'''
                    \begin{align*}
                    \frac{dS_t}{S_t} &= (r-d) \, dt + \sqrt{v_{1,t}} \, dW_t^{(1)} + \sqrt{v_{2,t}} \, dW_t^{(2)}, \\
                    dv_{1,t} &= \kappa_1 (\theta_1 - v_{1,t}) \, dt + \sigma_1 \sqrt{v_{1,t}} \, dZ_t^{(1)}, \\
                    dv_{2,t} &= \kappa_2 (\theta_2 - v_{2,t}) \, dt + \sigma_2 \sqrt{v_{2,t}} \, dZ_t^{(2)}, \\
                    \rho_1 \, dt &= dW_t^{(1)} \cdot dZ_t^{(1)}, \\
                    \rho_2 \, dt &= dW_t^{(2)} \cdot dZ_t^{(2)}, \\
                    0 &= dW_t^{(1)} \cdot dW_t^{(2)}, \\
                    0 &= dZ_t^{(1)} \cdot dZ_t^{(2)}
                    \end{align*}
                    ''')

            with space2:
                pass

            if button_heston:
                progress_bar = st.progress(0)
                iteration_count = 0
                estimated_max_iterations = 70
                Price = [CallBS(S0, IV_h[i], K_h[i], T_h[i], r_m_d) for i in range(len(T_h))]
                
                def rmse(params):
                    mse = 0
                    for i in range(len(T_h)):
                        mse += (heston_price(S0, K_h[i], T_h[i], r_m_d, *params) - Price[i])**2
                    return np.sqrt(mse / len(T_h))
                
                cons = [
                    {'type': 'ineq', 'fun': lambda x: 1. - x[4]}, #rho <= 1
                    {'type': 'ineq', 'fun': lambda x: x[4] + 1.}, #rho >= -1
                    {'type': 'ineq', 'fun': lambda x: x[3] - 0.00001}, #xi > 0
                    {'type': 'ineq', 'fun': lambda x: x[0] - 0.00001}, #kappa > 0
                    {'type': 'ineq', 'fun': lambda x: x[1] - 0.00001}, #nu0 > 0
                    {'type': 'ineq', 'fun': lambda x: x[2] - 0.00001}, #theta > 0
                    {'type': 'ineq', 'fun': lambda x: 2 * x[0] * x[2] - x[3]**2 - 0.00001}
                ]

                params = [3, 0.06, 0.06, 0.3, -0.5]  # Initial parameter guesses
                
                def update_progress(xk):
                    nonlocal iteration_count
                    iteration_count += 1
                    progress = min(iteration_count / estimated_max_iterations, 1.0)
                    progress_bar.progress(progress)
                
                result = minimize(rmse, params, method='SLSQP', constraints=cons, tol=1e-6, options={'disp': True},
                              callback=update_progress)
                new_params = result.x
                progress_bar.empty()
                st.session_state.calibrated = True
                st.session_state.parameters = new_params
            
            if st.session_state.calibrated:
                params_heston = st.session_state.parameters
                st.markdown(
    f"""
    <div style='text-align: center; border: 2px solid; padding: 10px;'>
        <h3>Optimized parameters</h3>
        <ul>
            <li>κ (Kappa): {st.session_state.parameters[0]:.4f}</li>
            <li>θ (Theta): {st.session_state.parameters[2]:.4f}</li>
            <li>σ (Sigma): {st.session_state.parameters[3]:.4f}</li>
            <li>ρ (Rho): {st.session_state.parameters[4]:.4f}</li>
            <li>ν₀ (V0): {st.session_state.parameters[1]:.4f}</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
                Price = [CallBS(S0, IV[i], K[i], T[i], r_m_d) for i in range(len(T))]
                Price_Heston = [heston_price(S0, K[i], T[i], r_m_d, *params_heston) for i in range(len(T))]
                Price_Heston_iv = [heston_price(S0, K_h[i], T_h[i], r_m_d, *params_heston) for i in range(len(T_h))]
                graph_heston_vs_price(Price, Price_Heston)
                IV_heston = [calculate_implied_volatility(Price_Heston_iv[i], S0, K_h[i], r_m_d, T_h[i], 10**-6, "call") for i in range(len(T_h))]
                graph_heston_vs_market_iv(IV_h, IV_heston)
                plot_K_fair_volatility("Heston", params_heston)
            
            if button_bates:
                progress_bar = st.progress(0)
                iteration_count = 0
                estimated_max_iterations = 80
                Price = [CallBS(S0, IV_h[i], K_h[i], T_h[i], r_m_d) for i in range(len(T_h))]
                
                def rmse(params):
                    mse = 0
                    for i in range(len(T_h)):
                        mse += (bates_price(S0, K_h[i], T_h[i], r_m_d, *params) - Price[i])**2
                    return np.sqrt(mse / len(T_h))
                
                cons = [
                    {'type': 'ineq', 'fun': lambda x: 1. - x[4]},  # rho <= 1
                    {'type': 'ineq', 'fun': lambda x: x[4] + 1.},  # rho >= -1
                    {'type': 'ineq', 'fun': lambda x: x[3] - 0.00001},  # xi > 0
                    {'type': 'ineq', 'fun': lambda x: x[0] - 0.00001},  # kappa > 0
                    {'type': 'ineq', 'fun': lambda x: x[1] - 0.00001},  # nu0 > 0
                    {'type': 'ineq', 'fun': lambda x: x[2] - 0.00001},  # theta > 0
                    {'type': 'ineq', 'fun': lambda x: 2 * x[0] * x[2] - x[3]**2 - 0.00001},  # Feller condition
                    {'type': 'ineq', 'fun': lambda x: x[5] - 0.00001},  # lambda > 0
                    {'type': 'ineq', 'fun': lambda x: x[7] - 0.00001}  # sigma_J > 0
                        ]

                
                params_bates = [3, 0.06, 0.06, 0.3, -0.5, 0.2, 0.02, 0.2]  # Initial parameter guesses
                
                def update_progress(xk):
                    nonlocal iteration_count
                    iteration_count += 1
                    progress = min(iteration_count / estimated_max_iterations, 1.0)
                    progress_bar.progress(progress)
                
                result_bates = minimize(rmse, params_bates, method='SLSQP', constraints=cons, tol=1e-4, options={'disp': True},
                              callback=update_progress)
                new_params_bates = result_bates.x
                progress_bar.empty()
                st.session_state.bates_calibrated = True
                st.session_state.parameters = new_params_bates
            
            if st.session_state.bates_calibrated:
                params_bates = st.session_state.parameters
                st.markdown(
                f"""
                <div style='text-align: center; border: 2px solid; padding: 10px;'>
                    <h3>Optimized parameters</h3>
                    <ul>
                        <li>κ (Kappa): {st.session_state.parameters[0]:.4f}</li>
                        <li>θ (Theta): {st.session_state.parameters[2]:.4f}</li>
                        <li>σ (Sigma): {st.session_state.parameters[3]:.4f}</li>
                        <li>ρ (Rho): {st.session_state.parameters[4]:.4f}</li>
                        <li>ν₀ (V0): {st.session_state.parameters[1]:.4f}</li>
                        <li>λ (Lambda): {st.session_state.parameters[5]:.4f}</li>
                        <li>μₛ (Mu_J): {st.session_state.parameters[6]:.4f}</li>
                        <li>σₛ (Sigma_J): {st.session_state.parameters[7]:.4f}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
                Price = [CallBS(S0, IV[i], K[i], T[i], r_m_d) for i in range(len(T))]
                Price_Bates = [bates_price(S0, K[i], T[i], r_m_d, *params_bates) for i in range(len(T))]
                Price_Bates_iv = [bates_price(S0, K_h[i], T_h[i], r_m_d, *params_bates) for i in range(len(T_h))]
                graph_heston_vs_price(Price, Price_Bates)
                IV_heston = [calculate_implied_volatility(Price_Bates_iv[i], S0, K_h[i], r_m_d, T_h[i], 10**-6, "call") for i in range(len(T_h))]
                graph_heston_vs_market_iv(IV_h, IV_heston)
                plot_K_fair_volatility("Bates", params_bates)
                
            if button_double:
                progress_bar = st.progress(0)
                iteration_count = 0
                estimated_max_iterations = 90
                Price = [CallBS(S0, IV_h[i], K_h[i], T_h[i], r_m_d) for i in range(len(T_h))]
                
                def rmse_double(params):
                    mse = 0
                    for i in range(len(T_h)):
                        mse += (double_price(S0, K_h[i], T_h[i], r_m_d, *params) - Price[i])**2
                    return np.sqrt(mse / len(T_h))
                
                cons = [
                    {'type': 'ineq', 'fun': lambda x: 1. - x[4]},  # rho1 <= 1
                    {'type': 'ineq', 'fun': lambda x: x[4] + 1.},  # rho1 >= -1
                    {'type': 'ineq', 'fun': lambda x: x[3] - 0.00001},  # sigma1 > 0
                    {'type': 'ineq', 'fun': lambda x: x[0] - 0.00001},  # kappa1 > 0
                    {'type': 'ineq', 'fun': lambda x: x[1] - 0.00001},  # v01 > 0
                    {'type': 'ineq', 'fun': lambda x: x[2] - 0.00001},  # theta1 > 0
                    {'type': 'ineq', 'fun': lambda x: 2 * x[0] * x[2] - x[3]**2 - 0.00001},  # Feller condition
                    {'type': 'ineq', 'fun': lambda x: 1. - x[9]},  # rho2 <= 1
                    {'type': 'ineq', 'fun': lambda x: x[9] + 1.},  # rho2 >= -1
                    {'type': 'ineq', 'fun': lambda x: x[8] - 0.00001},  # sigma2 > 0
                    {'type': 'ineq', 'fun': lambda x: x[5] - 0.00001},  # kappa2 > 0
                    {'type': 'ineq', 'fun': lambda x: x[6] - 0.00001},  # v02 > 0
                    {'type': 'ineq', 'fun': lambda x: x[7] - 0.00001},  # theta2 > 0
                    {'type': 'ineq', 'fun': lambda x: 2 * x[5] * x[7] - x[8]**2 - 0.00001}  # Feller condition
                    ]

                params_double = [3, 0.06, 0.06, 0.3, -0.5, 2, 0.03, 0.04, 0.2, -0.7]  
                
                def update_progress(xk):
                    nonlocal iteration_count
                    iteration_count += 1
                    progress = min(iteration_count / estimated_max_iterations, 1.0)
                    progress_bar.progress(progress)
                
                result = minimize(rmse_double, params_double, method='SLSQP', constraints=cons, tol=1e-4, options={'disp': True},
                              callback=update_progress)
                new_params_double = result.x
                progress_bar.empty()
                st.session_state.double_calibrated = True
                st.session_state.parameters = new_params_double
                
            if st.session_state.double_calibrated:
                params_double = st.session_state.parameters
                st.markdown(
                f"""
            <div style='text-align: center; border: 2px solid; padding: 10px;'>
                <h3>Optimized Parameters for Double Heston Model</h3>
                <h4>First Variance Process</h4>
                <ul>
                    <li>κ₁ (Kappa1): {st.session_state.parameters[0]:.4f}</li>
                    <li>ν₀₁ (V0_1): {st.session_state.parameters[1]:.4f}</li>
                    <li>θ₁ (Theta1): {st.session_state.parameters[2]:.4f}</li>
                    <li>σ₁ (Sigma1): {st.session_state.parameters[3]:.4f}</li>
                    <li>ρ₁ (Rho1): {st.session_state.parameters[4]:.4f}</li>
                </ul>
                <h4>Second Variance Process</h4>
                <ul>
                    <li>κ₂ (Kappa2): {st.session_state.parameters[5]:.4f}</li>
                    <li>ν₀₂ (V0_2): {st.session_state.parameters[6]:.4f}</li>
                    <li>θ₂ (Theta2): {st.session_state.parameters[7]:.4f}</li>
                    <li>σ₂ (Sigma2): {st.session_state.parameters[8]:.4f}</li>
                    <li>ρ₂ (Rho2): {st.session_state.parameters[9]:.4f}</li>
                </ul>
            </div>
            """
            ,
                unsafe_allow_html=True
            )  
                
                Price = [CallBS(S0, IV[i], K[i], T[i], r_m_d) for i in range(len(T))]
                Price_Double = [double_price(S0, K[i], T[i], r_m_d, *params_double) for i in range(len(T))]
                Price_Double_iv = [double_price(S0, K_h[i], T_h[i], r_m_d, *params_double) for i in range(len(T_h))]
                graph_heston_vs_price(Price, Price_Double)
                IV_heston = [calculate_implied_volatility(Price_Double_iv[i], S0, K_h[i], r_m_d, T_h[i], 10**-6, "call") for i in range(len(T_h))]
                graph_heston_vs_market_iv(IV_h, IV_heston)
                plot_K_fair_volatility("Double", params_double)


# In[21]:


def surface_slider():
    st.title('🎚️ Heston Model Volatility Surface')
    st.markdown("""
    <style>
        .big-font {
            font-size:16px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        mean_reversion = st.slider('Mean Reversion (κ)', 0.5, 7.0, 1.0, 0.1)
        initial_vol = st.slider('Initial Volatility (υ₀)', 0.1, 0.9, 0.2, 0.05)
        long_term_vol = st.slider('Long Term Volatility (θ)', 0.1, 0.9, 0.3, 0.05)
        correlation = st.slider('Correlation (ρ)', -0.9, 0.9, -0.5, 0.1)
        vol_of_vol = st.slider('Volatility of Volatility (σ)', 0.1, 2.0, 0.5, 0.1)
        risk_free_rate = st.slider('Risk-Free Rate (r)', -0.2, 0.2, 0.05, 0.01)
        elevation = st.slider('Elevation', 0, 90, 30, 5)
        azimuth = st.slider('Azimuth', 0, 360, 120, 10)

    initial_price = 100
    fixed_maturity = 1.0
    option_strikes = np.linspace(60, 150, 15) 
    maturity_periods = np.linspace(0.5, 5.5, 15) 
    model_params = [mean_reversion, initial_vol**2, long_term_vol**2, vol_of_vol, correlation]
    implied_vol_matrix = np.zeros((len(option_strikes), len(maturity_periods)))

    for i, strike in enumerate(option_strikes):
        for j, period in enumerate(maturity_periods):
            market_price = heston_price(initial_price, strike, period, risk_free_rate, *model_params)
            implied_vol = calculate_implied_volatility(market_price, initial_price, strike, risk_free_rate, period, 10**-8, "call")
            implied_vol_matrix[i, j] = 100 * implied_vol

    col1, col2, col3 = st.columns([2, 4, 2]) 

    with col2:
        fig = plt.figure(figsize=(6, 5)) 
        ax = fig.add_subplot(111, projection='3d', facecolor='white') 
        ax.plot_surface(np.meshgrid(maturity_periods, option_strikes)[0], np.meshgrid(maturity_periods, option_strikes)[1], implied_vol_matrix, cmap=cm.coolwarm, rstride=1, cstride=1, antialiased=False)
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike')
        ax.set_zlabel('Implied Vol')
        ax.set_title('')
        ax.grid(True)
        ax.view_init(elev=elevation, azim=azimuth)
        st.pyplot(fig)


# In[93]:


def mm():

    st.title("🏛️ Market Making")
    
    def load_excel(file):
        return pd.ExcelFile(file)

    file_path = 'market_data.xlsx'
    excel_file = load_excel(file_path)

    sheet_names = excel_file.sheet_names

    sheet_to_use = st.selectbox('Choose an index', sheet_names)

    df = pd.read_excel(excel_file, sheet_name=sheet_to_use)
    
    S0 = df.iloc[2, 4]  
    r = df.iloc[0, 10]  
    d = df.iloc[1, 10]
    r_m_d = r - d
    st.write(f"Spot Price (S0): {S0} | Risk-Free Rate: {r} | Dividend Rate: {d}")

    
    if sheet_to_use=="NDX-2015":
        params_ba=[0.1391, 0.0275, 0.1824, 0.2187 , -0.9773]
        params_ba_d=[3.7861, 0.0001, 0.0111, 0.2900, -1, 0.0634, 0.0212, 0.2664, 0.1838, -1]
        params_ba_j=[0.0435, 0.0198, 0.4115, 0.1829, -0.9891, 0.0727, -0.4033, 0.0355]
    elif sheet_to_use=="NDX-2014":
        params_ba=[0.1376, 0.0227, 0.1103, 0.1375, -0.9824]
        params_ba_d=[6.223, 0.0001, 0.0115, 0.3783, -1, 0.0711, 0.0144, 0.1501, 0.1461, -1]
        params_ba_j=[0.0585, 0.0151, 0.1756, 0.1320, -1, 0.2761, -0.0617, 0.1929]
    elif sheet_to_use=="SPX-2015":
        params_ba=[0.0729, 0.0220, 0.2802, 0.2021, -1]
        params_ba_d=[1.2763, 0.0197, 0.0432, 0.3321, -1, 0.0431, 0.0001, 0.1292, 0.0001, -0.7840]
        params_ba_j=[0.0949, 0.0195, 0.1437, 0.1651, -0.8779, 0.0109, -4.2098, 1.4172]
    elif sheet_to_use=="SPX-2014":
        params_ba_d=[0.5303, 0.0156, 0.0437, 0.2153, -1, 0.0363, 0.0001, 0.0744, 0.0001, -0.9735]
        params_ba_j=[0.1231, 0.0154, 0.1094, 0.1641, -0.9722, 0.0002, 1.0721, 1.9843]
        params_ba=[0.1098, 0.0161, 0.1226, 0.1382, -1]
        

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("1️⃣ Binary Options")
        
        strike_price_b = st.number_input(
        "Strike Price:",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="strike 1"
        )

        maturity_years_b = st.number_input(
        "Maturity (in years):",
        min_value=0.1,
        step=0.1,
        format="%.1f",
        key="tmt 1"
        )
        
        size_b = st.number_input(
        "Contract size:",
        min_value=1.0,
        step=1.0,
        format="%.1f",
        key="size binary"
        )
        
        c_o_p1 = st.selectbox('Type', ["High", "Low"], key="cop 1")
        
        if st.button("Compute Binary Bid and Ask"):
            bid, ask=binary(S0, strike_price_b, maturity_years_b, r_m_d, c_o_p1, params_ba)
            bid*=size_b
            ask*=size_b
            st.markdown(f"**Bid Price:** `{bid:.2f}`")
            st.markdown(f"**Ask Price:** `{ask:.2f}`")
            

    with col2:
        st.header("💶 European Options")
        strike_price_v = st.number_input(
        "Strike Price:",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="strike vanilla"
        )

        maturity_years_v = st.number_input(
        "Maturity (in years):",
        min_value=0.1,
        step=0.1,
        format="%.1f",
        key="tmt vanilla"
        )
        
        size_v = st.number_input(
        "Contract size:",
        min_value=1.0,
        step=1.0,
        format="%.1f",
        key="size vanilla"
        )
        
        c_o_p = st.selectbox('Type', ["Call", "Put"], key="cop v")
        
        if st.button("Compute Vanilla Bid and Ask"):
            bid, ask = bid_and_ask(S0, params_ba, params_ba_d, params_ba_j, strike_price_v, maturity_years_v, r_m_d, c_o_p)
            bid*=size_v
            ask*=size_v
            st.markdown(f"**Bid Price:** `{bid:.2f}`")
            st.markdown(f"**Ask Price:** `{ask:.2f}`")
            

    with col3:
        st.header("💱 Volatility Swaps")
        
        maturity_years_s = st.number_input(
        "Maturity (in years):",
        min_value=0.1,
        step=0.1,
        format="%.1f",
        key="tmt swap"
        )
        
        if st.button("Compute Swap Fair Strike"):
            bid, ask=k_fair_mm(params_ba, params_ba_d, params_ba_j, maturity_years_s)
            st.markdown(f"**Bid Fair Strike:** `{bid:.2f}`%")
            st.markdown(f"**Ask Fair Strike:** `{ask:.2f}`%")


# In[90]:


def bid_and_ask(S0, params_ba, params_ba_d, params_ba_j, strike, maturity, r_m_d, c_o_p):
    price_h=heston_price(S0, strike, maturity, r_m_d, *params_ba)
    price_hj=bates_price(S0, strike, maturity, r_m_d, *params_ba_j)
    price_d=double_price(S0, strike, maturity, r_m_d, *params_ba_d)
    
    if c_o_p=="Put":
        price_h=price_h-S0+strike*np.exp(-r_m_d*maturity)
        price_hj=price_hj-S0+strike*np.exp(-r_m_d*maturity)
        price_d=price_d-S0+strike*np.exp(-r_m_d*maturity)
    else:
        pass
    
    min_p=min(price_h, price_hj, price_d)
    max_p=max(price_h, price_hj, price_d)
    spread=0.01
    bid=min_p
    ask=max_p
    return bid, ask


# In[91]:


def binary(S0, K, T, r, typ, params_ba):
    kappa, v0, theta, sigma, rho = params_ba
    params = (S0, T, r, kappa, v0, theta, sigma, rho)

    P2 = 0.5  
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2

    for i in range(n):
        cf2 = heston_cf(phi, *params)
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du
        phi += du

    if typ == 'High':
        binary_price = np.exp(-r * T) * P2
    elif typ == 'Low':
        binary_price = np.exp(-r * T) * (1 - P2)
    spread=0.02
    bid=binary_price-spread/2
    ask=binary_price+spread/2
    return bid, ask


# In[83]:


def k_fair_mm(params_ba, params_ba_d, params_ba_j, T):

    K_fair_h = params_ba[2] + (params_ba[1] - params_ba[2]) * (1 - np.exp(-params_ba[0] * T)) / (params_ba[0] * T)

    K_fair_d = params_ba_d[2] + params_ba_d[7] + (params_ba_d[1] - params_ba_d[2]) * (1 - np.exp(-params_ba_d[0] * T)) / (params_ba_d[0] * T) \
                 + (params_ba_d[6] - params_ba_d[7]) * (1 - np.exp(-params_ba_d[5] * T)) / (params_ba_d[5] * T)
        
    K_fair_j = params_ba_j[2] + (params_ba_j[1] - params_ba_j[2]) * ((1 - np.exp(-params_ba_j[0] * T)) / (params_ba_j[0] * T)) \
                 + params_ba_j[5] * (params_ba_j[6]**2)
    vol_h = np.sqrt(K_fair_h) * 100
    vol_d = np.sqrt(K_fair_d) * 100
    vol_j = np.sqrt(K_fair_j) * 100
    vol_min=min(vol_h, vol_d, vol_j)
    vol_max=max(vol_h, vol_d, vol_j)
    return vol_min, vol_max


# In[58]:


def graph_heston_vs_price(Price, Price_Heston):
    slice_range = st.slider('Select the range of options to display', 
                            min_value=1, 
                            max_value=len(Price), 
                            value=(1, 20),
                            key='price_range_slider'
                           )

    Price_sliced = Price[slice_range[0]-1:slice_range[1]]
    Price_Heston_sliced = Price_Heston[slice_range[0]-1:slice_range[1]]

    sns.set(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(14, 6))  

    Y = np.arange(slice_range[0], slice_range[1] + 1)

    bars1 = ax.barh(Y - 0.125, Price_sliced, height=0.25, color='lightcoral', label='BS model')
    bars2 = ax.barh(Y + 0.125, Price_Heston_sliced, height=0.25, color='lightskyblue', label='Heston model')

    ax.legend(frameon=True, framealpha=0.8, shadow=True, borderpad=1)

    ax.set_title('Comparison of Market vs Heston Model Prices', fontsize=16, fontweight='bold')

    ax.set_xlabel('Option Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Index', fontsize=12, fontweight='bold')

    ax.set_yticks(Y)

    for bar, price in zip(bars1, Price_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    for bar, price in zip(bars2, Price_Heston_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold') 
    st.pyplot(fig)


# In[24]:


def calculate_implied_volatility(market_price, spot_price, strike_price, interest_rate, time_to_expiry, tolerance, option_type):
    def initial_guess():
        moneyness_factor = spot_price / (strike_price * math.exp(-interest_rate * time_to_expiry))
        return math.sqrt(2 * abs(math.log(moneyness_factor)) / time_to_expiry)

    def option_price(volatility):
        d1 = (math.log(spot_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        if option_type == "call":
            price = spot_price * norm.cdf(d1) - strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
        elif option_type == "put":
            price = strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        return price

    def calculate_vega(volatility):
        d1 = (math.log(spot_price / strike_price) + (interest_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return spot_price * math.sqrt(time_to_expiry) * norm.pdf(d1)

    volatility_guess = initial_guess()
    while True:
        current_price = option_price(volatility_guess)
        vega_value = calculate_vega(volatility_guess)
        price_difference = current_price - market_price
        if abs(price_difference / vega_value) < tolerance:
            break
        volatility_guess -= price_difference / vega_value

    return volatility_guess


# In[25]:


def graph_heston_vs_market_iv(IV, IV_heston):
    slice_range_iv = st.slider('Select the range of options to display', 
                            min_value=1, 
                            max_value=len(IV), 
                            value=(1, 10),
                            key='iv_range_slider'
                              )

    IV_sliced = IV[slice_range_iv[0]-1:slice_range_iv[1]]
    IV_heston_sliced = IV_heston[slice_range_iv[0]-1:slice_range_iv[1]]

    sns.set(style="whitegrid") 
    fig, ax = plt.subplots(figsize=(14, 6))  

    Y = np.arange(slice_range_iv[0], slice_range_iv[1] + 1)

    bars1 = ax.barh(Y - 0.125, IV_sliced, height=0.25, color='#90ee90', label='Market Implied Volatility')
    bars2 = ax.barh(Y + 0.125, IV_heston_sliced, height=0.25, color='#d8bfd8', label='Heston Model Implied Volatility')

    ax.legend(frameon=True, framealpha=0.8, shadow=True, borderpad=1)

    ax.set_title('Market vs Heston Model Implied Volatilities for Deep ITM, ATM and Deep OTM options', fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Implied Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Index', fontsize=12, fontweight='bold')

    ax.set_yticks(Y)

    for bar, iv in zip(bars1, IV_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{iv:.2%}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    for bar, iv in zip(bars2, IV_heston_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{iv:.2%}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    st.pyplot(fig)


# In[26]:


def plot_K_fair_volatility(model, params):
    sns.set(style="whitegrid")
    T_values = np.linspace(0.1, 10, 100) 
    volatilities = []

    for T in T_values:
        if model == "Heston":
            K_fair = params[2] + (params[1] - params[2]) * (1 - np.exp(-params[0] * T)) / (params[0] * T)
        elif model == "Double":
            K_fair = params[2] + params[7] + (params[1] - params[2]) * (1 - np.exp(-params[0] * T)) / (params[0] * T) \
                     + (params[6] - params[7]) * (1 - np.exp(-params[5] * T)) / (params[5] * T)
        elif model == "Bates":
            K_fair = params[2] + (params[1] - params[2]) * ((1 - np.exp(-params[0] * T)) / (params[0] * T)) \
                     + params[5] * (params[6]**2)
        volatility = np.sqrt(K_fair) * 100 
        volatilities.append(volatility)

    fig, ax = plt.subplots()
    ax.plot(T_values, volatilities, linestyle='-', color='red', linewidth=2)
    ax.set_title('Fair Strike for Volatility Swaps (%)')
    ax.set_xlabel('Time to Maturity T')
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, linestyle='--')
    space1, col1, space2 = st.columns([1,4,1])
    with space1:
        pass
    with col1:
        st.pyplot(fig)
    with space2:
        pass


# In[27]:


if page[1] == 'Basic Heston Model':
    show_home()
elif page[1] == 'Bates Model':
    show_second_page()
elif page[1] == 'Black-Scholes Model':
    show_black_scholes()
elif page[1] == 'Calibrator with Excel':
    show_excel_processor()
elif page[1] == 'Double Heston Model':
    Double_model_page()
elif page[1] == 'Surface Slider':
    surface_slider()
elif page[1] == 'Market Making':
    mm()


# In[ ]:




