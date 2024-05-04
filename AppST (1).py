#!/usr/bin/env python
# coding: utf-8

# In[125]:


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

# Sample Navigation with icons
st.sidebar.title('Navigation')
page = st.sidebar.radio(
    "Choose a page:",
    [
        ('🏛 Basic Heston Model', 'Basic Heston Model'),
        ('🧪 Bates Model', 'Bates Model'),
        ('📊 Black-Scholes Model', 'Black-Scholes Model'),
        ('📈 Double Heston Model', 'Double Heston Model'),
        ('📈 Volatility Surface with Excel', 'Volatility Surface with Excel')  
    ],
    index=0,  # Default to first option
    format_func=lambda x: x[0]
)


# In[126]:


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


# In[78]:


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


# In[79]:


def callback_func(xk, progress_bar, max_iter):
    # xk: Current parameter values (unused in this context, but required by minimize)
    global iteration_count
    iteration_count += 1
    progress = iteration_count / max_iter
    progress_bar.progress(progress)


# ### Black-Scholes

# In[80]:


def CallBS(S, sigma, K, T, r):
    d1 = (math.log(S / K) + (r + .5 * sigma**2) * T) / (sigma * T**.5)
    d2 = d1 - sigma * T**0.5
    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    DF = math.exp(-r * T)
    price=S * n1 - K * DF * n2
    return price


# In[81]:


def black_scholes(S, sigma, K, T, r, option_type='call'):
    # Compute d1 and d2 parameters
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Compute the discount factor
    DF = math.exp(-r * T)

    if option_type.lower() == 'call':
        # Calculate call price
        price = S * norm.cdf(d1) - K * DF * norm.cdf(d2)
    elif option_type.lower() == 'put':
        # Calculate put price
        price = K * DF * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


# ### Heston basic model

# In[82]:


def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    a = -0.5 * phi**2 - 0.5j * phi
    b = kappa - rho * sigma * 1j * phi
    g = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) / ((b + np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2)
    C = kappa * (((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * T - 2 / sigma**2 * np.log((1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g)))
    D = ((b - np.sqrt(b**2 - 2 * sigma**2 * a)) / sigma**2) * (1 - np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T)) / (1 - g * np.exp(-np.sqrt(b**2 - 2 * sigma**2 * a) * T))
    
    cf = np.exp(C * theta + D * v0 + 1j * phi * np.log(S0 * np.exp(r * T)))
    return cf


# In[83]:


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


# In[147]:


if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False
if 'bates_calibrated' not in st.session_state:
    st.session_state.bates_calibrated = False
if 'double_calibrated' not in st.session_state:
    st.session_state.double_calibrated = False


# ### Bates Model

# In[138]:


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


# In[139]:


def bates_price(S0, K, T, r, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J):
    params = (S0, T, r, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2

    for i in range(n):
        # Integration over phi using the trapezoidal rule
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi

        # Use the Bates characteristic function
        cf1 = bates_cf(phi - 1j, *params) / bates_cf(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        P1 += 1 / np.pi * np.real(temp1) * du

        cf2 = bates_cf(phi, *params)
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du

        phi += du

    # Calculate the call price from the probabilities P1 and P2
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price


# ### Double Heston

# In[140]:


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


# In[141]:


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

# In[142]:


# Title of the app
st.title("")


# In[143]:


def show_home():
    st.title("Heston Model Option Pricing")
    
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
    d = st.number_input("Dividend Yield (d)", value=0.02)  # Dividend rate input
    r_m_d = r - d  # Adjusted risk-free rate
    kappa = st.number_input("Mean Reversion Rate (kappa)", value=2.0)
    v0 = st.number_input("Initial Variance (v0)", value=0.04)
    theta = st.number_input("Long-Term Variance (theta)", value=0.2)
    sigma = st.number_input("Volatility of Volatility (sigma)", value=0.2)
    rho = st.number_input("Correlation between Stock and Variance (rho)", value=-0.6)
    if st.button("Calculate Heston Price"):
        price = heston_price(S0, K, T, r_m_d, kappa, v0, theta, sigma, rho)
        st.write(f"The Heston model price is: ${price:.2f}")


# In[144]:


def show_second_page():
    st.title("🦘 Bates Model Option Pricing")
    
    st.latex(r'''
\begin{align*}
\frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)} + dJ_t, \\
dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
\rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
\end{align*}
''')

    
    # Similar inputs, with additional parameters for jumps and dividend
    S0 = st.number_input("Spot Price (S0) [Bates]", value=100.0)
    K = st.number_input("Strike Price (K) [Bates]", value=100.0)
    T = st.number_input("Time to Maturity (T in years) [Bates]", value=1.0)
    r = st.number_input("Risk-Free Rate (r) [Bates]", value=0.05)
    d = st.number_input("Dividend Yield (d) [Bates]", value=0.02)  # Dividend rate input
    r_m_d = r - d  # Adjusted risk-free rate for dividends
    kappa = st.number_input("Mean Reversion Rate (kappa) [Bates]", value=2.0)
    v0 = st.number_input("Initial Variance (v0) [Bates]", value=0.04)
    theta = st.number_input("Long-Term Variance (theta) [Bates]", value=0.2)
    sigma = st.number_input("Volatility of Volatility (sigma) [Bates]", value=0.2)
    rho = st.number_input("Correlation between Stock and Variance (rho) [Bates]", value=-0.6)
    lambda_ = st.number_input("Jump Intensity (lambda) [Bates]", value=0.1)
    mu_J = st.number_input("Mean of Jump Size (mu_J) [Bates]", value=0.05)
    delta_J = st.number_input("Volatility of Jump (delta_J) [Bates]", value=0.1)
    if st.button("Calculate Bates Price"):
        price = bates_price(S0, K, T, r_m_d, kappa, v0, theta, sigma, rho, lambda_, mu_J, delta_J)
        st.write(f"The Bates model price is: ${price:.2f}")


# In[145]:


def Double_model_page():
    st.title("Double Heston Model Option Pricing")
    
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



    # Similar inputs, with additional parameters for jumps and dividend
    S0 = st.number_input("Spot Price (S0)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    d = st.number_input("Dividend Yield (d)", value=0.02)  # Dividend rate input
    r_m_d = r - d  # Adjusted risk-free rate for dividends
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


# In[146]:


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


# In[162]:


def show_excel_processor():
    st.title("🔧 Calibrator")
    st.write('Please select an Excel file containing the volatility surface')
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

    if uploaded_file:
        # Load the Excel file
        xl = pd.ExcelFile(uploaded_file)

        # Get a list of all sheets in the file
        sheet_names = xl.sheet_names

        # Dropdown to select a sheet
        selected_sheet = st.selectbox("Select a sheet", sheet_names)

        # Read the selected sheet
        if selected_sheet:
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            # Display data
            st.write(f"Displaying data from the sheet: {selected_sheet}")
            st.dataframe(df)
            
            # New code: Extract data for maturities, strikes, and implied volatilities
            maturity_rows = df.iloc[7:20]  # Rows for maturities from 1M to 10Y
            strike_prices = df.iloc[6, 3:]  # Strike prices from column D onwards

            T = []
            K = []
            IV = []

            def convert_to_years(maturity):
                if 'M' in maturity:
                    return int(maturity.replace('M', '')) / 12
                elif 'Y' in maturity:
                    return int(maturity.replace('Y', ''))

            for index, row in maturity_rows.iterrows():
                maturity = row['Unnamed: 2']  # Maturity label (e.g., "1M", "2M", etc.)
                maturity_in_years = convert_to_years(maturity)  # Convert maturity to years as fraction
                ivs = row[3:]  

                for strike, iv in zip(strike_prices, ivs):
                    T.append(maturity_in_years)
                    K.append(strike)
                    IV.append(iv / 100)  # Convert percentage to decimal for consistency
            
            S0 = df.iloc[2, 4]  # Spot price is in row 3, column E
            r = df.iloc[0, 10]  # Risk-free rate is in row 1, column K
            d = df.iloc[1, 10]
            r_m_d = r - d
            # Desired maturity times
            desired_T = [0.25, 0.5, 1, 2, 5, 10]
            # Desired strike prices (assuming S0 is defined somewhere in your code)
            desired_K = [S0*0.95, S0, S0*1.05]

            # Filter the data to keep only those entries where T is in desired_T and K is in desired_K
            T_h = [t for t, k in zip(T, K) if t in desired_T and k in desired_K]
            K_h = [k for k, t in zip(K, T) if t in desired_T and k in desired_K]
            IV_h = [iv for iv, t, k in zip(IV, T, K) if t in desired_T and k in desired_K]

            
            # Output some data to the user
            st.write("Spot Price (S0):", S0)
            st.write("Risk-Free Rate:", r)
            st.write("Dividend Rate:", d)

             # Plotting the volatility surface
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Convert lists to arrays for plotting
            T_plot = np.array(T)
            K_plot = np.array(K)
            IV_plot = np.array(IV)

            # Create the 3D plot
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

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            space1, col1, col2, col3, space2 = st.columns([2,2,2,2,1])
            with space1:
                pass  # Empty for additional spacing
            with col1:
                button_heston = st.button("Calibrate the Heston Model")
                st.latex(r'''
                \begin{align*}
                \frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)}, \\
                dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
                \rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
                \end{align*}
                ''')
            with col2:
                button_bates = st.button("Calibrate the Bates Model")
                st.latex(r'''
                \begin{align*}
                \frac{dS_t}{S_t} &= (r - d) \, dt + \sqrt{v_t} \, dW_t^{(1)} + dJ_t, \\
                dv_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^{(2)}, \\
                \rho \, dt &= dW_t^{(1)} \cdot dW_t^{(2)}
                \end{align*}
                ''')
            with col3:
                button_double = st.button("Calibrate the Double Model")
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
                pass  # Empty for additional spacing
            
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
                    {'type': 'ineq', 'fun': lambda x: 2 * x[0] * x[2] - x[3]**2 - 0.00001},  # Feller condition for process 1
                    {'type': 'ineq', 'fun': lambda x: 1. - x[9]},  # rho2 <= 1
                    {'type': 'ineq', 'fun': lambda x: x[9] + 1.},  # rho2 >= -1
                    {'type': 'ineq', 'fun': lambda x: x[8] - 0.00001},  # sigma2 > 0
                    {'type': 'ineq', 'fun': lambda x: x[5] - 0.00001},  # kappa2 > 0
                    {'type': 'ineq', 'fun': lambda x: x[6] - 0.00001},  # v02 > 0
                    {'type': 'ineq', 'fun': lambda x: x[7] - 0.00001},  # theta2 > 0
                    {'type': 'ineq', 'fun': lambda x: 2 * x[5] * x[7] - x[8]**2 - 0.00001}  # Feller condition for process 2
                    ]

                params_double = [3, 0.06, 0.06, 0.3, -0.5, 2, 0.03, 0.04, 0.2, -0.7]  # Initial parameter guesses
                
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


# In[160]:


def graph_heston_vs_price(Price, Price_Heston):
    # Let the user select the slice range via a slider
    slice_range = st.slider('Select the range of options to display', 
                            min_value=1, 
                            max_value=len(Price), 
                            value=(1, 20),
                            key='price_range_slider'
                           )

    # Slicing the lists to include only the options within the specified range
    Price_sliced = Price[slice_range[0]-1:slice_range[1]]
    Price_Heston_sliced = Price_Heston[slice_range[0]-1:slice_range[1]]

    # Set up a larger single figure with seaborn styles - adjust the size as needed.
    sns.set(style="whitegrid")  # Use seaborn style settings to enhance visual appearance
    fig, ax = plt.subplots(figsize=(14, 6))  # Increased width to accommodate more data points

    # Generate y-axis labels for the sliced range
    Y = np.arange(slice_range[0], slice_range[1] + 1)

    # Create the horizontal bar plots with an offset for the second set to distinguish them.
    bars1 = ax.barh(Y - 0.125, Price_sliced, height=0.25, color='lightcoral', label='BS model')
    bars2 = ax.barh(Y + 0.125, Price_Heston_sliced, height=0.25, color='lightskyblue', label='Heston model')

    # Add the legend with enhanced formatting
    ax.legend(frameon=True, framealpha=0.8, shadow=True, borderpad=1)

    # Set the title of the graph.
    ax.set_title('Comparison of Market vs Heston Model Prices', fontsize=16, fontweight='bold')

    # Set labels for the axes with enhanced styling.
    ax.set_xlabel('Option Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Index', fontsize=12, fontweight='bold')

    # Set y-axis ticks to match the indexes
    ax.set_yticks(Y)

    # Annotate each bar with its respective price, enhancing visibility and aesthetics
    for bar, price in zip(bars1, Price_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    for bar, price in zip(bars2, Price_Heston_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    # Display the plot in Streamlit
    st.pyplot(fig)


# In[161]:


def graph_bates_vs_price(Price, Price_bates):
    # Let the user select the slice range via a slider
    slice_range = st.slider('Select the range of options to display', 
                            min_value=1, 
                            max_value=len(Price), 
                            value=(1, 20))

    # Slicing the lists to include only the options within the specified range
    Price_sliced = Price[slice_range[0]-1:slice_range[1]]
    Price_Heston_sliced = Price_bates[slice_range[0]-1:slice_range[1]]

    # Set up a larger single figure with seaborn styles - adjust the size as needed.
    sns.set(style="whitegrid")  # Use seaborn style settings to enhance visual appearance
    fig, ax = plt.subplots(figsize=(14, 6))  # Increased width to accommodate more data points

    # Generate y-axis labels for the sliced range
    Y = np.arange(slice_range[0], slice_range[1] + 1)

    # Create the horizontal bar plots with an offset for the second set to distinguish them.
    bars1 = ax.barh(Y - 0.125, Price_sliced, height=0.25, color='lightcoral', label='BS model')
    bars2 = ax.barh(Y + 0.125, Price_Heston_sliced, height=0.25, color='lightskyblue', label='Bates model')

    # Add the legend with enhanced formatting
    ax.legend(frameon=True, framealpha=0.8, shadow=True, borderpad=1)

    # Set the title of the graph.
    ax.set_title('Comparison of Market vs Bates Model Prices', fontsize=16, fontweight='bold')

    # Set labels for the axes with enhanced styling.
    ax.set_xlabel('Option Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Index', fontsize=12, fontweight='bold')

    # Set y-axis ticks to match the indexes
    ax.set_yticks(Y)

    # Annotate each bar with its respective price, enhancing visibility and aesthetics
    for bar, price in zip(bars1, Price_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    for bar, price in zip(bars2, Price_Heston_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{price:.2f}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    # Display the plot in Streamlit
    st.pyplot(fig)


# In[97]:


def calculate_implied_volatility(market_price, spot_price, strike_price, interest_rate, time_to_expiry, tolerance, option_type):
    # Helper function to calculate initial guess based on moneyness
    def initial_guess():
        moneyness_factor = spot_price / (strike_price * math.exp(-interest_rate * time_to_expiry))
        return math.sqrt(2 * abs(math.log(moneyness_factor)) / time_to_expiry)

    # Helper function to calculate option price using Black-Scholes formula
    def option_price(volatility):
        d1 = (math.log(spot_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        if option_type == "call":
            price = spot_price * norm.cdf(d1) - strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
        elif option_type == "put":
            price = strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        return price

    # Helper function to calculate vega, the derivative of price with respect to volatility
    def calculate_vega(volatility):
        d1 = (math.log(spot_price / strike_price) + (interest_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        return spot_price * math.sqrt(time_to_expiry) * norm.pdf(d1)

    # Main iteration loop for Newton-Raphson method to find implied volatility
    volatility_guess = initial_guess()
    while True:
        current_price = option_price(volatility_guess)
        vega_value = calculate_vega(volatility_guess)
        price_difference = current_price - market_price
        if abs(price_difference / vega_value) < tolerance:
            break
        volatility_guess -= price_difference / vega_value

    return volatility_guess


# In[98]:


def graph_heston_vs_market_iv(IV, IV_heston):
    # Let the user select the slice range via a slider
    slice_range_iv = st.slider('Select the range of options to display', 
                            min_value=1, 
                            max_value=len(IV), 
                            value=(1, 10),
                            key='iv_range_slider'
                              )

    # Slicing the lists to include only the options within the specified range
    IV_sliced = IV[slice_range_iv[0]-1:slice_range_iv[1]]
    IV_heston_sliced = IV_heston[slice_range_iv[0]-1:slice_range_iv[1]]

    # Set up a larger single figure with seaborn styles - adjust the size as needed.
    sns.set(style="whitegrid")  # Use seaborn style settings to enhance visual appearance
    fig, ax = plt.subplots(figsize=(14, 6))  # Increased width to accommodate more data points

    # Generate y-axis labels for the sliced range
    Y = np.arange(slice_range_iv[0], slice_range_iv[1] + 1)

    # Create the horizontal bar plots with an offset for the second set to distinguish them.
    bars1 = ax.barh(Y - 0.125, IV_sliced, height=0.25, color='#90ee90', label='Market Implied Volatility')
    bars2 = ax.barh(Y + 0.125, IV_heston_sliced, height=0.25, color='#d8bfd8', label='Heston Model Implied Volatility')

    # Add the legend with enhanced formatting
    ax.legend(frameon=True, framealpha=0.8, shadow=True, borderpad=1)

    # Set the title of the graph.
    ax.set_title('Market vs Heston Model Implied Volatilities for Deep ITM, ATM and Deep OTM options', fontsize=16, fontweight='bold')

    # Set labels for the axes with enhanced styling.
    ax.set_xlabel('Implied Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Option Index', fontsize=12, fontweight='bold')

    # Set y-axis ticks to match the indexes
    ax.set_yticks(Y)

    # Annotate each bar with its respective implied volatility, enhancing visibility and aesthetics
    for bar, iv in zip(bars1, IV_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{iv:.2%}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    for bar, iv in zip(bars2, IV_heston_sliced):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{iv:.2%}',
                va='center', ha='left', fontsize=10, color='black', fontweight='bold')

    # Display the plot in Streamlit
    st.pyplot(fig)


# In[165]:


def K_fair(model, params, T):
    if model=="Heston":
        K_fair=params[2]+(params[1]-params[2])*(1-np.exp(-params[0]*T))/(params[0]*T)
    elif model=="Double":
        K_fair=params[2]+params[7]+(params[1]-params[2])*(1-np.exp(-params[0]*T))/(params[0]*T) + (params[6]-params[7])*(1-np.exp(-params[5]*T))/(params[5]*T)
    elif model=="Bates":
        K_fair=params[2]+(params[1]-params[2])*((1-np.exp(-params[0]*T))/(params[0]*T))+params[5]*(params[6]**2)


# In[99]:


if page[1] == 'Basic Heston Model':
    show_home()
elif page[1] == 'Bates Model':
    show_second_page()
elif page[1] == 'Black-Scholes Model':
    show_black_scholes()
elif page[1] == 'Volatility Surface with Excel':
    show_excel_processor()
elif page[1] == 'Double Heston Model':
    Double_model_page()

