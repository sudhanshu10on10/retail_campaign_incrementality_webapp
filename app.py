import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(
    page_title="Campaign Incrementality Analyzer",
    layout="wide"
)

st.title("Retail Campaign Incrementality Analyzer")
st.caption("DiD Causal Inference + MMM + RFM Segmentation — Superstore Data")

@st.cache_data
def load_and_process():
    df = pd.read_csv('superstore.csv', encoding='latin-1')
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.rename(columns={'Sub-Category': 'Sub_Category'}, inplace=True)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='mixed', dayfirst=True)
    df['treatment']   = df['Region'].isin(['West', 'East']).astype(int)
    df['post']        = (df['Order_Date'] >= pd.Timestamp('2017-06-01')).astype(int)
    df['Year_Month']  = df['Order_Date'].dt.to_period('M').astype(str)
    df['group_binary']= df['treatment'].map({
        1: 'Treatment (East + West)',
        0: 'Control (Central + South)'
    })

    # Monthly aggregation
    monthly = (
        df.groupby(['Year_Month', 'Region', 'treatment', 'post'])
        .agg(
            total_sales     =('Sales', 'sum'),
            order_count     =('Order_ID', 'nunique'),
            avg_order_value =('Sales', 'mean')
        ).reset_index()
    )
    monthly['month_num'] = pd.to_datetime(monthly['Year_Month']).dt.month
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num'] / 12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num'] / 12)

    # RFM
    snapshot = df['Order_Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer_ID').agg(
        recency   =('Order_Date', lambda x: (snapshot - x.max()).days),
        frequency =('Order_ID',   'nunique'),
        monetary  =('Sales',      'sum')
    ).reset_index()
    rfm['R_score'] = pd.qcut(rfm['recency'],  4, labels=[4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'],  4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    rfm['segment'] = rfm['RFM_score'].apply(
        lambda s: 'Champions' if s>=10 else
                  'Loyal'     if s>=7  else
                  'At-Risk'   if s>=5  else 'Lost'
    )
    df_seg = df.merge(rfm[['Customer_ID','segment']], on='Customer_ID', how='left')
    return df, monthly, rfm, df_seg

df, monthly, rfm, df_seg = load_and_process()

# ── Sidebar ──
page = st.sidebar.radio(
    "Select Analysis",
    ["Campaign Lift (DiD)", "Revenue Decomposition (MMM)", "Segment Response"]
)

# ── Page 1: DiD ──
if page == "Campaign Lift (DiD)":
    st.header("Did the campaign cause a real sales lift?")

    did_model   = smf.ols('total_sales ~ treatment * post', data=monthly).fit()
    causal_lift = did_model.params['treatment:post']
    pvalue      = did_model.pvalues['treatment:post']
    conf_low    = did_model.conf_int().loc['treatment:post', 0]
    conf_high   = did_model.conf_int().loc['treatment:post', 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Causal Lift",   f"${causal_lift:,.0f}")
    col2.metric("P-value",               f"{pvalue:.4f}")
    col3.metric("95% CI",                f"[${conf_low:,.0f}, ${conf_high:,.0f}]")

    st.caption("p < 0.05 = statistically significant. Current result is directional.")

    monthly_chart = (
        df.groupby(['Year_Month', 'group_binary'])['Sales']
        .sum().reset_index()
    )
    fig1 = px.line(
        monthly_chart,
        x='Year_Month', y='Sales', color='group_binary',
        title='Monthly Sales: Treatment vs Control',
        labels={'group_binary': 'Group', 'Sales': 'Total Sales ($)'}
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Page 2: MMM ──
elif page == "Revenue Decomposition (MMM)":
    st.header("What drove revenue variation?")

    features  = ['treatment', 'post', 'month_sin', 'month_cos', 'avg_order_value']
    X_scaled  = StandardScaler().fit_transform(monthly[features])
    mmm       = Ridge(alpha=1.0).fit(X_scaled, monthly['total_sales'])

    coefficients  = mmm.coef_
    feature_means = X_scaled.mean(axis=0)
    contributions = {
        feat: abs(coefficients[i] * feature_means[i] * len(monthly))
        for i, feat in enumerate(features)
    }
    total = sum(contributions.values())

    contrib_df = pd.DataFrame({
        'Driver': ['Region Effect', 'Post-Period Trend',
                   'Seasonality sin', 'Seasonality cos',
                   'Order Value Effect'],
        'Contribution_pct': [
            round(v / total * 100, 2)
            for v in contributions.values()
        ]
    })

    fig2 = px.bar(
        contrib_df.sort_values('Contribution_pct', ascending=False),
        x='Driver', y='Contribution_pct',
        title='Revenue Decomposition by Driver (%)',
        color='Driver', text_auto='.1f',
        labels={'Contribution_pct': 'Contribution (%)'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        contrib_df.sort_values('Contribution_pct', ascending=False),
        use_container_width=True
    )

# ── Page 3: Segment Response ──
elif page == "Segment Response":
    st.header("Which customer segments responded to the campaign?")

    pre  = df_seg[(df_seg['treatment']==1) & (df_seg['post']==0)].groupby('segment')['Sales'].mean()
    post = df_seg[(df_seg['treatment']==1) & (df_seg['post']==1)].groupby('segment')['Sales'].mean()

    lift_df = ((post - pre) / pre * 100).reset_index()
    lift_df.columns = ['segment', 'Lift_pct']
    lift_df['Lift_pct'] = lift_df['Lift_pct'].round(2)

    pre_df  = pre.reset_index().rename(columns={'Sales': 'Avg_Sales_Pre'})
    post_df = post.reset_index().rename(columns={'Sales': 'Avg_Sales_Post'})
    lift_df = lift_df.merge(pre_df, on='segment').merge(post_df, on='segment')

    fig3 = px.bar(
        lift_df.sort_values('Lift_pct', ascending=False),
        x='segment', y='Lift_pct',
        title='Campaign Sales Lift % by Customer Segment',
        color='segment', text_auto='.1f',
        labels={'Lift_pct': 'Lift (%)'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Recommendation")
    st.info(
        "Reallocate budget toward **Loyal** segment (+7.4% lift). "
        "Champions and At-Risk show negative incrementality — "
        "Champions buy regardless, At-Risk need personalised interventions."
    )
    st.dataframe(
        lift_df.sort_values('Lift_pct', ascending=False),
        use_container_width=True
    )