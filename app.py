
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.express as px, plotly.graph_objects as go

st.set_page_config(page_title="Universal Bank - Pretrained Models", layout="wide")
st.title("Universal Bank — Marketing Dashboard (Pre-trained models)")

# Load pre-trained models (bundled with app)
MODEL_DT = joblib.load('model_dt.pkl')
MODEL_RF = joblib.load('model_rf.pkl')
MODEL_GB = joblib.load('model_gb.pkl')
MODEL_MAP = {'Decision Tree': MODEL_DT, 'Random Forest': MODEL_RF, 'Gradient Boosting': MODEL_GB}

# Top navigation (tabs)
tabs = st.tabs(["Overview (Insights)", "Models (Evaluate)", "Predict (Upload & Score)", "Data & Notes"])

@st.cache_data
def sample_df():
    rng = np.random.default_rng(1)
    n=500
    df = pd.DataFrame({
        'Age': rng.integers(22,65,n),
        'Experience': rng.integers(0,40,n),
        'Income': rng.integers(10,220,n),
        'Family': rng.integers(1,4,n),
        'CCAvg': np.round(rng.random(n)*10,2),
        'Education': rng.choice([1,2,3], n, p=[0.6,0.3,0.1]),
        'Mortgage': rng.integers(0,500,n),
        'Securities Account': rng.choice([0,1], n, p=[0.92,0.08]),
        'CD Account': rng.choice([0,1], n, p=[0.96,0.04]),
        'Online': rng.choice([0,1], n, p=[0.7,0.3]),
        'CreditCard': rng.choice([0,1], n, p=[0.7,0.3])
    })
    logits = (df['Income']*0.02 + df['CCAvg']*0.6 + (df['Education']-1)*1.0) - 5
    probs = 1/(1+np.exp(-logits))
    df['Personal Loan'] = (rng.random(n) < probs).astype(int)
    return df

# ---------- Overview ----------
with tabs[0]:
    st.header("Overview — Business + Statistical Insights")
    uploaded = st.file_uploader("Upload dataset (optional) — for best results upload UniversalBank.csv", type=['csv'], key='overview_upload')
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Dataset loaded for insights.")
    else:
        df = sample_df()
        st.info("Using bundled sample dataset. Upload your file to see real results.")

    st.subheader("Top 5 rows")
    st.dataframe(df.head())

    st.subheader("1) Income bin — observed conversion rate (actionable)")
    df['Income_bin'] = pd.cut(df['Income'], bins=[0,25,50,75,100,150,1000], labels=['0-25','25-50','50-75','75-100','100-150','150+'])
    bin_rates = df.groupby('Income_bin')['Personal Loan'].mean().reset_index()
    fig = px.bar(bin_rates, x='Income_bin', y='Personal Loan', text='Personal Loan', title='Acceptance rate by Income bin', labels={'Personal Loan':'Acceptance rate','Income_bin':'Income bin'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2) Education x Family — heatmap of acceptance rates (target segments)")
    pivot = pd.pivot_table(df, values='Personal Loan', index='Education', columns='Family', aggfunc='mean').fillna(0)
    heat = px.imshow(pivot.values, x=pivot.columns.astype(str), y=["Edu_"+str(i) for i in pivot.index], text_auto='.3f', color_continuous_scale='RdYlGn', labels=dict(x='Family', y='Education Level'))
    st.plotly_chart(heat, use_container_width=True)

    st.subheader("3) Income distribution by loan acceptance (violin)")
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(x=df['Personal Loan'].astype(str), y=df['Income'], box_visible=True, meanline_visible=True))
    fig2.update_layout(title='Income distribution - No vs Yes', xaxis_title='Personal Loan', yaxis_title='Income ($000)')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("4) Feature importance (from bundled Random Forest) — prioritise levers")
    try:
        rf = MODEL_RF
        importances = rf.feature_importances_
        feat = df.drop(columns=['Personal Loan']).columns.tolist()
        imp_df = pd.DataFrame({'feature':feat, 'importance':importances}).sort_values('importance', ascending=False)
        fig3 = px.bar(imp_df, x='feature', y='importance', title='Random Forest feature importances')
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.write("Could not compute/import feature importances:", e)

    st.subheader("5) CCAvg vs Age (color = Personal Loan) — demographic targeting")
    fig4 = px.scatter(df, x='Age', y='CCAvg', color=df['Personal Loan'].astype(str), size='Income', hover_data=['Income','Education','Family'], title='Age vs CCAvg (color=PersonalLoan)')
    st.plotly_chart(fig4, use_container_width=True)

# ---------- Models (Evaluate) ----------
with tabs[1]:
    st.header("Pre-trained Models — Quick Evaluation on Uploaded Data")
    st.markdown("These models are pre-trained and bundled with the app. Upload your dataset (UniversalBank.csv) to compute metrics on your data.")
    uploaded = st.file_uploader("Upload dataset for evaluation", type=['csv'], key='eval_upload')
    if uploaded:
        df_eval = pd.read_csv(uploaded)
        st.write("Rows:", df_eval.shape[0], "Columns:", df_eval.shape[1])
        if 'Personal Loan' not in df_eval.columns:
            st.error("Dataset must include 'Personal Loan' column to evaluate. Upload dataset with the target.")
        else:
            X = df_eval.drop(columns=['Personal Loan'])
            y = df_eval['Personal Loan']
            missing = [c for c in X.columns if c not in MODEL_RF.feature_names_in_]
            extra = [c for c in MODEL_RF.feature_names_in_ if c not in X.columns]
            if extra:
                st.warning(f"Missing expected columns: {extra}. Fill them with zeros or upload full dataset.")
                for c in extra:
                    X[c] = 0
            X = X[MODEL_RF.feature_names_in_]
            rows = []
            fig_roc = go.Figure()
            for name, m in MODEL_MAP.items():
                preds = m.predict(X)
                proba = m.predict_proba(X)[:,1] if hasattr(m,'predict_proba') else preds
                acc = accuracy_score(y, preds)
                prec = precision_score(y, preds, zero_division=0)
                rec = recall_score(y, preds, zero_division=0)
                f1 = f1_score(y, preds, zero_division=0)
                auc = roc_auc_score(y, proba)
                rows.append({'Algorithm':name, 'Accuracy':acc, 'Precision':prec, 'Recall':rec, 'F1':f1, 'AUC':auc})
                fpr, tpr, _ = roc_curve(y, proba)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})"))
                cm = confusion_matrix(y, preds)
                st.subheader(f"{name} - Confusion Matrix")
                cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred0','Pred1'], y=['True0','True1'], colorscale='Blues', showscale=True, text=cm, texttemplate='%{text}'))
                cm_fig.update_layout(width=500, height=400)
                st.plotly_chart(cm_fig)
            st.write("### Metrics summary")
            st.dataframe(pd.DataFrame(rows).set_index('Algorithm'))
            st.subheader("ROC curves overlay")
            fig_roc.update_layout(width=800, height=500, xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("Upload a dataset to evaluate pre-trained models on your data.")

# ---------- Predict (Upload & Score) ----------
with tabs[2]:
    st.header("Predict Personal Loan for New Customers (Upload & Download)")
    uploaded = st.file_uploader("Upload new customers CSV (no Personal Loan column required)", type=['csv'], key='pred_upload')
    model_choice = st.selectbox("Choose model for prediction", list(MODEL_MAP.keys()))
    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.write("Uploaded rows:", df_new.shape[0])
        expected = list(MODEL_RF.feature_names_in_)
        for c in expected:
            if c not in df_new.columns:
                df_new[c] = 0
        Xnew = df_new[expected]
        model = MODEL_MAP[model_choice]
        preds = model.predict(Xnew)
        proba = model.predict_proba(Xnew)[:,1] if hasattr(model,'predict_proba') else np.zeros(len(preds))
        df_new['Predicted_PersonalLoan'] = preds.astype(int)
        df_new['Prediction_Prob'] = np.round(proba,4)
        st.dataframe(df_new.head(50))
        csv = df_new.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, "predictions.csv", mime='text/csv')
    else:
        st.info("Upload new customer CSV to predict. The file should contain the same feature columns as UniversalBank dataset (ID can be present).")

# ---------- Data & Notes ----------
with tabs[3]:
    st.header("Data Dictionary & Deployment Notes")
    st.markdown("Universal Bank Data Fields - ID, Personal Loan, Age, Experience, Income, Zip code, Family, CCAvg, Education, Mortgage, Securities, CDAccount, Online, CreditCard")
    st.markdown("Deployment notes: models are bundled as model_dt.pkl, model_rf.pkl, model_gb.pkl. Include requirements.txt when deploying on Streamlit Cloud.")
