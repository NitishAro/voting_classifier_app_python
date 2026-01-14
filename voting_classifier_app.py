import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Custom function to replace mlxtend for Python 3.13 compatibility
def plot_decision_boundary(clf, X, y, ax):
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu')

st.set_page_config(page_title="Voting Classifier Lab", layout="wide")
st.title("🗳️ Voting Classifier Visualizer")

# --- SIDEBAR ---
st.sidebar.header("1. Dataset Settings")
ds_type = st.sidebar.selectbox("Select Dataset", ("Linearly Separable", "Moons", "Circles"))
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)

st.sidebar.header("2. Choose Estimators")
estimators = []
if st.sidebar.checkbox("Logistic Regression", value=True):
    estimators.append(('lr', LogisticRegression()))
if st.sidebar.checkbox("Random Forest", value=True):
    estimators.append(('rf', RandomForestClassifier()))
if st.sidebar.checkbox("Decision Tree"):
    estimators.append(('dt', DecisionTreeClassifier()))
if st.sidebar.checkbox("Naive Bayes"):
    estimators.append(('nb', GaussianNB()))
if st.sidebar.checkbox("SVM"):
    estimators.append(('svc', SVC(probability=True)))

voting_type = st.sidebar.radio("Voting Type", ("hard", "soft"))

# --- DATA GENERATION ---
if ds_type == "Moons":
    X, y = make_moons(n_samples=300, noise=noise, random_state=42)
elif ds_type == "Circles":
    X, y = make_circles(n_samples=300, noise=noise, factor=0.5, random_state=42)
else:
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, 
                               n_clusters_per_class=1, class_sep=2.0 - noise)

# --- RUN ---
if st.sidebar.button("Run Algorithm"):
    if len(estimators) < 2:
        st.error("Select at least 2 models!")
    else:
        cols = st.columns(len(estimators) + 1)
        scores = {}

        for idx, (name, clf) in enumerate(estimators):
            clf.fit(X, y)
            acc = accuracy_score(y, clf.predict(X))
            scores[name] = acc
            with cols[idx]:
                st.write(f"**{name.upper()}**")
                fig, ax = plt.subplots(figsize=(4,4))
                plot_decision_boundary(clf, X, y, ax)
                st.pyplot(fig)
                st.write(f"Acc: {acc:.2%}")

        vc = VotingClassifier(estimators=estimators, voting=voting_type)
        vc.fit(X, y)
        vc_acc = accuracy_score(y, vc.predict(X))
        with cols[-1]:
            st.write("**VOTING**")
            fig, ax = plt.subplots(figsize=(4,4))
            plot_decision_boundary(vc, X, y, ax)
            st.pyplot(fig)
            st.write(f"**Final: {vc_acc:.2%}**")

        st.divider()
        st.table({"Model": list(scores.keys()) + ["COMBINED"], 
                  "Accuracy": [f"{v:.2%}" for v in scores.values()] + [f"{vc_acc:.2%}"]})
else:
    st.info("👈 Set parameters and click 'Run Algorithm'")