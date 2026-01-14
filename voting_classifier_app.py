import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
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
    try:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    except:
        pass
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu', s=20)
    ax.set_xticks([])
    ax.set_yticks([])

st.set_page_config(page_title="Voting Classifier Lab", layout="wide")
st.title("🗳️ Advanced Voting Classifier Visualizer")

# --- SIDEBAR ---
st.sidebar.header("1. Dataset Settings")
ds_type = st.sidebar.selectbox("Select Dataset", 
    ("Concentric Circles", "U-Shape", "XOR (Cross)", "Moons", "Linearly Separable", "Blobs"))
noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)

st.sidebar.header("2. Choose Estimators")
estimators = []
if st.sidebar.checkbox("Logistic Regression", value=True):
    estimators.append(('lr', LogisticRegression()))
if st.sidebar.checkbox("Random Forest", value=True):
    estimators.append(('rf', RandomForestClassifier(n_estimators=10)))
if st.sidebar.checkbox("Decision Tree"):
    estimators.append(('dt', DecisionTreeClassifier(max_depth=3)))
if st.sidebar.checkbox("Naive Bayes"):
    estimators.append(('nb', GaussianNB()))
if st.sidebar.checkbox("SVM (Linear)"):
    estimators.append(('svc', SVC(kernel='linear', probability=True)))

voting_type = st.sidebar.radio("Voting Type", ("hard", "soft"))

# --- DATA GENERATION ---
n_samples = 300
if ds_type == "Concentric Circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.3, random_state=42)
elif ds_type == "U-Shape":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X[y == 1] *= [1.5, -1.5] 
elif ds_type == "XOR (Cross)":
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    X += rng.normal(size=X.shape) * noise
elif ds_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
elif ds_type == "Blobs":
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise*3, random_state=42)
else:
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
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
        st.subheader("📊 Comparison Table")
        st.table({"Model": list(scores.keys()) + ["VOTING (COMBINED)"], 
                  "Accuracy Score": [f"{v:.2%}" for v in scores.values()] + [f"{vc_acc:.2%}"]})
else:
    st.info("👈 Set parameters and click 'Run Algorithm'")
