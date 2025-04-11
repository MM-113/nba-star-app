import streamlit as st
import numpy as np
from scipy.stats import nbinom, poisson

NBA_STD_DEV = 12.5
NB_R = 6.8
NUM_SIMULATIONS = 50000

class StarRating:
    @staticmethod
    def calculate(prob, mean_diff, over_consistency, volatility, trend_strength):
        base = min(5, 0.5 + (prob/20)**1.5)
        adjustments = (
            0.8 * mean_diff +
            0.6 * over_consistency +
            0.4 * np.log1p(volatility) -
            0.3 * trend_strength
        )
        return min(5, max(1, round(base + adjustments, 1)))

def simulate_game(home, away, target):
    def weighted_score(data, opp_allow):
        return (
            (data['avg'] * 0.7 + opp_allow * 0.3) *
            (1 + data['over_rate'] * 0.05)
        )

    home_score = weighted_score(home, away['allow'])
    away_score = weighted_score(away, home['allow'])
    combined_avg = home_score + away_score

    mc_scores = np.random.normal(home_score, NBA_STD_DEV, NUM_SIMULATIONS) +                 np.random.normal(away_score, NBA_STD_DEV, NUM_SIMULATIONS)
    mc_prob = np.mean(mc_scores > target) * 100

    def generate_nb(mean):
        p = NB_R / (NB_R + mean)
        return nbinom.rvs(NB_R, p, size=NUM_SIMULATIONS)

    nb_prob = np.mean((generate_nb(home_score) + generate_nb(away_score)) > target) * 100

    def generate_poisson(mean):
        return poisson.rvs(mean, size=NUM_SIMULATIONS)

    poisson_prob = np.mean((generate_poisson(home_score) + generate_poisson(away_score)) > target) * 100

    final_prob = mc_prob * 0.4 + nb_prob * 0.4 + poisson_prob * 0.2

    mean_diff = (combined_avg - target) / 5
    over_consistency = (home['over_rate'] + away['over_rate'] - 1)
    volatility = np.std([home['avg'], home['allow']])
    trend_strength = home['over_rate'] - away['over_rate']

    stars = StarRating.calculate(
        prob=final_prob,
        mean_diff=mean_diff,
        over_consistency=over_consistency,
        volatility=volatility,
        trend_strength=trend_strength
    )

    return {
        'combined_avg': combined_avg,
        'probability': final_prob,
        'stars': stars,
        'mc_prob': mc_prob,
        'nb_prob': nb_prob,
        'poisson_prob': poisson_prob,
        'home_score': home_score,
        'away_score': away_score,
        'recommendation': '大分' if final_prob >= 50 else '小分'
    }

st.set_page_config(page_title="NBA 星級預測系統", layout="centered")
st.title("🏀 NBA 星級預測系統")
st.markdown("三種機率模型 + 五維度星級評分")

st.subheader("輸入比賽數據")

with st.form("nba_form"):
    st.markdown("**主隊資料**")
    home_avg = st.number_input("主隊場均得分", min_value=50.0, max_value=150.0, value=110.0)
    home_allow = st.number_input("主隊場均失分", min_value=50.0, max_value=150.0, value=108.0)
    home_over = st.slider("主隊大分過盤率(%)", 0, 100, 55)

    st.markdown("**客隊資料**")
    away_avg = st.number_input("客隊場均得分", min_value=50.0, max_value=150.0, value=112.0)
    away_allow = st.number_input("客隊場均失分", min_value=50.0, max_value=150.0, value=111.0)
    away_over = st.slider("客隊大分過盤率(%)", 0, 100, 50)

    target = st.number_input("盤口總分", min_value=100.0, max_value=300.0, value=225.5)

    submitted = st.form_submit_button("開始模擬")

if submitted:
    home = {'avg': home_avg, 'allow': home_allow, 'over_rate': home_over / 100}
    away = {'avg': away_avg, 'allow': away_allow, 'over_rate': away_over / 100}

    result = simulate_game(home, away, target)

    st.subheader("📊 模擬結果")
    st.write(f"預期主隊得分: **{result['home_score']:.2f}**, 客隊得分: **{result['away_score']:.2f}**")
    st.write(f"預期總得分: **{result['combined_avg']:.2f}** | 盤口: **{target}**")

    st.write("### 🧮 各模型預測機率")
    st.write(f"- 蒙特卡洛模擬機率: **{result['mc_prob']:.1f}%**")
    st.write(f"- 負二項分布機率: **{result['nb_prob']:.1f}%**")
    st.write(f"- 泊松分布機率: **{result['poisson_prob']:.1f}%**")
    st.write(f"- 綜合加權機率: **{result['probability']:.1f}%**")

    st.write("### ⭐ 星級推薦")
    stars = int(round(result['stars']))
    st.write(f"推薦結果：**{result['recommendation']}**")
    st.markdown(f"{'★' * stars}{'☆' * (5 - stars)} ({result['stars']:.1f}/5.0)")

    explanation = {
        5.0: "🔥 極強信號：大分價值極高",
        4.0: "✅ 強力推薦：大分優勢明顯",
        3.0: "➖ 中性建議：需綜合判斷",
        2.0: "⚠️ 謹慎考慮：傾向小分",
        1.0: "❌ 強烈規避：小分機率高"
    }
    st.write(explanation.get(round(result['stars']), ""))
