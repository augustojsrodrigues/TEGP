import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(
    page_title="Política de Inspeções Puras",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS customizado
# ============================================================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #f7fbff 0%, #eef5ff 45%, #ffffff 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #003366 0%, #005a9c 55%, #0099cc 100%);
            padding: 2.2rem;
            border-radius: 24px;
            box-shadow: 0 16px 40px rgba(0, 51, 102, 0.18);
            color: white;
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.18;
            margin-bottom: 0.6rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            opacity: 0.94;
            margin-bottom: 0.25rem;
        }
        .institution {
            font-size: 0.96rem;
            line-height: 1.55;
            opacity: 0.96;
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(0, 64, 128, 0.09);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 10px 28px rgba(20, 50, 90, 0.08);
            height: 100%;
        }
        .metric-label {
            font-size: 0.86rem;
            color: #52677a;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #003366;
        }
        .section-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(0, 64, 128, 0.09);
            border-radius: 22px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 10px 28px rgba(20, 50, 90, 0.07);
            margin-bottom: 1rem;
        }
        .small-note {
            color: #52677a;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .stButton > button {
            border-radius: 14px;
            height: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #003366 0%, #0077b6 100%);
            color: white;
            border: none;
            box-shadow: 0 8px 20px rgba(0, 51, 102, 0.18);
        }
        .stButton > button:hover {
            color: white;
            border: none;
            filter: brightness(1.06);
        }
        .gif-frame img {
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.10);
        }
        div[data-testid="stMetricValue"] {
            color: #003366;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Funções do modelo
# ============================================================
def build_model(eta, beta_weibull, lh, cf, cp, ci, cd_coef, valor_grande):
    """Cria a função objetivo da política de inspeções puras."""

    def fx(x):
        if x < 0:
            return 0.0
        return (beta_weibull / eta) * ((x / eta) ** (beta_weibull - 1)) * np.exp(-((x / eta) ** beta_weibull))

    def fh(h):
        if h < 0:
            return 0.0
        return lh * np.exp(-lh * h)

    def Rh(h):
        if h < 0:
            return 1.0
        return np.exp(-lh * h)

    def cd(h):
        return cd_coef * h

    def funcao_objetivo(Delta):
        if Delta <= 0:
            return np.inf

        custo_falha = 0.0
        custo_nao_falha = 0.0
        vida_falha = 0.0
        vida_nao_falha = 0.0

        for i in range(1, int(valor_grande) + 1):
            a = (i - 1) * Delta
            b = i * Delta

            # Caso 1: defeito surge no intervalo e a falha ocorre antes da inspeção i.
            try:
                int_falha_prob_cd = dblquad(
                    lambda h, x: cd(h) * fx(x) * fh(h),
                    a,
                    b,
                    lambda x: 0,
                    lambda x: b - x,
                    epsabs=1e-7,
                    epsrel=1e-5,
                )[0]

                int_falha_vida = dblquad(
                    lambda h, x: (x + h) * fx(x) * fh(h),
                    a,
                    b,
                    lambda x: 0,
                    lambda x: b - x,
                    epsabs=1e-7,
                    epsrel=1e-5,
                )[0]

                # Caso 2: defeito surge no intervalo e não falha até a inspeção i.
                int_nao_falha_prob_cd = quad(
                    lambda x: cd(b - x) * fx(x) * Rh(b - x),
                    a,
                    b,
                    epsabs=1e-7,
                    epsrel=1e-5,
                    limit=80,
                )[0]

                int_nao_falha_vida = quad(
                    lambda x: fx(x) * Rh(b - x),
                    a,
                    b,
                    epsabs=1e-7,
                    epsrel=1e-5,
                    limit=80,
                )[0]

            except Exception:
                return np.inf

            custo_falha += ((i - 1) * ci + cf) * int_falha_prob_cd
            custo_nao_falha += (i * ci + cp) * int_nao_falha_prob_cd
            vida_falha += int_falha_vida
            vida_nao_falha += b * int_nao_falha_vida

        custo_esperado = custo_falha + custo_nao_falha
        vida_esperada = vida_falha + vida_nao_falha

        if vida_esperada <= 0 or not np.isfinite(vida_esperada):
            return np.inf

        return custo_esperado / vida_esperada

    return funcao_objetivo


@st.cache_data(show_spinner=False)
def optimize_policy(eta, beta_weibull, lh, cf, cp, ci, cd_coef, valor_grande, lower_bound, upper_bound, n_grid):
    funcao_objetivo = build_model(eta, beta_weibull, lh, cf, cp, ci, cd_coef, valor_grande)

    res = minimize_scalar(
        funcao_objetivo,
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={"xatol": 1e-3},
    )

    deltas = np.linspace(lower_bound, upper_bound, int(n_grid))
    taxas = np.array([funcao_objetivo(float(d)) for d in deltas])

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "delta_otimo": float(res.x),
        "taxa_custo_otima": float(res.fun),
        "deltas": deltas,
        "taxas": taxas,
    }

# ============================================================
# Barra lateral
# ============================================================
st.sidebar.header("Parâmetros do modelo")
st.sidebar.caption("Os valores iniciais seguem o código original, mas podem ser alterados.")

with st.sidebar.expander("Distribuição do tempo até o defeito, X", expanded=True):
    eta = st.number_input("η, parâmetro de escala Weibull", min_value=0.001, value=50.0, step=1.0, format="%.4f")
    beta_weibull = st.number_input("β, parâmetro de forma Weibull", min_value=0.001, value=3.0, step=0.1, format="%.4f")

with st.sidebar.expander("Delay-time, H", expanded=True):
    lh = st.number_input("λh, taxa da distribuição exponencial", min_value=0.001, value=2.0, step=0.1, format="%.4f")

with st.sidebar.expander("Custos", expanded=True):
    cf = st.number_input("cf, custo de falha", min_value=0.0, value=100.0, step=10.0, format="%.4f")
    cp = st.number_input("cp, custo preventivo", min_value=0.0, value=10.0, step=1.0, format="%.4f")
    ci = st.number_input("ci, custo de inspeção", min_value=0.0, value=1.0, step=0.1, format="%.4f")
    cd_coef = st.number_input("Coeficiente de cd(h) = k · h", min_value=0.0, value=0.1, step=0.01, format="%.4f")

with st.sidebar.expander("Configurações numéricas", expanded=False):
    valor_grande = st.number_input("Número máximo de intervalos na soma", min_value=20, max_value=2000, value=200, step=20)
    lower_bound = st.number_input("Limite inferior para Δ", min_value=0.001, value=0.01, step=0.01, format="%.4f")
    upper_bound = st.number_input("Limite superior para Δ", min_value=0.01, value=float(eta * 4), step=5.0, format="%.4f")
    n_grid = st.slider("Pontos para desenhar o gráfico", min_value=20, max_value=160, value=60, step=10)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Rodar otimização", use_container_width=True)

# ============================================================
# Cabeçalho
# ============================================================
left, right = st.columns([1.15, 2.85], gap="large")

with left:
    try:
        st.image("ufpe_logo.png", use_container_width=True)
    except Exception:
        st.info("Coloque o arquivo **ufpe_logo.png** na mesma pasta do app.py para exibir a logo da UFPE.")

with right:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Política de Inspeções Puras</div>
            <div class="hero-subtitle">Sistemas sujeitos a custo de remanufatura devido a consequências geradas no estado defeituoso</div>
            <div class="institution">
                Universidade Federal de Pernambuco (UFPE)<br>
                Centro Acadêmico do Agreste (CAA)<br>
                Engenharia de Produção<br>
                Disciplina: Tópicos Especiais em Gestão da Produção
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gifs decorativos
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
with g1:
    st.image("https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif", caption="Modelagem probabilística", use_container_width=True)
with g2:
    st.image("https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif", caption="Otimização da política", use_container_width=True)
with g3:
    st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif", caption="Análise de resultados", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Explicação curta
# ============================================================
st.markdown(
    """
    <div class="section-card">
        <h3>Objetivo do aplicativo</h3>
        <p class="small-note">
        O aplicativo calcula a taxa de custo de longo prazo para uma política de inspeções puras.
        A variável de decisão é o tempo entre inspeções, representado por Δ. O otimizador busca o valor de Δ que minimiza a taxa de custo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Execução
# ============================================================
if run_button:
    if upper_bound <= lower_bound:
        st.error("O limite superior de Δ precisa ser maior que o limite inferior.")
        st.stop()

    with st.spinner("Calculando a política ótima. Dependendo do número de intervalos, isso pode levar alguns segundos."):
        result = optimize_policy(
            eta=eta,
            beta_weibull=beta_weibull,
            lh=lh,
            cf=cf,
            cp=cp,
            ci=ci,
            cd_coef=cd_coef,
            valor_grande=int(valor_grande),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            n_grid=int(n_grid),
        )

    if not result["success"]:
        st.warning(f"O otimizador retornou uma mensagem de atenção: {result['message']}")

    delta_otimo = result["delta_otimo"]
    taxa_custo_otima = result["taxa_custo_otima"]
    deltas = result["deltas"]
    taxas = result["taxas"]

    st.markdown("### Resultados da otimização")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tempo ótimo entre inspeções</div>
                <div class="metric-value">Δ* = {delta_otimo:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Taxa de custo mínima</div>
                <div class="metric-value">TC(Δ*) = {taxa_custo_otima:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Status do otimizador</div>
                <div class="metric-value">{'OK' if result['success'] else 'Atenção'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Curva da taxa de custo")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(deltas, taxas, linewidth=2.2, label="Taxa de custo")
    ax.scatter([delta_otimo], [taxa_custo_otima], s=120, zorder=5, label="Ponto ótimo")
    ax.axvline(delta_otimo, linestyle="--", linewidth=1.3)
    ax.axhline(taxa_custo_otima, linestyle="--", linewidth=1.3)
    ax.set_xlabel("Tempo entre inspeções, Δ")
    ax.set_ylabel("Taxa de custo")
    ax.set_title("Otimização da política de inspeções puras")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    with st.expander("Ver parâmetros utilizados"):
        st.write(
            {
                "eta": eta,
                "beta_weibull": beta_weibull,
                "lambda_h": lh,
                "cf": cf,
                "cp": cp,
                "ci": ci,
                "cd(h)": f"{cd_coef} * h",
                "valor_grande": int(valor_grande),
                "limite_inferior_delta": lower_bound,
                "limite_superior_delta": upper_bound,
                "pontos_grafico": n_grid,
            }
        )
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em **Rodar otimização**.")

st.markdown(
    """
    <div class="section-card">
        <h3>Observação sobre desempenho</h3>
        <p class="small-note">
        O modelo usa integrais numéricas dentro de somatórios. Para testes rápidos no Streamlit, o número máximo de intervalos foi deixado em 200 por padrão.
        Caso deseje maior aproximação em estudos finais, aumente esse valor na barra lateral, sabendo que o tempo de execução também aumenta.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
