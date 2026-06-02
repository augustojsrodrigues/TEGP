import base64
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import dblquad, quad
from scipy.optimize import minimize_scalar


# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(
    page_title="Política de Inspeções Puras",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Estilo visual
# ============================================================
def image_to_base64(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


logo_base64 = image_to_base64("ufpe_logo.png")

st.markdown(
    """
    <style>
        :root {
            --bg: #f6f8fc;
            --card: #ffffff;
            --text: #172033;
            --muted: #667085;
            --primary: #1f4fd8;
            --primary-soft: #eaf0ff;
            --border: #e5e7eb;
            --accent: #02a6a6;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(31, 79, 216, 0.10), transparent 28%),
                radial-gradient(circle at 90% 20%, rgba(2, 166, 166, 0.10), transparent 25%),
                linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.88);
            border-right: 1px solid rgba(229, 231, 235, 0.9);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        .hero {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid rgba(229, 231, 235, 0.95);
            border-radius: 28px;
            padding: 28px 28px 22px 28px;
            box-shadow: 0 18px 55px rgba(15, 23, 42, 0.08);
            margin-bottom: 22px;
        }

        .institution {
            text-align: center;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.45;
            margin-bottom: 16px;
        }

        .ufpe-logo {
            width: 54px;
            height: auto;
            display: block;
            margin: 0 auto 6px auto;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 800;
            text-align: center;
            color: #111827;
            margin: 8px 0 10px 0;
            letter-spacing: -0.04em;
        }

        .hero-subtitle {
            text-align: center;
            color: #475467;
            font-size: 1.03rem;
            max-width: 850px;
            margin: 0 auto;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(229, 231, 235, 0.95);
            border-radius: 22px;
            padding: 20px 20px 16px 20px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.07);
            min-height: 132px;
        }

        .metric-label {
            color: #667085;
            font-size: 0.90rem;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #101828;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .metric-footnote {
            color: #667085;
            font-size: 0.82rem;
            margin-top: 6px;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid rgba(229, 231, 235, 0.95);
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 14px 45px rgba(15, 23, 42, 0.07);
            margin-bottom: 18px;
        }

        .tech-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 8px;
        }

        .tech-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #ffffff 0%, #f3f7ff 100%);
            border: 1px solid rgba(229, 231, 235, 0.95);
            border-radius: 20px;
            padding: 18px;
            min-height: 120px;
        }

        .tech-title {
            font-weight: 800;
            color: #172033;
            margin-bottom: 5px;
        }

        .tech-text {
            color: #667085;
            font-size: 0.88rem;
            line-height: 1.35;
            max-width: 85%;
        }

        .gear {
            position: absolute;
            right: 14px;
            bottom: 8px;
            font-size: 2.8rem;
            opacity: 0.18;
            animation: spin 5.5s linear infinite;
        }

        .scanner {
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 4px;
            background: linear-gradient(90deg, transparent, #1f4fd8, #02a6a6, transparent);
            animation: scan 2.4s ease-in-out infinite;
        }

        .conveyor {
            position: absolute;
            right: 12px;
            bottom: 17px;
            width: 86px;
            height: 22px;
            border-radius: 99px;
            background: repeating-linear-gradient(90deg, #d9e3ff 0 10px, #eef3ff 10px 20px);
            opacity: 0.75;
            animation: belt 1.1s linear infinite;
        }

        .pulse-dot {
            position: absolute;
            right: 34px;
            bottom: 54px;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #02a6a6;
            box-shadow: 0 0 0 rgba(2, 166, 166, 0.4);
            animation: pulse 1.8s infinite;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes scan {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(0%); }
            100% { transform: translateX(100%); }
        }

        @keyframes belt {
            from { background-position: 0 0; }
            to { background-position: 20px 0; }
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(2, 166, 166, 0.40); }
            70% { box-shadow: 0 0 0 18px rgba(2, 166, 166, 0); }
            100% { box-shadow: 0 0 0 0 rgba(2, 166, 166, 0); }
        }

        div.stButton > button:first-child {
            width: 100%;
            min-height: 3.2rem;
            border-radius: 16px;
            border: 0;
            color: white;
            font-weight: 800;
            background: linear-gradient(135deg, #1f4fd8 0%, #02a6a6 100%);
            box-shadow: 0 14px 30px rgba(31, 79, 216, 0.22);
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(31, 79, 216, 0.28);
        }

        .small-note {
            color: #667085;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        @media (max-width: 900px) {
            .tech-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.8rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Cabeçalho
# ============================================================
logo_html = (
    f'<img class="ufpe-logo" src="data:image/png;base64,{logo_base64}" alt="UFPE">'
    if logo_base64
    else ""
)

st.markdown(
    f"""
    <div class="hero">
        <div class="institution">
            {logo_html}
            <strong>Universidade Federal de Pernambuco (UFPE)</strong><br>
            Centro Acadêmico do Agreste (CAA)<br>
            Engenharia de Produção<br>
            Disciplina: Tópicos Especiais em Gestão da Produção
        </div>
        <div class="hero-title">Política de Inspeções Puras</div>
        <div class="hero-subtitle">
            Otimização do <em>tempo entre inspeções</em> em sistemas sujeitos a custo de remanufatura
            devido às consequências geradas no estado defeituoso.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Animações criadas no próprio app
# ============================================================
st.markdown(
    """
    <div class="tech-grid">
        <div class="tech-card">
            <div class="scanner"></div>
            <div class="tech-title">Inspeção inteligente</div>
            <div class="tech-text">Monitoramento do sistema antes da falha funcional.</div>
            <div class="gear">⚙️</div>
        </div>
        <div class="tech-card">
            <div class="scanner"></div>
            <div class="tech-title">Gestão da produção</div>
            <div class="tech-text">A decisão de inspeção afeta custo, ciclo operacional e desempenho.</div>
            <div class="conveyor"></div>
        </div>
        <div class="tech-card">
            <div class="scanner"></div>
            <div class="tech-title">Otimização</div>
            <div class="tech-text">Busca automática pelo intervalo que minimiza a taxa de custo.</div>
            <div class="pulse-dot"></div>
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Entradas do usuário
# ============================================================
st.sidebar.markdown("### Parâmetros do modelo")
st.sidebar.caption("Os valores iniciais reproduzem o código-base. Todos podem ser alterados.")

eta = st.sidebar.number_input(
    "Escala da Weibull, $\\eta$",
    min_value=0.01,
    value=50.00,
    step=1.00,
    format="%.2f",
)

beta_shape = st.sidebar.number_input(
    "Forma da Weibull, $\\beta$",
    min_value=0.01,
    value=3.00,
    step=0.10,
    format="%.2f",
)

lh = st.sidebar.number_input(
    "Taxa do delay-time exponencial, $\\lambda_h$",
    min_value=0.01,
    value=2.00,
    step=0.10,
    format="%.2f",
)

st.sidebar.markdown("### Custos")
cf = st.sidebar.number_input(
    "Custo de falha, $c_f$",
    min_value=0.00,
    value=100.00,
    step=1.00,
    format="%.2f",
)

cp = st.sidebar.number_input(
    "Custo preventivo, $c_p$",
    min_value=0.00,
    value=10.00,
    step=1.00,
    format="%.2f",
)

ci = st.sidebar.number_input(
    "Custo de inspeção, $c_i$",
    min_value=0.00,
    value=1.00,
    step=0.10,
    format="%.2f",
)

cd_unit = st.sidebar.number_input(
    "Custo do estado defeituoso por unidade de tempo, $c_d$",
    min_value=0.00,
    value=0.10,
    step=0.01,
    format="%.2f",
)


# ============================================================
# Funções do modelo
# ============================================================
VALOR_GRANDE = 1000
XATOL = 0.01
MAXITER = 60
N_PONTOS_GRAFICO = 35


@st.cache_data(show_spinner=False)
def funcao_objetivo(
    Delta: float,
    eta: float,
    beta_shape: float,
    lh: float,
    cf: float,
    cp: float,
    ci: float,
    cd_unit: float,
) -> float:
    Delta = float(Delta)

    def fx(x: float) -> float:
        return (beta_shape / eta) * ((x / eta) ** (beta_shape - 1)) * np.exp(-((x / eta) ** beta_shape))

    def fh(h: float) -> float:
        return lh * np.exp(-lh * h)

    def Rh(h: float) -> float:
        return np.exp(-lh * h)

    def cd(h: float) -> float:
        return cd_unit * h

    # Limite prático para reduzir tempo computacional sem expor configuração ao usuário.
    # O valor máximo permanece compatível com o código-base.
    limite_por_horizonte = int(math.ceil((eta * 5.0) / max(Delta, 1e-9))) + 2
    limite_soma = max(5, min(VALOR_GRANDE, limite_por_horizonte))

    prob_falha = 0.0
    prob_nao_falha = 0.0
    custo_falha = 0.0
    custo_nao_falha = 0.0
    vida_falha = 0.0
    vida_nao_falha = 0.0

    for i in range(1, limite_soma):
        inicio = (i - 1) * Delta
        fim = i * Delta

        p1 = dblquad(
            lambda h, x: fx(x) * fh(h),
            inicio,
            fim,
            lambda x: 0.0,
            lambda x: fim - x,
        )[0]

        p2 = quad(
            lambda x: fx(x) * Rh(fim - x),
            inicio,
            fim,
        )[0]

        c1_integral = dblquad(
            lambda h, x: cd(h) * fx(x) * fh(h),
            inicio,
            fim,
            lambda x: 0.0,
            lambda x: fim - x,
        )[0]

        c2_integral = quad(
            lambda x: cd(fim - x) * fx(x) * Rh(fim - x),
            inicio,
            fim,
        )[0]

        v1 = dblquad(
            lambda h, x: (x + h) * fx(x) * fh(h),
            inicio,
            fim,
            lambda x: 0.0,
            lambda x: fim - x,
        )[0]

        v2 = fim * quad(
            lambda x: fx(x) * Rh(fim - x),
            inicio,
            fim,
        )[0]

        prob_falha += p1
        prob_nao_falha += p2
        custo_falha += (((i - 1) * ci) + cf) * c1_integral
        custo_nao_falha += ((i * ci) + cp) * c2_integral
        vida_falha += v1
        vida_nao_falha += v2

        if fim >= eta * 5.0 and (p1 + p2) < 1e-10:
            break

    custo_esperado = custo_falha + custo_nao_falha
    vida_esperada = vida_falha + vida_nao_falha

    if vida_esperada <= 0 or not np.isfinite(vida_esperada):
        return float("inf")

    taxa_custo = custo_esperado / vida_esperada
    return float(taxa_custo)


def gerar_grafico(deltas: np.ndarray, custos: np.ndarray, delta_otimo: float, custo_otimo: float):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(deltas, custos, linewidth=2.3)
    ax.scatter([delta_otimo], [custo_otimo], s=95, zorder=4)
    ax.axvline(delta_otimo, linestyle="--", linewidth=1.3)
    ax.axhline(custo_otimo, linestyle="--", linewidth=1.3)
    ax.set_xlabel(r"Tempo entre inspeções, $\Delta$")
    ax.set_ylabel("Taxa de custo")
    ax.set_title("Taxa de custo em função do tempo entre inspeções")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ============================================================
# Execução principal
# ============================================================
left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top: 0;">Rodar otimização</h3>
            <p class="small-note">
                Clique no botão abaixo para calcular a variável de decisão ótima,
                isto é, o <em>tempo entre inspeções</em> que minimiza a taxa de custo do modelo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_button = st.button("Executar otimização", type="primary")

with right_col:
    st.markdown(
        f"""
        <div class="section-card">
            <h3 style="margin-top: 0;">Modelo em análise</h3>
            <p class="small-note">
                Defeito: Weibull com <em>η</em> = {eta:.2f} e <em>β</em> = {beta_shape:.2f}.<br>
                Delay-time: exponencial com <em>λ<sub>h</sub></em> = {lh:.2f}.<br>
                Custo do estado defeituoso: <em>c<sub>d</sub></em> = {cd_unit:.2f} por unidade de tempo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_button:
    lower_bound = 0.01
    upper_bound = eta * 4.00

    progress_bar = st.progress(0, text="Preparando otimização: 0%")
    status_placeholder = st.empty()
    eval_counter = {"n": 0}

    def objetivo_com_progresso(delta: float) -> float:
        eval_counter["n"] += 1
        pct = min(78, int((eval_counter["n"] / MAXITER) * 78))
        progress_bar.progress(pct, text=f"Otimizando a variável de decisão Δ: {pct}%")
        status_placeholder.caption(f"Avaliação {eval_counter['n']} de aproximadamente {MAXITER}.")
        return funcao_objetivo(delta, eta, beta_shape, lh, cf, cp, ci, cd_unit)

    resultado = minimize_scalar(
        objetivo_com_progresso,
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={"xatol": XATOL, "maxiter": MAXITER},
    )

    delta_otimo = float(resultado.x)
    custo_otimo = float(resultado.fun)

    progress_bar.progress(82, text="Gerando curva de taxa de custo: 82%")

    margem_inferior = max(lower_bound, delta_otimo * 0.25)
    margem_superior = min(upper_bound, max(delta_otimo * 2.75, delta_otimo + 1.00))

    if margem_superior <= margem_inferior:
        margem_inferior = lower_bound
        margem_superior = upper_bound

    deltas = np.linspace(margem_inferior, margem_superior, N_PONTOS_GRAFICO)
    custos = []

    for idx, delta in enumerate(deltas, start=1):
        custos.append(funcao_objetivo(float(delta), eta, beta_shape, lh, cf, cp, ci, cd_unit))
        pct = 82 + int((idx / N_PONTOS_GRAFICO) * 17)
        progress_bar.progress(min(pct, 99), text=f"Gerando gráfico: {min(pct, 99)}%")

    custos = np.array(custos, dtype=float)
    progress_bar.progress(100, text="Cálculo concluído: 100%")
    status_placeholder.caption("Otimização finalizada.")

    st.markdown("### Resultados")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Variável de decisão ótima, <em>Δ*</em></div>
                <div class="metric-value">{delta_otimo:.2f}</div>
                <div class="metric-footnote">Tempo entre inspeções</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Taxa de custo mínima</div>
                <div class="metric-value">{custo_otimo:.2f}</div>
                <div class="metric-footnote">Custo esperado por unidade de tempo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        status_text = "Convergiu" if resultado.success else "Verificar"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Status do otimizador</div>
                <div class="metric-value">{status_text}</div>
                <div class="metric-footnote">Avaliações: {resultado.nfev}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Curva da taxa de custo")
    fig = gerar_grafico(deltas, custos, delta_otimo, custo_otimo)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Ver detalhes numéricos"):
        st.write(
            {
                "Delta_otimo": round(delta_otimo, 2),
                "Taxa_de_custo_minima": round(custo_otimo, 2),
                "Convergiu": bool(resultado.success),
                "Avaliacoes_da_funcao": int(resultado.nfev),
            }
        )

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em **Executar otimização** para obter os resultados.")

st.markdown(
    """
    <br>
    <div class="small-note">
        Observação: a barra de carregamento apresenta uma estimativa aproximada do avanço computacional,
        pois o tempo de cada avaliação depende das integrais numéricas calculadas pelo modelo.
    </div>
    """,
    unsafe_allow_html=True,
)
