"""
NewsGuard Interactive Gradio Web Interface:
- Article Input (Title + Body)
- Visual Credibility Gauge (0-100 Score with dynamic color styling)
- SHAP Word Importance Visualization (Bar plot)
- Linguistic Metrics Breakdown (Readability, Sentiment, Style)
- Preloaded benchmark examples for 1-click evaluation
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from src.pipeline import NewsGuardPredictor

# Initialize predictor
predictor = None

def get_predictor_instance():
    global predictor
    if predictor is None:
        try:
            predictor = NewsGuardPredictor()
        except Exception as e:
            print(f"[Gradio] Notice loading predictor: {e}")
            predictor = None
    return predictor


# Example articles for 1-click testing
EXAMPLES = [
    [
        "Diplomats Reach Preliminary Agreement on Clean Energy Accord",
        "GENEVA — International delegates gathered on Thursday to finalize a historic cooperative framework "
        "addressing clean energy investments and cross-border grid resiliency. Representatives from 42 nations "
        "concluded three days of structured negotiations, agreeing on verification standards and financial disclosure "
        "rules. According to the lead commissioner, the formal ratification process will begin next month following "
        "review by legislative committees in each participating member nation."
    ],
    [
        "SHOCKING SECRET: Alien Microchips Found Inside Tap Water Everywhere!!",
        "You will NOT BELIEVE what government whistleblowers just leaked to us today! Secret globalist billionaires "
        "have been dumping mind-controlling nano-chips into public reservoirs across the country! Mainstream media "
        "is completely SILENT and censoring anyone who shares the truth! Share this immediately with everyone you know "
        "before they delete this post and shut down our servers! Wake up sheeple!!"
    ],
    [
        "Pundits Debate the Electoral Impact of Recent Budget Proposals",
        "WASHINGTON — The debate over proposed tax adjustments continues to dominate political commentary this week. "
        "While administration allies argue the initiative will foster capital formation and stabilize consumer prices, "
        "dissenting lawmakers warn of potential deficit expansion. Independent economic analysts noted that historical "
        "outcomes vary significantly depending on execution timing and overall monetary conditions."
    ]
]


def create_credibility_gauge_html(score: float, verdict: str, badge_color: str, label: str) -> str:
    """
    Renders a stunning HTML/CSS credibility gauge and scorecard.
    """
    color_map = {
        "green": ("#2ecc71", "#27ae60", "#e8f8f0"),
        "teal": ("#1abc9c", "#16a085", "#e8f8f5"),
        "yellow": ("#f39c12", "#d68910", "#fef9e7"),
        "orange": ("#e67e22", "#d35400", "#fdf2e9"),
        "red": ("#e74c3c", "#c0392b", "#fdedec")
    }
    primary, secondary, bg = color_map.get(badge_color, ("#3498db", "#2980b9", "#ebf5fb"))
    
    html = f"""
    <div style="background: {bg}; border: 2px solid {primary}; border-radius: 12px; padding: 20px; text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.06);">
        <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #555; font-weight: 700; margin-bottom: 6px;">
            CREDIBILITY SCORING ENGINE
        </div>
        <div style="font-size: 48px; font-weight: 800; color: {primary}; line-height: 1.1;">
            {score}<span style="font-size: 22px; font-weight: 500; color: #777;"> / 100</span>
        </div>
        <div style="display: inline-block; background: {primary}; color: white; font-weight: 700; font-size: 14px; padding: 6px 16px; border-radius: 20px; margin-top: 10px; letter-spacing: 0.5px;">
            VERDICT: {verdict.upper()} ({label.upper()} NEWS)
        </div>
        
        <!-- Visual Meter Bar -->
        <div style="background: #e0e0e0; border-radius: 10px; height: 14px; width: 85%; margin: 18px auto 6px auto; overflow: hidden; position: relative;">
            <div style="background: linear-gradient(90deg, #e74c3c 0%, #f39c12 40%, #1abc9c 70%, #2ecc71 100%); height: 100%; width: 100%; position: absolute;"></div>
            <div style="background: white; position: absolute; right: 0; height: 100%; width: {100 - score}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 85%; margin: 0 auto; font-size: 11px; color: #888; font-weight: 600;">
            <span>0 (Fake)</span>
            <span>50 (Uncertain)</span>
            <span>100 (Verified Real)</span>
        </div>
    </div>
    """
    return html


def plot_top_features_chart(top_features: list[dict]):
    """
    Generates a horizontal bar chart showing top SHAP features and their directional pull.
    """
    if not top_features:
        fig, ax = plt.subplots(figsize=(7, 2.5), dpi=150)
        ax.text(0.5, 0.5, "No specific feature attributions available.", ha="center", va="center")
        ax.axis("off")
        return fig

    feats = [item["feature"] for item in top_features][::-1]
    vals = [item.get("shap_value", item.get("importance_percentage", 0.05)) for item in top_features][::-1]
    colors = ["#2ecc71" if item["direction"] == "Supports Real" else "#e74c3c" for item in top_features][::-1]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    bars = ax.barh(feats, vals, color=colors, height=0.55, edgecolor="none")
    ax.axvline(0, color="#333", linestyle="--", alpha=0.5, lw=1)
    ax.set_title("Top Linguistic & Lexical Indicators (SHAP Feature Attribution)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Impact on Credibility (+ Supports Real / - Supports Fake)", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def analyze_article(title: str, text: str):
    """
    Analysis handler triggered by user submission in the Gradio UI.
    """
    p = get_predictor_instance()
    if p is None:
        return (
            "<div style='color:red; font-weight:bold;'>Error: Model artifacts not found. Please run the training pipeline first.</div>",
            None,
            "Model not loaded.",
            {},
            "Please train the model."
        )

    clean_text = text.strip()
    if not clean_text:
        return (
            "<div style='color:red; font-weight:bold;'>Error: Please enter article text to analyze.</div>",
            None,
            "Input error.",
            {},
            "Empty text."
        )

    try:
        res = p.predict(text=clean_text, title=title.strip())
        gauge_html = create_credibility_gauge_html(
            res["credibility_score"],
            res["verdict"],
            res["badge_color"],
            res["label"]
        )
        chart_fig = plot_top_features_chart(res["top_features"])

        metrics_display = {
            "Flesch Reading Ease": f"{res['linguistic_metrics']['flesch_reading_ease']} / 100",
            "Flesch-Kincaid Grade": f"Grade {res['linguistic_metrics']['flesch_kincaid_grade']}",
            "Sentiment Polarity": f"{res['linguistic_metrics']['sentiment_polarity']} (-1 to +1)",
            "Sentiment Subjectivity": f"{res['linguistic_metrics']['sentiment_subjectivity']} (0 to 1)",
            "Word Count": f"{int(res['linguistic_metrics']['word_count'])} words",
            "Exclamation Frequency": f"{res['linguistic_metrics']['exclamation_ratio']:.4f}",
            "Uppercase Ratio": f"{res['linguistic_metrics']['uppercase_ratio']:.4f}"
        }

        return (
            gauge_html,
            chart_fig,
            res["summary"],
            metrics_display,
            res["verdict"]
        )
    except Exception as e:
        return (
            f"<div style='color:red;'>Inference Error: {str(e)}</div>",
            None,
            f"Error: {str(e)}",
            {},
            "Error"
        )


def build_gradio_app():
    """Builds and configures the Gradio Blocks interface."""
    custom_css = """
    .gradio-container { max-width: 1050px !important; margin: auto; }
    #header-box { text-align: center; margin-bottom: 20px; }
    """
    with gr.Blocks(title="NewsGuard — Credibility Scoring", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(elem_id="header-box"):
                gr.Markdown(
                    """
                    # 🛡️ NewsGuard
                    ### NLP-Powered Fake News Detection & Credibility Scoring System
                    *Classifies news articles, computes calibrated 0–100 credibility scores, and visualizes linguistic and SHAP explainability indicators.*
                    """
                )

        with gr.Row():
            with gr.Column(scale=5):
                title_input = gr.Textbox(
                    label="Article Headline / Title (Optional)",
                    placeholder="e.g., Summit concludes with historic multilateral climate framework...",
                    lines=1
                )
                text_input = gr.Textbox(
                    label="Article Body Text (Required)",
                    placeholder="Paste full news story, claim, or press release here...",
                    lines=10
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Credibility", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[title_input, text_input],
                    label="Quick Test Benchmark Articles"
                )

            with gr.Column(scale=5):
                gauge_output = gr.HTML(label="Credibility Score")
                verdict_status = gr.Textbox(label="Verdict Summary", interactive=False)
                chart_output = gr.Plot(label="SHAP Linguistic Attributions")
                narrative_output = gr.Textbox(label="Detailed Explanation", lines=3, interactive=False)
                metrics_output = gr.JSON(label="Extracted Linguistic & Statistical Metrics")

        # Hook events
        analyze_btn.click(
            fn=analyze_article,
            inputs=[title_input, text_input],
            outputs=[gauge_output, chart_output, narrative_output, metrics_output, verdict_status]
        )
        clear_btn.click(
            fn=lambda: ("", "", "", None, "", {}, ""),
            outputs=[title_input, text_input, gauge_output, chart_output, narrative_output, metrics_output, verdict_status]
        )

    return demo


if __name__ == "__main__":
    demo = build_gradio_app()
    port = int(os.environ.get("GRADIO_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
