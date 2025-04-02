from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_layout(title: str, suptitle: str, x_label: str, y_label: str, ticksuffix: str = "", 
               template: str = "plotly_dark", purpose: str = "app") -> go.Layout:
    
    if purpose == "app":
        template = 'plotly_dark'
        tickfont_size = 12
        title_font_size = 14
        font_size = 20
        zoomable = False
    else:
        template = 'plotly_white'
        tickfont_size = 14
        title_font_size = 16
        font_size = 24
        zoomable = True
    
    return go.Layout(
            yaxis={"title": y_label, "ticksuffix": ticksuffix, 
                   'tickfont_size': tickfont_size, 'title_font': {'size': title_font_size}, 'fixedrange': zoomable},
            xaxis={"title": x_label, 'categoryorder':'array', 'categoryarray':['CPU','NPU','iGPU','GPU'], 
                   'tickfont_size': tickfont_size, 'title_font': {'size': title_font_size}, 'fixedrange': zoomable},
            title={
            "text": f"{title}<br><sup style='font-weight: normal'>{suptitle}</sup>",
            "font_size": font_size,
            "font_weight": "bold"},
            legend={
            "title": "<b>Device</b>"},
            dragmode="pan",
            template=template
            # paper_bgcolor="lightgrey",
        )


def create_figure(df, var: str, title: str, suptitle: str, x_label: str, 
                  y_label: str, ticksuffix: str = "", theme="plotly_white", purpose: str = 'app'):

    layout = get_layout(title, suptitle, x_label, y_label, ticksuffix, theme, purpose)
    if var == 'fps':
        df_plot = df[df['variable'] == var].sort_values(by='value')
    else:
        df_plot = df[df['variable'] == var].sort_values(by='value', ascending=False)
    fig = px.bar(df_plot, 
                 x='index', y='value', color='index', title="yolo12s", text='relative_change',
                 template=theme,
                 color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False, texttemplate='%{text:.2f}%')

    fig.update_layout(layout)
    
    
    return fig

def get_benchmarkings() -> pd.DataFrame:
    fname = Path('reports', "benchmarks", 'detection_benchmarks_yolo12s.csv')
    df_results = pd.read_csv(fname, index_col=0)

    df_results_long = df_results.T.reset_index().melt(id_vars=['index'])
    # df_results = df_results.set_index(df_results.columns[0])

    rel_change = df_results.T.apply(lambda x: ((x - x.loc['CPU']) / x.loc['CPU'] ) * 100)
    rel_change_long = rel_change.reset_index().melt(id_vars='index')
    df_results_long['relative_change'] = rel_change_long['value']
    return df_results_long


if __name__ == "__main__":
    
    df = get_benchmarkings()
    title = "Benchmark results - Frames per seconds" 
    suptitle = 'Performance Analysis: Frames per Second for Object Detection Task Across Different Devices using YOLO12s'

    fig_fps = create_figure(df, 'fps', title, suptitle, 'Device', "Frames per Second")
    fig_fps.show()

    fig_fps.write_image(Path('reports', 'figures', 'detection_benchmarks_yolo12s_fps.pdf'), width=600, format='pdf')
    fig_fps.write_html(Path('reports', 'figures', 'detection_benchmarks_yolo12s_fps.html'))

    title = "Benchmark results - Inference time" 
    suptitle = 'Performance Analysis: Inference Time for Object Detection Task Across Different Devices using YOLO12s'
    fig_inference = create_figure(df, 'inference_time', title, suptitle, "Device", "Time", 'ms')

    fig_inference.write_image(Path('reports', 'figures', 'detection_benchmarks_yolo12s_inference_time.pdf'), width=600, format='pdf')
    fig_inference.write_html(Path('reports', 'figures', 'detection_benchmarks_yolo12s_inference_time.html'))
    fig_inference.show()