"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.
s
For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""

# TODO: Settings button

from pathlib import Path
import os
import dash
import warnings
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
# from flask_caching import Cache


from dashboard.components.sidebar import create_sidebar
from src.dashboard.components.chatbot_button import get_popover_chatbot
from src.dashboard.callbacks import callbacks

# Caching for global store


# from pages.resource_monitor import create_page

warnings.filterwarnings("ignore", category=DeprecationWarning)

assets_src = Path("src", "dashboard", "components", "assets")
asst_path = os.path.join(os.getcwd(), "assets")
assets_folder = "dashboard/assets"

app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP], 
                use_pages=True,
                pages_folder="dashboard/pages/",
                assets_folder=assets_folder,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                suppress_callback_exceptions=True
                )
app.config.suppress_callback_exceptions = True

# CACHE_CONFIG  = {
#     "DEBUG": True,          # some Flask specific configs
#     "CACHE_TYPE": "FileSystemCache ",  # Flask-Caching related configs
#     "CACHE_DEFAULT_TIMEOUT": 300
# }
# server = app.server
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)



# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = create_sidebar(assets=assets_src)

content = html.Div(id="page-content", style=CONTENT_STYLE)

# app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


app.layout = html.Div([
    dash.page_container,
    dcc.Store(id="global-store", storage_type="local", data={}),
    sidebar,
    get_popover_chatbot()
], style=CONTENT_STYLE)


# @app.callback(
#     Output("sidebar", "className"),
#     [Input("sidebar-toggle", "n_clicks")],
#     [State("sidebar", "className")],
# )
# def toggle_classname(n, classname):
#     if n and classname == "":
#         return "collapsed"
#     return ""


# @app.callback(
#     Output("collapse", "is_open"),
#     [Input("navbar-toggle", "n_clicks")],
#     [State("collapse", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open


if __name__ == "__main__":
    app.run(port=8888, debug=True)
