from dash import html, dcc, callback, Input, Output

# Upload & Control Section
upload_section = html.Div(
    className="upload-section", 
    children=[
        dcc.Upload(
            id="upload-image-detect",
            children=html.Div([
                "Drag and Drop or ", html.A("Select an Image File")
            ]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px"
            },
            multiple=False
        ),
        # Display the uploaded image immediately
        html.Div(id="output-image-upload-detect", className="output-image", style={"margin": "10px"}),
        html.Div(className="dropdown-section", children=[
            html.Label("Select Processor:"),
            dcc.Dropdown(
                id="processor-dropdown-detect",
                options=[
                    {"label": "CPU", "value": "CPU"},
                    {"label": "GPU", "value": "GPU"},
                    {"label": "NPU", "value": "NPU"},
                ],
                value="CPU",
                clearable=False
            )
        ], style={"width": "50%", "margin": "10px"}),
        # Detect button
        html.Button("Detect", id="detect-button", n_clicks=0, className="btn btn-primary", style={"margin": "10px"}),
    ]
)

# Callback to display the uploaded image immediately.
@callback(
    Output("output-image-upload-detect", "children"),
    [Input("upload-image-detect", "contents")]
)
def update_uploaded_image_detect(contents):
    if contents is None:
        return ""
    return html.Img(src=contents, className="uploaded-image", style={"maxWidth": "100%", "height": "auto"})

def get_upload_form(id: str = "upload-image-detect"):
    return html.Div(
    className="upload-section", 
    children=[
        dcc.Upload(
            id=id,
            children=html.Div([
                "Drag and Drop or ", html.A("Select an Image File")
            ]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px"
            },
            multiple=False
        ),
        # Display the uploaded image immediately
        html.Div(id="output-image-upload-detect", className="output-image", style={"margin": "10px"}),
        html.Div(className="dropdown-section", children=[
            html.Label("Select Processor:"),
            dcc.Dropdown(
                id="processor-dropdown-detect",
                options=[
                    {"label": "CPU", "value": "CPU"},
                    {"label": "GPU", "value": "GPU"},
                    {"label": "NPU", "value": "NPU"},
                ],
                value="CPU",
                clearable=False
            )
        ], style={"width": "50%", "margin": "10px"}),
        # Detect button
        html.Button("Detect", id="detect-button", n_clicks=0, className="btn btn-primary", style={"margin": "10px"}),
    ]
)