from dash import Input, Output, callback, html


# Callback to display the uploaded image immediately.
@callback(
    Output("output-image-upload", "children", allow_duplicate=True),
    [Input("upload-image", "contents")],
    prevent_initial_call='initial_duplicate'
)
def update_uploaded_image(contents):
    if contents is None:
        return ""
    return html.Img(src=contents, className="uploaded-image", style={"maxWidth": "100%", "height": "auto"})
