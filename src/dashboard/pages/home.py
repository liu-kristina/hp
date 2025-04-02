import dash
from pathlib import Path
from src.dashboard.components.chatbot_button import get_popover_chatbot
from src.dashboard.components.title_section import create_title_section


# Import packages
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


dash.register_page(__name__, path='/', title="Home", name="Home", external_stylesheets=[dbc.themes.CYBORG])

img_src = Path("pages", "assets")

# Path(img_src, '1_HP_ZBook_Power.png').exists()

# Image section

images_section = html.Div([
        # Image 1
        html.Div([
            html.Img(src=dash.get_asset_url('1_HP_ZBook_Power.png'), style={'width': '300px','height': 'auto', 'margin': '30px'}),
            html.H4('ZBook Power'),
            #html.Br(),
            html.P('Intel Core U9, RTX 3000 Ada (8GB)', style={'margin-top': '5px'}),
        ], style={'display': 'inline-block', 'text-align': 'center'}),
        
        # Image 2
        html.Div([
            html.Img(src=dash.get_asset_url('2_HP_Firefly.png'), style={'width': '300px','height': 'auto', 'margin': '30px'}),
            html.H4('ZBook Firefly'),
            #html.Br(),
            html.P('Intel Core U7 (NPU), RTX A500', style={'margin-top': '5px'}),
        ], style={'display': 'inline-block', 'text-align': 'center'}),
        
        # Image 3
        #html.Div([
        #    html.Img(src=dash.get_asset_url('3_HP_ZBook_Ultra.png'), style={'width': '310px','height': 'auto', 'margin': '20px'}),
        #    html.H4('ZBook Ultra'),
            #html.Br(),
        #    html.P('AMD SOC, 128GB RAM with up to 96GB VRAM', style={'margin-top': '5px'}),
        #], style={'display': 'inline-block', 'text-align': 'center'})
        
    ], style={'text-align': 'center', 'margin-top': '20px'}
   )

# App description section
description_box = html.Div([
    dcc.Markdown("""
                **Welcome to the AI Performance Showcase**
                
                AI is taking over the digital world in a storm. As does the growing demand for computational
                power to run cutting-edge AI applications on local devices. To meet this demand,
                HP's offers their newest AI mobile workstations, sporting a 'Neural-Processing Unit' (NPU)
                specifically designed for this task.
                This app explores processor capabilities for mobile AI on HP mobile workstations. 
                 
                **Explore & Experiment**
                 
                - Perform question-answering with a tailored chatbot to answer all questions related to HP's products
                - Try out an image classification and object segmentation task
                - Benchmark AI tasks on different processing devices like the CPU, GPU or NPU.
                 
                """)
    ],
        style={
            'background-color': '#1b2631', 
            'color': '#FAF9F6',
            'border-radius': '10px',  # Rounded corners
            'padding': '20px',  # Padding around the content
            'margin-top': '20px',  # Top margin to separate from images
            'max-width': '800px',  # Max width to keep the box compact
            'margin-left': 'auto',
            'margin-right': 'auto'
    })

# Footer image
footer_image = html.Div([
    html.Img(src=dash.get_asset_url('4_footer.png'), style={'width': '150%', 'height': '40px','margin-top': '30px'})
])

layout = html.Div([
    create_title_section("AI Performance Showcase: CPU, GPU, and NPU on HP Mobile Workstations", 
                         "Experience how NPUs are shaping the future of mobile AI"),
    images_section,
    description_box,
    footer_image,
    get_popover_chatbot()
])

# Run the app
if __name__ == '__main__':
    # app.run(debug=True)
    None