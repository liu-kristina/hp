from dash import html

def create_title_section(title_text, subtitle_text):
    return html.Div([
        # HP logo
        html.Img(src='/assets/0_HP_logo.png', 
                 style={'width': '80px', 
                        'height': 'auto', 
                        'margin-right': '20px'}),  
        
        # Title & Subtitle
        html.Div([
            html.H1(title_text, 
                    style={'color': '#004ad8', 'margin': '5px', 'font-weight': 'bold', 'font-size': '45px'}),
            html.H3(subtitle_text, 
                    style={'margin-top': '5px', 'font-size': '35px'}),
                    html.Hr()
        ]),

    # Styling the flex box
    ], style={'display': 'flex',  
              'align-items': 'center', 
              'justify-content': 'flex-start', 
              'margin-bottom': '20px',
              'padding-left': '5px',
              'padding-top': '5px',
              'height': '150px',
              'width': '100%',
            #   'position': 'fixed',  # Fix at the top
              'top': '0',  # Position at the top of the page
             'z-index': '1000'  # Ensure it stays on top of other elements })
    })