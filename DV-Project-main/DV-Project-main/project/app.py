#!/usr/bin/env python
# coding: utf-8

# In[16]:


from dash import Dash, html, dcc

app = Dash(__name__)

app.layout = html.Div(style={'background-image': 'linear-gradient(to right, lightblue , darkblue)',  # Gradient background
                             'height': '100vh',  # Full height
                             'textAlign': 'center'},  # Center align text
                  children=[
                      html.H1('Welcome to Goalwise: Insight into the past 4 seasons of the premier league of soccer in Spain',
                              style={'color': 'white', 'paddingTop': '50px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),  # Title styling
                      
                      # Container for buttons
                      html.Div(
                          [
                              html.Div(dcc.Link('2020-2021 Season', href='/2020-21', className='button'), style={'padding': '15px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),
                              html.Div(dcc.Link('2021-22 Season', href='/2021-22', className='button'), style={'padding': '15px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),
                              html.Div(dcc.Link('2022-23 Season', href='/2022-23', className='button'), style={'padding': '15px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),
                              html.Div(dcc.Link('2023-24 Season', href='/2023-24', className='button'), style={'padding': '15px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),
                          ],
                          style={
                              'marginTop': '250px', 
                              'display': 'inline-block', 
                              'background-color': '#fff',  # Background color of the rectangle
                              'border-radius': '15px',  # Smooth edges
                              'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',  # Optional shadow for effect
                              'padding': '20px'
                          }
                      )
                  ])

# # Additional CSS to style the buttons
# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'  # Using Dash's default stylesheet for simplicity
# })

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, use_reloader=False)


# In[ ]:




