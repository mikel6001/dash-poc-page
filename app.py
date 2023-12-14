import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import datetime as dt
import numpy as np
import plotly.graph_objs as go

import matplotlib
matplotlib.use('Agg')  # changed renderer to prevent "main thread is not in the main loop"
import shap
from dash.exceptions import PreventUpdate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from dash import html
from sklearn.tree import DecisionTreeClassifier

# Dash App Initialization
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
app.title = "POC Dashboard"

def_fig = px.density_heatmap()

def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Data Exploration Dashboard"),
            html.H6("Explore your data through visualization."),
            html.Div(
                id="intro",
                children=[
                    html.H4("Data Density: Heatmap"),
                    html.Hr(),
                    html.I("Select values for x, y and date range to vizualize data density.")
                ],
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[

            ### Dropdown for X value
            html.B("Select X value:"),
            dcc.Dropdown(
                id='dropdown-x',
                # options=[{"label": i, "value": i} for i in category_list],
                options=[
                    {'label': 'Age', 'value': 'age'},
                    {'label': 'Category', 'value': 'category'},
                    # {'label': 'Frequency of Purchases', 'value': 'fop'},
                    # {'label': 'Gender', 'value': 'gender'},
                    {'label': 'Location', 'value': 'location'},
                    {'label': 'Purchase Amount (USD)', 'value': 'pa'},
                    {'label': 'Season', 'value': 'season'},
                    # {'label': 'Review Rating', 'value': 'reviewr'}
                ],
            ),
            html.Br(),

            ### Dropdown for Y value
            html.B("Select Y value:"),
            dcc.Dropdown(
                id='dropdown-y',
                options=[
                    {'label': 'Age', 'value': 'age'},
                    {'label': 'Category', 'value': 'category'},
                    # {'label': 'Frequency of Purchases', 'value': 'fop'},
                    # {'label': 'Gender', 'value': 'gender'},
                    {'label': 'Location', 'value': 'location'},
                    {'label': 'Purchase Amount (USD)', 'value': 'pa'},
                    {'label': 'Season', 'value': 'season'},
                    # {'label': 'Review Rating', 'value': 'reviewr'}
                ],
            ),
            html.Br(),

            ### Date Range Input slicer
            html.Div(
                children=[
                    html.B('Start Date:'),
                    html.B('End Date:', style={'margin-left':'13%'}),
                ]
            ),
            html.Div(
                children=[
                    dcc.Input(id='start-date', type='text', placeholder='yyyy-mm-dd', value='2023-01-01', style={'width':'35%','display':'inline-block'}),
                    dcc.Input(id='end-date', type='text', placeholder='yyyy-mm-dd', value='2023-12-31', style={'width':'35%','display':'inline-block'}),
                    html.Button('Enter', id='submit-val', n_clicks=0, style={'display':'inline'})
                ]
            ),

            html.Br(),
            html.Hr(style={'width': '170%'}),
        ],style={'width': '105%'},
    )

# #--------------------------
# # Dropdown Options
# dropdown_options = [{'label': f'Dropdown Option {i}', 'value': f'option-{i}'} for i in range(1, 4)]
# #--------------------------

# Layout for Heatmap Tab
heatmap_tab = html.Div([
    html.Div(
    id="app-container",
    children=[
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),

        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # First Figure
                html.Div(
                    id="first_card",
                    children=[
                        html.H6("Data Density: Heatmap"),
                        html.Hr(),
                        dcc.Graph(id='den_heatmap', figure = def_fig)
                    ],
                ),
                html.Br(),
                html.Hr(style={'width': '200%'}),
            ],
        ),
    ],
)
])

################ Tab2
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

y = df['Outcome']
X = df.drop('Outcome', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LogisticRegression(random_state=42)
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

explainer1 = shap.Explainer(model1, X_train)

model2 = DecisionTreeClassifier(random_state=40)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Model 2 Accuracy: {accuracy2}")

explainer2 = shap.Explainer(model2, X_train, check_additivity=False)

# Layout for Shap Analysis Tab
shap_analysis_tab = html.Div([
    html.Div(
    id="app-container",
    children=[
        # Left column
        html.Div(
            id="left-column-shap",
            className="four columns",
            children=[
                html.H5("Dataset:"), 
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[
                        {'label': 'Antibiotics', 'value': 'Anti'},
                        {'label': 'Diabetes', 'value': 'Dia'}
                    ],
                    value='Anti',  # Default value
                    style={'width': '100%', 'margin-bottom': '10px'},
                    ),
                html.H5("Models"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                    {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                    {'label': 'Decision Tree', 'value': 'Decision Tree'}
                    ],
                    value='Logistic Regression',  # Default value
                    style={'width': '100%', 'margin-bottom': '10px'},
                ),
                html.H5("Depth"),
                dcc.Dropdown(
                    id='depth-dropdown',
                    options=[
                        {'label': 'All Features', 'value': 'all'},
                        {'label': '2 Features', 'value': 2},
                        {'label': '3 Features', 'value': 3},
                    ],
                    value='all',  # Default value
                    style={'width': '100%', 'margin-bottom': '10px'},
                ),
            ],
        )
    ]
    ),
        # Right column
        html.Div(
            id="right-column-shap",
            className="eight columns",
            children=[
                # First Figure
                html.Div(
                    id="first_card",
                    children=[
                        html.H6("Shap Analysis Plot"),
                        html.Hr(),
                        html.Div(id='shap-plot'),
                        html.Div(id='model-insights'),
                    ],
                ),
                html.Br(),
                html.Hr(style={'width': '200%'}),
            ],
        ),
    ],
)
################ Tab2

################ Tab3

ds = pd.read_csv(url)
X = ds.drop('Outcome', axis=1)
y = ds['Outcome']

# Model and Explainer 1
model_a = LogisticRegression(random_state=42)
model_a.fit(X, y)

explainer1 = shap.Explainer(model_a, X)

# Model and Explainer 2 
model_b = LogisticRegression(random_state=30)
model_b.fit(X, y)

explainer2 = shap.Explainer(model_b, X, check_additivity=False)

DEFAULT_PLOT_TYPE = 'bar'
DEFAULT_SELECTED_FEATURES = [X.columns[0]]
DEFAULT_DATASET = 'Dataset 1'

# Layout for Other Graphs Tab
other_graphs_tab = html.Div([
    html.Div(
        id="app-container-other-graphs",
        children=[
            html.Div(
                id="left-column-other-graphs",
                className="four columns",
                children=[
                    html.H5("Dataset:"),
                    dcc.Dropdown(
                        id='dataset-dropdown-other-graphs',
                        options=[
                            {'label': 'Dataset 1', 'value': 'Dataset 1'},
                            {'label': 'Dataset 2', 'value': 'Dataset 2'},
                        ],
                        value=DEFAULT_DATASET,
                        style={'width': '100%', 'margin-bottom': '10px'}
                    ),
                    html.H5("Graph Type:"),
                    dcc.Dropdown(
                        id='additional-plot-type-other-graphs',
                        options=[
                            {'label': 'Bar Chart', 'value': 'bar'},
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Scatter Plot', 'value': 'scatter'},
                        ],
                        value=DEFAULT_PLOT_TYPE,
                        style={'width': '100%', 'margin-bottom': '10px'}
                    ),
                    html.H5("Data Point:"),
                    dcc.Dropdown(
                        id='heatmap-feature-dropdown-other-graphs',
                        options=[{'label': col, 'value': col} for col in X.columns],
                        value=DEFAULT_SELECTED_FEATURES,
                        multi=True,
                        style={'width': '100%'}
                    ),
                    html.Div([
                        html.P("Description: "),
                        html.Div(id='description-text-other-graphs')
                    ]),
                ],
            ),
            html.Div(
                id="right-column-other-graphs",
                className="eight columns",
                children=[
                    dcc.Graph(id='additional-plot-other-graphs'),
                ],
            ),
        ],
    ),
])
################ Tab3

################ Tab4
# Layout for Tabs Tab
# tabs_tab = html.Div([])
################ Tab4

# App Layout with Tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs',
            value='tab-1', 
            children=[
                dcc.Tab(label='Heatmap', value='tab-1'),
                dcc.Tab(label='Shap Analysis', value='tab-2'),
                dcc.Tab(label='Other Graphs', value='tab-3')
                # dcc.Tab(label='', value='tab-4')
            ]),
    html.Div(id='tabs-content'),
])


# Callback to update tab content
@app.callback(Output('tabs-content', 'children'),
            [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return heatmap_tab
    elif tab == 'tab-2':
        return shap_analysis_tab
    elif tab == 'tab-3':
        return other_graphs_tab
    # elif tab == 'tab-4':
    #     return tabs_tab
    else:
        return html.Div('Error')

# Callback to update the heatmap based on the selected slicer value
@app.callback(
    Output('den_heatmap', 'figure'),
    [
        Input('dropdown-x', 'value'),
        Input('dropdown-y', 'value'),
        Input('submit-val', 'n_clicks'),
    ],
    [   
        dash.dependencies.State('start-date', 'value'),
        dash.dependencies.State('end-date', 'value')
    ]
)
def update_figure(selected_x, selected_y, n_clicks, start_date, end_date):
    if selected_x is None:
        return def_fig

    if selected_x and selected_y not in ('age', 'category', 'fop', 'gender', 'location', 'pa', 'season', 'reviewr'):
        return def_fig
    
    else:
        return return_figure(selected_x, selected_y, n_clicks, start_date, end_date)
    
    return def_fig


def return_figure(selected_x, selected_y, n_clicks, start_date, end_date):

    # Read data
    df = pd.read_csv('shopping_trends _dateadded.csv')
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    ## For color_continuous_scale
    scale_value = "blugrn_r"
    if selected_x == "age":
        scale_value = "ice_r"

    elif selected_x == "category":
        scale_value = "tropic_r"

    elif selected_x == "location":
        scale_value = "earth_r"

    elif selected_x == "gender":
        scale_value = "balance_r"

    elif selected_x == "season":
        scale_value = "tealrose_r"
    
    else:
        scale_value = "blugrn_r"

    # Heatmaps to be displayed
    ## Calls params() to convert the input to actual x and y values
    fig = px.density_heatmap(filtered_df, x=params(selected_x), y=params(selected_y), color_continuous_scale=(scale_value)) #, text_auto=True

    return fig

def params(val):
    parameters = {
        "category": "Category",
        "location": "Location",
        "pa": "Purchase Amount (USD)",
        "season": "Season",
        "age": "Age"
    }
    return parameters[val]

# Shap Analysis Callback
@app.callback(
    [Output('shap-plot', 'children'),
    Output('model-insights', 'children')],
    [Input('model-dropdown', 'value'),
    Input('dataset-dropdown', 'value'),
    Input('depth-dropdown', 'value'),],
)
def update_shap_plot(selected_model, selected_dataset, selected_depth):
    if selected_dataset == 'Dia':
        global img, encoded_img
        if selected_model is None:
            raise PreventUpdate
        
        print(f"Selected Model: {selected_model}")
        print(f"Selected Dataset: {selected_dataset}")

        if selected_model == 'Logistic Regression':
            model = model1 
            explainer = explainer1
        elif selected_model == 'Decision Tree':
            model = model2
            explainer = explainer2
        else:
            raise PreventUpdate

        shap_values = explainer.shap_values(X_test)
        
        if selected_depth == 'all':
            selected_features = np.arange(len(X.columns))
        else:
            selected_features = np.argsort(np.abs(shap_values.mean(axis=0)))[::-1][:int(selected_depth)]

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[:, selected_features], X_test.iloc[:, selected_features], feature_names=X.columns[selected_features], show=False)

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        encoded_img = base64.b64encode(img.read()).decode('ascii')
        plt.close()

        shap_plot = html.Div(
        html.Img(
        src=f'data:image/png;base64,{encoded_img}',
            style={'display': 'block', 'margin': 'auto', 'width': '100%', 'height': '100%'}
        ),
        style={'text-align': 'center'}
        )

        return shap_plot, None
    elif selected_dataset == 'Anti':
        url2 = "https://raw.githubusercontent.com/plotly/datasets/master/Antibiotics.csv"
        df2 = pd.read_csv(url2)

        y2 = df2[' Gram']
        X2 = df2.drop(['Bacteria', ' Gram'], axis=1)

        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
        
        global img2, encoded_img2
        if selected_model is None or selected_depth is None:
            raise PreventUpdate

        print(f"Selected Model: {selected_model}")
        print(f"Selected Depth: {selected_depth}")
        print(f"Selected Dataset: {selected_dataset}")

        if selected_model == 'Logistic Regression':
            shap_values = pd.read_csv('model_1.csv')
        elif selected_model == 'Decision Tree':
            shap_values = pd.read_csv('model_2.csv')
        else:
            raise PreventUpdate
            
        shap_values = pd.DataFrame(shap_values).to_numpy()

        if selected_depth == 'all':
            selected_features = np.arange(len(X2.columns))
        else:
            selected_features = np.argsort(np.abs(shap_values.mean(axis=0)))[::-1][:int(selected_depth)]

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[:, selected_features], X2_test.iloc[:, selected_features], feature_names=X2.columns[selected_features], show=False)

        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)

        encoded_img2 = base64.b64encode(img2.read()).decode('ascii')
        plt.close()

        shap_plot = html.Div(
        html.Img(
        src=f'data:image/png;base64,{encoded_img2}',
            style={'display': 'block', 'margin': 'auto', 'width': '100%', 'height': '100%'}
        ),
        style={'text-align': 'center'}
        )

        feature_importance = np.abs(shap_values.mean(axis=0))
        sorted_features = np.argsort(feature_importance)[::-1]
        most_important_feature = X2.columns[sorted_features[0]]
        least_important_feature = X2.columns[sorted_features[-1]]

        print("Length of X.columns:", len(X2.columns))

        model_insights = html.Div([
            html.H3("Model Insights"),
            html.P(f"The most important feature is {most_important_feature}, while the least important feature is {least_important_feature} according to {selected_model}."),
        ])    
        return shap_plot, model_insights
    else:
        PreventUpdate

@app.callback(
    [Output('additional-plot-other-graphs', 'figure'),
     Output('description-text-other-graphs', 'children')],
    [Input('dataset-dropdown-other-graphs', 'value'),
     Input('additional-plot-type-other-graphs', 'value'),
     Input('heatmap-feature-dropdown-other-graphs', 'value')]
)
def update_additional_plot_other_graphs(selected_dataset, additional_plot_type, selected_features):
    if additional_plot_type is None or selected_features is None:
        raise PreventUpdate

    if selected_dataset == 'Dataset 1':
        explainer = explainer1
    else:
        explainer = explainer2

    if additional_plot_type == 'bar':
        plot_type = 'bar'
    elif additional_plot_type == 'line':
        plot_type = 'line'
    elif additional_plot_type == 'scatter':
        plot_type = 'scatter'
    else:
        raise PreventUpdate

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)

    if shap_values.shape[0] != X.shape[0]:
        raise PreventUpdate

    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)

    selected_features_shap = [X.columns.get_loc(feature) for feature in selected_features]

    # Additional Plot
    fig_additional = generate_additional_plot(plot_type, selected_features_shap, shap_values, X)

    # Description Text
    description_text = generate_description_text(plot_type, selected_features)

    return fig_additional, description_text

def update_additional_plot_other_graphs(selected_dataset, additional_plot_type, selected_features):
    if additional_plot_type is None or selected_features is None:
        raise PreventUpdate

    if selected_dataset == 'Dataset 1':
        explainer = explainer1
    else:
        explainer = explainer2

    if additional_plot_type == 'bar':
        plot_type = 'bar'
    elif additional_plot_type == 'line':
        plot_type = 'line'
    elif additional_plot_type == 'scatter':
        plot_type = 'scatter'
    else:
        raise PreventUpdate

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)

    if shap_values.shape[0] != X.shape[0]:
        raise PreventUpdate

    if len(shap_values.shape) == 3:
        shap_values = np.mean(shap_values, axis=2)

    selected_features_shap = [X.columns.get_loc(feature) for feature in selected_features]

    # Additional Plot
    fig_additional = generate_additional_plot(plot_type, selected_features_shap, shap_values, X)

    # Description Text
    description_text = generate_description_text(plot_type, selected_features)

    return fig_additional, description_text

def generate_additional_plot(plot_type, selected_features, shap_values, X):
    # Additional Plot (Bar Chart, Line Chart, or Scatter Plot)
    fig_additional = go.Figure()

    for selected_feature in selected_features:
        selected_shap_values = shap_values[:, selected_feature]

        if plot_type == 'bar':
            # Use feature values as x-axis instead of constant X.iloc[:, selected_feature]
            fig_additional.add_trace(go.Bar(
                x=X.iloc[:, selected_feature],
                y=selected_shap_values,
                name=X.columns[selected_feature],
                marker=dict(color=px.colors.qualitative.Plotly[len(fig_additional.data) % len(px.colors.qualitative.Plotly)]),
                opacity=0.7,
            ))
        elif plot_type == 'line':
            # Use go.Line for line charts
            fig_additional.add_trace(go.Line(
                x=X.iloc[:, selected_feature],
                y=selected_shap_values,
                name=X.columns[selected_feature],
                marker=dict(color=px.colors.qualitative.Plotly[len(fig_additional.data) % len(px.colors.qualitative.Plotly)]),
            ))
        elif plot_type == 'scatter':
            fig_additional.add_trace(go.Scatter(
                x=X.iloc[:, selected_feature],
                y=selected_shap_values,
                mode='markers',
                name=X.columns[selected_feature],
                marker=dict(color=px.colors.qualitative.Plotly[len(fig_additional.data) % len(px.colors.qualitative.Plotly)]),
                opacity=0.7,
            ))

    fig_additional.update_layout(
        xaxis_title="Feature Values",
        yaxis_title="SHAP Values",
        legend_title="Features",
        title=f"{plot_type.capitalize()} Plot of SHAP Values for Selected Features",
        barmode='overlay' if plot_type == 'bar' else None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig_additional

def generate_description_text(plot_type, selected_features):
    # Generate description based on the plot type and selected features
    if plot_type == 'bar':
        description = f"The bar chart shows the impact of selected features ({', '.join(selected_features)}) on SHAP values."
    elif plot_type == 'line':
        description = f"The line chart illustrates how SHAP values change with varying feature values for ({', '.join(selected_features)})."
    elif plot_type == 'scatter':
        description = f"The scatter plot demonstrates the relationship between feature values and SHAP values for ({', '.join(selected_features)})."
    else:
        description = "Description not available for the selected plot type."

    return html.P(description)
    

if __name__ == '__main__':
    app.run_server(debug=True)
